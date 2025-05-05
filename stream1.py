import fitz  # PyMuPDF
import pdfplumber
import camelot
import pandas as pd
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import os
import re
import json
import numpy as np
import multiprocessing
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore", category=UserWarning, module='pdfminer')

# --- Config ---
PDF_PATH = "sample.pdf"
OCR_OUTPUT_DIR = "ocr_outputs"
TARGET_HEADINGS = {
    "consolidated statements of operations": "income_statement",
    "consolidated statements of income": "income_statement",
    "consolidated balance sheets": "balance_sheet",
    "consolidated statements of cash flows": "cash_flows",
    "consolidated statements of comprehensive income": "comprehensive_income",
    "consolidated statements of stockholders' equity": "equity_statement"
}

ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

# --- Utility: Table Validation ---
def is_valid_table(df: pd.DataFrame) -> bool:
    if df.empty or df.shape[0] < 2 or df.shape[1] < 2:
        return False

    # Remove rows that are entirely or mostly empty
    df = df.dropna(how='all')
    df = df.loc[df.apply(lambda row: row.count(), axis=1) >= 2]

    if df.empty:
        return False

    mode_cols = df.apply(lambda row: len(row.dropna()), axis=1).mode()[0]
    consistent_rows = sum(df.apply(lambda row: len(row.dropna()) == mode_cols, axis=1))
    if consistent_rows / len(df) < 0.4:
        return False

    # Require presence of at least two keywords in different rows (not just header)
    keywords = ["revenue", "income", "net", "cash", "liabilities", "assets", "equity", "operations", "shares"]
    lower_df = df.astype(str).applymap(lambda x: x.lower())
    keyword_hits = sum(any(k in cell for k in keywords) for row in lower_df.itertuples(index=False) for cell in row)

    if keyword_hits < 2:
        return False

    return True

# --- Step 1: Locate Matching Pages ---
def locate_financial_pages(pdf_path, font_size_threshold_ratio=0.8):  # adjusted threshold
    doc = fitz.open(pdf_path)
    matched_pages = []

    for page_num, page in enumerate(doc):
        text_instances = page.get_text("dict")["blocks"]
        all_font_sizes = []
        heading_instances = []

        for block in text_instances:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        span_text = span["text"].strip().lower()
                        span_font_size = span["size"]
                        all_font_sizes.append(span_font_size)

                        for heading, label in TARGET_HEADINGS.items():
                            if heading in span_text:
                                heading_instances.append({
                                    "heading": heading,
                                    "label": label,
                                    "font_size": span_font_size,
                                    "page_number": page_num
                                })

        if not heading_instances or not all_font_sizes:
            continue

        largest_font_size = np.max(all_font_sizes)
        threshold = largest_font_size * font_size_threshold_ratio

        for instance in heading_instances:
            if instance["font_size"] >= threshold:
                matched_pages.append(instance)
            else:
                print(f"⚠️ Skipping page {page_num+1} ({instance['label']}): heading font size too small.")

    return matched_pages

# --- Step 2: Structured Table Extraction (Camelot First, fallback to pdfplumber) ---
def extract_table(pdf_path, page_num):
    try:
        tables = camelot.read_pdf(pdf_path, pages=str(page_num + 1), flavor='stream')
        print(f"⚠️ Camelot tables {tables}")
        if tables and tables.n > 0:
            df = tables[0].df
            df = df.replace('', np.nan).dropna(how='all')
            df.dropna(axis=1, how='all', inplace=True)
            if is_valid_table(df):
                print(f"✅ Camelot extracted valid table on page {page_num+1} (shape: {df.shape})")
                return df
            else:
                print(f"⚠️ Camelot found table on page {page_num+1}, but it's invalid")
    except Exception as e:
        print(f"⚠️ Camelot failed on page {page_num+1}: {e}")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num]
            tables = page.extract_tables()

            if not tables:
                print(f"⚠️ pdfplumber found no tables on page {page_num+1}")
                raise ValueError("No tables found")

            combined_rows = []
            for table in tables:
                combined_rows.extend(table)

            df = pd.DataFrame(combined_rows)
            df.dropna(how='all', inplace=True)
            df.dropna(axis=1, how='all', inplace=True)

            if is_valid_table(df):
                print(f"✅ pdfplumber extracted valid table on page {page_num+1} (shape: {df.shape})")
                return df
            else:
                print(f"⚠️ pdfplumber table invalid on page {page_num+1}, shape: {df.shape}")
                raise ValueError("Invalid table")
    except Exception as e:
        print(f"⚠️ Falling back to OCR for page {page_num+1} due to error: {e}")
        return pd.DataFrame()

# --- Step 3: Fallback OCR ---
def fallback_ocr(pdf_path, page_number, label):
    os.makedirs(OCR_OUTPUT_DIR, exist_ok=True)
    image = convert_from_path(pdf_path, first_page=page_number+1, last_page=page_number+1)[0]
    image_path = os.path.join(OCR_OUTPUT_DIR, f"{label}_page_{page_number+1}.png")
    image.save(image_path)

    print(f"🔁 Fallback OCR on page {page_number+1} ({label})")
    result = ocr_engine.ocr(image_path, cls=True)
    lines = [line[1][0] for line in result[0]]

    rows = []
    for line in lines:
        parts = re.split(r'\s{2,}|\t', line.strip())
        if len(parts) >= 2:
            rows.append(parts)

    df = pd.DataFrame(rows)
    if not df.empty:
        csv_path = os.path.join(OCR_OUTPUT_DIR, f"{label}_page_{page_number+1}_ocr.csv")
        df.to_csv(csv_path, index=False, header=False)
        print(f"✅ OCR table saved: {csv_path} (shape: {df.shape})")
    else:
        print(f"⚠️ OCR found no valid rows on page {page_number+1}")

# --- Step 4: Process Pages ---
def process_label_group(args):
    label, pages, pdf_path = args
    combined_df = pd.DataFrame()

    for pg in pages:
        print(f"\n📊 Processing page {pg+1} ({label}):")
        df = extract_table(pdf_path, pg)

        if df.empty or df.shape[0] <= 3:
            print(f"⚠️ Skipping page {pg+1} due to empty or invalid table")
            fallback_ocr(pdf_path, pg, label)
        else:
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    if not combined_df.empty:
        out_path = os.path.join(OCR_OUTPUT_DIR, f"{label}_combined.csv")
        combined_df.to_csv(out_path, index=False, header=False)
        print(f"✅ Structured tables saved to {out_path} (shape: {combined_df.shape})")
    else:
        print(f"⚠️ No structured tables for {label}. Relying on OCR outputs.")

# --- Postprocess to JSON ---
def postprocess_tables_to_json():
    output = defaultdict(list)
    for fname in os.listdir(OCR_OUTPUT_DIR):
        if not fname.endswith(".csv"):
            continue

        label = fname.split("_")[0]
        csv_path = os.path.join(OCR_OUTPUT_DIR, fname)
        df = pd.read_csv(csv_path, header=None)

        if df.empty:
            continue

        df.columns = df.iloc[0]
        df = df[1:]

        year_cols = [col for col in df.columns if re.search(r"20\d{2}", str(col)) or "202" in str(col)]
        for _, row in df.iterrows():
            key = str(row.iloc[0])
            values = {str(year): str(row.get(year, "NaN")) for year in year_cols}
            output[label].append({key: values})

    json_path = os.path.join(OCR_OUTPUT_DIR, "clean_output.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✅ Cleaned JSON saved to {json_path}")

# --- Main Pipeline ---
def main(pdf_path):
    # Clear output folder if it exists
    if os.path.exists(OCR_OUTPUT_DIR):
        for f in os.listdir(OCR_OUTPUT_DIR):
            os.remove(os.path.join(OCR_OUTPUT_DIR, f))
    else:
        os.makedirs(OCR_OUTPUT_DIR, exist_ok=True)
    matched_pages = locate_financial_pages(pdf_path, font_size_threshold_ratio=0.8)
    groups = defaultdict(list)
    for m in matched_pages:
        groups[m['label']].append(m['page_number'])

    tasks = [(label, pages, pdf_path) for label, pages in groups.items()]

    os.makedirs(OCR_OUTPUT_DIR, exist_ok=True)  # Ensure output directory exists

    with multiprocessing.Pool(len(tasks)) as pool:
        pool.map(process_label_group, tasks)

    postprocess_tables_to_json()

if __name__ == "__main__":
    main(PDF_PATH)
