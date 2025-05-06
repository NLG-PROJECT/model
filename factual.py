import fitz  # PyMuPDF
import pdfplumber
import camelot
import pandas as pd
# from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import os
import re
import json
import numpy as np
import multiprocessing
import warnings
from collections import defaultdict
from utils.constants import PDF_PATH, OCR_OUTPUT_DIR, TARGET_HEADINGS
import asyncio

warnings.filterwarnings("ignore", category=UserWarning, module='pdfminer')




# ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

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
def locate_financial_pages(pdf_path, font_size_threshold_ratio=0.8):  # removed async
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
                print(f"Skipping page {page_num+1} ({instance['label']}): heading font size too small.")

    return matched_pages

# --- Step 2: Structured Table Extraction (Camelot First, fallback to pdfplumber) ---
async def extract_table(pdf_path, page_num):
    try:
        tables = camelot.read_pdf(pdf_path, pages=str(page_num + 1), flavor='stream')
        print(f"Camelot tables {tables}")
        if tables and tables.n > 0:
            df = tables[0].df
            df = df.replace('', np.nan).dropna(how='all')
            df.dropna(axis=1, how='all', inplace=True)
            if is_valid_table(df):
                print(f"Camelot extracted valid table on page {page_num+1} (shape: {df.shape})")
                return df
            else:
                print(f"Camelot found table on page {page_num+1}, but it's invalid")
    except Exception as e:
        print(f"Camelot failed on page {page_num+1}: {e}")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num]
            tables = page.extract_tables()

            if not tables:
                print(f"pdfplumber found no tables on page {page_num+1}")
                raise ValueError("No tables found")

            combined_rows = []
            for table in tables:
                combined_rows.extend(table)

            df = pd.DataFrame(combined_rows)
            df.dropna(how='all', inplace=True)
            df.dropna(axis=1, how='all', inplace=True)

            if is_valid_table(df):
                print(f"pdfplumber extracted valid table on page {page_num+1} (shape: {df.shape})")
                return df
            else:
                print(f"pdfplumber table invalid on page {page_num+1}, shape: {df.shape}")
                raise ValueError("Invalid table")
    except Exception as e:
        print(f"Falling back to OCR for page {page_num+1} due to error: {e}")
        return pd.DataFrame()

# --- Step 3: Fallback OCR ---
# def fallback_ocr(pdf_path, page_number, label):
#     os.makedirs(OCR_OUTPUT_DIR, exist_ok=True)
#     image = convert_from_path(pdf_path, first_page=page_number+1, last_page=page_number+1)[0]
#     image_path = os.path.join(OCR_OUTPUT_DIR, f"{label}_page_{page_number+1}.png")
#     image.save(image_path)

#     print(f"Fallback OCR on page {page_number+1} ({label})")
#     result = ocr_engine.ocr(image_path, cls=True)
#     lines = [line[1][0] for line in result[0]]

#     rows = []
#     for line in lines:
#         parts = re.split(r'\s{2,}|\t', line.strip())
#         if len(parts) >= 2:
#             rows.append(parts)

#     df = pd.DataFrame(rows)
#     if not df.empty:
#         csv_path = os.path.join(OCR_OUTPUT_DIR, f"{label}_page_{page_number+1}_ocr.csv")
#         df.to_csv(csv_path, index=False, header=False)
#         print(f"OCR table saved: {csv_path} (shape: {df.shape})")
#     else:
#         print(f"OCR found no valid rows on page {page_number+1}")

# --- Step 4: Process Pages ---
async def process_label_group(args):
    label, pages, pdf_path = args
    combined_df = pd.DataFrame()

    for pg in pages:
        print(f"\nProcessing page {pg+1} ({label}):")
        df = await extract_table(pdf_path, pg)  # Make sure to await this

        if df.empty or df.shape[0] <= 3:
            print(f"Skipping page {pg+1} due to empty or invalid table")
        else:
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    if not combined_df.empty:
        out_path = os.path.join(OCR_OUTPUT_DIR, f"{label}_combined.csv")
        combined_df.to_csv(out_path, index=False, header=False)
        print(f"Structured tables saved to {out_path} (shape: {combined_df.shape})")
    else:
        print(f"No structured tables for {label}. Relying on OCR outputs.")

# --- Postprocess to JSON ---
# --- Custom Handler: Income Statement JSON ---
# --- Custom Handler: Income Statement JSON ---
async def process_income_statement_to_json(csv_path):
    json_path = os.path.join(OCR_OUTPUT_DIR, "income_statement_clean.json")

    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
    else:
        data = {}

    df = pd.read_csv(csv_path, header=None)
    df = df.dropna(how='all').reset_index(drop=True)

    header_row_idx = None
    for idx, row in df.iterrows():
        if sum(bool(re.search(r"20\d{2}", str(cell))) for cell in row) >= 2:
            header_row_idx = idx
            break

    if header_row_idx is None:
        print("Could not locate header row with years.")
        return

    header = df.iloc[header_row_idx].tolist()

    normalized_header = []
    year_to_full_date = {}
    for col in header:
        match = re.search(r"20\d{2}", str(col))
        if match:
            year = match.group(0)
            normalized_header.append(year)
            year_to_full_date[year] = str(col).strip()
        else:
            normalized_header.append(str(col).strip())

    existing_dates = data.get("dates", {})
    existing_dates.update(year_to_full_date)
    data["dates"] = existing_dates

    df_data = df.iloc[header_row_idx + 1:].copy()
    df_data.columns = normalized_header
    df_data = df_data.dropna(how='all')

    year_cols = [col for col in df_data.columns if re.fullmatch(r"20\d{2}", col)]
    item_col = [col for col in df_data.columns if col not in year_cols][0]

    result = []
    for _, row in df_data.iterrows():
        entry = {"item": str(row[item_col]).strip()}
        for year in year_cols:
            raw_val = str(row.get(year, "")).replace("$", "").replace(",", "").replace("(", "-").replace(")", "").strip()
            try:
                val = float(raw_val) if raw_val not in ["", "-", "—"] else None
            except:
                val = None
            entry[year] = val
        result.append(entry)

    data.setdefault("ConsolidatedStatementsOfIncomeOrComprehensiveIncome", []).extend(result)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Income statement JSON saved to {json_path}")



# --- Custom Handler: Comprehensive Income JSON ---
async def process_comprehensive_income_to_json(csv_path):
    try:
        with open(os.path.join(OCR_OUTPUT_DIR, "income_statement_clean.json")) as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Income statement JSON not found. Cannot merge.")
        return

    df = pd.read_csv(csv_path, header=None)
    df = df.dropna(how='all').reset_index(drop=True)

    header_row_idx = None
    for idx, row in df.iterrows():
        if sum(bool(re.search(r"20\d{2}", str(cell))) for cell in row) >= 2:
            header_row_idx = idx
            break

    if header_row_idx is None:
        print("Could not locate header row in comprehensive income.")
        return

    header = df.iloc[header_row_idx].tolist()

    normalized_header = []
    for col in header:
        match = re.search(r"20\d{2}", str(col))
        if match:
            year = match.group(0)
            normalized_header.append(year)
        else:
            normalized_header.append(str(col).strip())

    df_data = df.iloc[header_row_idx + 1:].copy()
    df_data.columns = normalized_header
    df_data = df_data.dropna(how='all')

    year_cols = [col for col in df_data.columns if re.fullmatch(r"20\d{2}", col)]
    item_col = [col for col in df_data.columns if col not in year_cols][0]

    additional_entries = []
    for _, row in df_data.iterrows():
        entry = {"item": str(row[item_col]).strip()}
        for year in year_cols:
            raw_val = str(row.get(year, "")).replace("$", "").replace(",", "").replace("(", "-").replace(")", "").strip()
            try:
                val = float(raw_val) if raw_val not in ["", "-", "—"] else None
            except:
                val = None
            entry[year] = val
        additional_entries.append(entry)

    data.setdefault("ConsolidatedStatementsOfIncomeOrComprehensiveIncome", []).extend(additional_entries)

    json_path = os.path.join(OCR_OUTPUT_DIR, "income_statement_clean.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Comprehensive income data merged and saved to {json_path}")




# --- Custom Handler: Cash Flow JSON ---
async def process_cashflow_to_json(csv_path):
    json_path = os.path.join(OCR_OUTPUT_DIR, "cashflow_statement_clean.json")
    data = {}

    df = pd.read_csv(csv_path, header=None)
    df = df.dropna(how='all').reset_index(drop=True)

    header_row_idx = None
    for idx, row in df.iterrows():
        if sum(bool(re.search(r"20\d{2}", str(cell))) for cell in row) >= 2:
            header_row_idx = idx
            break

    if header_row_idx is None:
        print("Could not locate header row in cashflow.")
        return

    header = df.iloc[header_row_idx].tolist()

    normalized_header = []
    year_to_full_date = {}
    for col in header:
        match = re.search(r"20\d{2}", str(col))
        if match:
            year = match.group(0)
            normalized_header.append(year)
            year_to_full_date[year] = str(col).strip()
        else:
            normalized_header.append(str(col).strip())

    data["dates"] = year_to_full_date

    df_data = df.iloc[header_row_idx + 1:].copy()
    df_data.columns = normalized_header
    df_data = df_data.dropna(how='all')

    year_cols = [col for col in df_data.columns if re.fullmatch(r"20\d{2}", col)]
    item_col = [col for col in df_data.columns if col not in year_cols][0]

    result = []
    for _, row in df_data.iterrows():
        entry = {"item": str(row[item_col]).strip()}
        for year in year_cols:
            raw_val = str(row.get(year, "")).replace("$", "").replace(",", "").replace("(", "-").replace(")", "").strip()
            try:
                val = float(raw_val) if raw_val not in ["", "-", "—"] else None
            except:
                val = None
            entry[year] = val
        result.append(entry)

    data["ConsolidatedStatementsOfCashFlows"] = result

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Cash flow statement JSON saved to {json_path}")

def process_balance_sheet_to_json(csv_path):
    df = pd.read_csv(csv_path, header=None)
    if df.empty:
        print(f"Balance sheet CSV at {csv_path} is empty.")
        return

    year_row_idx = None
    for i, row in df.iterrows():
        year_candidates = [str(cell) for cell in row if re.search(r"20\d{2}", str(cell))]
        if len(year_candidates) >= 2:
            year_row_idx = i
            break

    if year_row_idx is None:
        print(f"Could not find year header in balance sheet.")
        return

    header_row = df.iloc[year_row_idx].fillna("").astype(str)
    year_cols = {}
    for idx, cell in enumerate(header_row):
        match = re.search(r"(20\d{2})", cell)
        if match:
            year = match.group(1)
            year_cols[year] = idx

    print(f"📌 Detected year columns: {year_cols}")

    main_data = []
    assets_group = []
    liabilities_group = []
    derivatives_group = []

    current_group = None

    for i in range(year_row_idx + 1, len(df)):
        row = df.iloc[i]
        first_col = str(row[0]).strip() if pd.notna(row[0]) else ""
        second_col = str(row[1]).strip() if pd.notna(row[1]) else ""
        item = first_col if not first_col.startswith("$") and len(first_col) > 1 else second_col

        if not item:
            continue

        item_lower = item.lower()
        if "asset" in item_lower:
            current_group = "assets"
        elif "liabilit" in item_lower:
            current_group = "liabilities"
        elif "derivative" in item_lower:
            current_group = "derivatives"

        if item_lower in ["assets", "liabilities", "stockholders' equity"]:
            continue

        entry = {"item": item}
        for year, idx in year_cols.items():
            try:
                val = row[idx] if idx < len(row) else None
                entry[year] = try_float(val)
            except:
                entry[year] = None

        main_data.append(entry)
        if current_group == "assets":
            assets_group.append(entry)
        elif current_group == "liabilities":
            liabilities_group.append(entry)
        elif current_group == "derivatives":
            derivatives_group.append(entry)

    balance_output = {
        "balance_sheet": {
            "main": main_data,
            "assets_group": assets_group,
            "liabilities_group": liabilities_group,
            "derivatives_group": derivatives_group
        }
    }

    json_path = os.path.join(OCR_OUTPUT_DIR, "balance_sheet_clean.json")
    with open(json_path, "w") as f:
        json.dump(balance_output, f, indent=2)

    print(f"Balance sheet JSON saved to {json_path}")

async def process_equity_statement_to_json(csv_path):
    df = pd.read_csv(csv_path, header=None)
    if df.empty:
        print(f"Equity CSV at {csv_path} is empty.")
        return

    # Try to extract header row
    header_row_idx = None
    for i, row in df.iterrows():
        if row.apply(lambda x: bool(re.search(r"\$|\d", str(x)))).sum() >= 3:
            header_row_idx = i - 1 if i > 0 else i
            break

    if header_row_idx is None:
        print("Could not locate a suitable header row in equity statement.")
        return

    header = df.iloc[header_row_idx].fillna("").astype(str).tolist()
    header = ["description" if "description" in col.lower() or col.strip() == "" else col.strip() for col in header]
    df = df.iloc[header_row_idx + 1:].reset_index(drop=True)
    df.columns = header[:len(df.columns)]

    equity_data = []
    description = ""

    for _, row in df.iterrows():
        row = row.fillna("")
        has_number = False
        data = {}

        for col, val in row.items():
            val_str = str(val).strip()
            if re.match(r"^[-$()\d.,]+$", val_str) and any(char.isdigit() for char in val_str):
                try:
                    num = try_float(val_str)
                    data[col] = num
                    has_number = True
                except:
                    continue
            elif val_str:
                if description:
                    description += " " + val_str
                else:
                    description = val_str

        if has_number:
            data["description"] = description.strip()
            equity_data.append(data)
            description = ""  # Reset for next block

    equity_output = {
        "equity_statement": equity_data
    }

    json_path = os.path.join(OCR_OUTPUT_DIR, "equity_statement_clean.json")
    with open(json_path, "w") as f:
        json.dump(equity_output, f, indent=2)

    print(f"Equity statement JSON saved to {json_path}")

def try_float(val):
    try:
        return float(str(val).replace(",", "").replace("$", "").replace("—", "").strip())
    except:
        return None




def try_float(val):
    try:
        return float(str(val).replace(",", "").replace("$", "").replace("\u2014", "").strip())
    except:
        return None


def postprocess_tables_to_json():
    clean_json_path = os.path.join(OCR_OUTPUT_DIR, "clean_output.json")
    if os.path.exists(clean_json_path):
        with open(clean_json_path) as f:
            output = json.load(f)
    else:
        output = {}

    for fname in os.listdir(OCR_OUTPUT_DIR):
        if not fname.endswith(".csv"):
            continue

        label = fname.split("_")[0]
        csv_path = os.path.join(OCR_OUTPUT_DIR, fname)
        print(f"label: {label}")

        if label == "income":
            process_income_statement_to_json(csv_path)
            # Load and store the result
            income_json_path = os.path.join(OCR_OUTPUT_DIR, "income_statement_clean.json")
            if os.path.exists(income_json_path):
                with open(income_json_path) as f:
                    output["income_statement"] = json.load(f)
            continue

        if label == "comprehensive":
            process_comprehensive_income_to_json(csv_path)
            # Reload income after merge
            income_json_path = os.path.join(OCR_OUTPUT_DIR, "income_statement_clean.json")
            if os.path.exists(income_json_path):
                with open(income_json_path) as f:
                    output["income_statement"] = json.load(f)
            continue

        if label == "cash":
            process_cashflow_to_json(csv_path)
            cashflow_json_path = os.path.join(OCR_OUTPUT_DIR, "cashflow_statement_clean.json")
            if os.path.exists(cashflow_json_path):
                with open(cashflow_json_path) as f:
                    output["cashflow_statement"] = json.load(f)
            continue

        if label == "balance":
            process_balance_sheet_to_json(csv_path)
            balance_json_path = os.path.join(OCR_OUTPUT_DIR, "balance_sheet_clean.json")
            if os.path.exists(balance_json_path):
                with open(balance_json_path) as f:
                    output["balance_sheet"] = json.load(f)
            continue

        if label == "equity":
            process_equity_statement_to_json(csv_path)
            equity_json_path = os.path.join(OCR_OUTPUT_DIR, "equity_statement_clean.json")
            if os.path.exists(equity_json_path):
                with open(equity_json_path) as f:
                    output["equity_statement"] = json.load(f)
            continue


        df = pd.read_csv(csv_path, header=None)
        if df.empty:
            continue

        tables = []
        current_table = []

        def is_header_row(row):
            return sum(bool(re.search(r"20\\d{2}", str(cell))) for cell in row) >= 2

        for _, row in df.iterrows():
            if is_header_row(row):
                if current_table:
                    tables.append(current_table)
                current_table = [row]
            else:
                if current_table:
                    current_table.append(row)

        if current_table:
            tables.append(current_table)

        for idx, table_rows in enumerate(tables):
            table_df = pd.DataFrame(table_rows).reset_index(drop=True)
            if len(table_df) < 2:
                continue
            table_df.columns = table_df.iloc[0]
            table_df = table_df[1:]

            year_cols = [col for col in table_df.columns if re.search(r"20\\d{2}", str(col)) or "202" in str(col)]
            for _, row in table_df.iterrows():
                item = str(row.iloc[0])
                values = {str(year): row.get(year, None) for year in year_cols}
                output.setdefault(label, []).append({"item": item, **values})

    with open(clean_json_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Cleaned JSON saved to {clean_json_path}")



# --- Main Pipeline ---
async def obtain_financial_statements(pdf_path: str):
    try:
        # First locate the pages (this is now synchronous)
        pages = locate_financial_pages(pdf_path)
        if not pages:
            print("No financial statement pages found")
            return {}

        # Group pages by label
        label_groups = defaultdict(list)
        for page in pages:
            label_groups[page["label"]].append(page["page_number"])

        # Process each label group
        tasks = []
        for label, page_numbers in label_groups.items():
            tasks.append(process_label_group((label, page_numbers, pdf_path)))

        # Wait for all processing to complete
        await asyncio.gather(*tasks)

        # Postprocess to JSON
        return postprocess_tables_to_json()
    except Exception as e:
        print(f"Error in obtain_financial_statements: {str(e)}")
        return {}

if __name__ == "__main__":
    asyncio.run(obtain_financial_statements(PDF_PATH))
