import camelot
import os
import json
import sys

def extract_tables_from_pdf(pdf_path, output_dir="extracted_tables_json", flavor="stream", pages="all"):
    """
    Extract tables from a PDF using Camelot and save each as a JSON file.
    Args:
        pdf_path (str): Path to the PDF file.
        output_dir (str): Directory to save JSON files.
        flavor (str): 'stream' for whitespace tables, 'lattice' for bordered tables.
        pages (str): Pages to extract from (e.g., 'all', '1', '1-3').
    """
    os.makedirs(output_dir, exist_ok=True)
    tables = camelot.read_pdf(pdf_path, pages=pages, flavor=flavor)
    print(f"Detected {len(tables)} tables in {pdf_path}")
    for i, table in enumerate(tables):
        df = table.df
        # Optional: Clean up header if needed
        # df.columns = df.iloc[0]
        # df = df[1:].reset_index(drop=True)
        json_data = df.to_dict(orient="records")
        json_path = os.path.join(output_dir, f"table_{i+1}.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"Saved table {i+1} to {json_path}")
    print("Extraction complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python table_extractor.py <pdf_path> [output_dir] [flavor] [pages]")
        sys.exit(1)
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "extracted_tables_json"
    flavor = sys.argv[3] if len(sys.argv) > 3 else "stream"
    pages = sys.argv[4] if len(sys.argv) > 4 else "all"
    extract_tables_from_pdf(pdf_path, output_dir, flavor, pages) 