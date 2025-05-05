def process_income_statement_to_json(csv_path):
    json_path = os.path.join(OCR_OUTPUT_DIR, "income_statement_clean.json")

    # Load existing JSON if available
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
        print("⚠️ Could not locate header row with years.")
        return

    header = df.iloc[header_row_idx].tolist()

    # Normalize header to just extract the year (e.g., "Dec 30, 2023" -> "2023") and capture full date
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

    # Update or set dates
    existing_dates = data.get("dates", {})
    existing_dates.update(year_to_full_date)
    data["dates"] = existing_dates

    df_data = df.iloc[header_row_idx + 1:].copy()
    df_data.columns = normalized_header
    df_data = df_data.dropna(how='all')

    # Identify year columns and item column
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
    print(f"✅ Income statement JSON saved to {json_path}")


# --- Custom Handler: Comprehensive Income JSON ---
def process_comprehensive_income_to_json(csv_path):
    try:
        with open(os.path.join(OCR_OUTPUT_DIR, "income_statement_clean.json")) as f:
            data = json.load(f)
    except FileNotFoundError:
        print("⚠️ Income statement JSON not found. Cannot merge.")
        return

    df = pd.read_csv(csv_path, header=None)
    df = df.dropna(how='all').reset_index(drop=True)

    header_row_idx = None
    for idx, row in df.iterrows():
        if sum(bool(re.search(r"20\d{2}", str(cell))) for cell in row) >= 2:
            header_row_idx = idx
            break

    if header_row_idx is None:
        print("⚠️ Could not locate header row in comprehensive income.")
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
    print(data)
    json_path = os.path.join(OCR_OUTPUT_DIR, "income_statement_clean.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Comprehensive income data merged and saved to {json_path}")

def postprocess_tables_to_json():
    output = defaultdict(list)
    for fname in os.listdir(OCR_OUTPUT_DIR):
        if not fname.endswith(".csv"):
            continue
        label = fname.split("_")[0]
        csv_path = os.path.join(OCR_OUTPUT_DIR, fname)
        print(f"label: {label}")
        if label == "income":
            process_income_statement_to_json(csv_path)
            continue
        if label == "comprehensive":
            process_comprehensive_income_to_json(csv_path)
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
                output[label].append({"item": item, **values})
    json_path = os.path.join(OCR_OUTPUT_DIR, "clean_output.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✅ Cleaned JSON saved to {json_path}")

postprocess_tables_to_json()