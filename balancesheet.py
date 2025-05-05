# --- Custom Handler: Income Statement JSON ---
# --- Custom Handler: Income Statement JSON ---
def process_income_statement_to_json(csv_path):
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
        print("⚠️ Could not locate header row with years.")
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

    json_path = os.path.join(OCR_OUTPUT_DIR, "income_statement_clean.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Comprehensive income data merged and saved to {json_path}")




# --- Custom Handler: Cash Flow JSON ---
def process_cashflow_to_json(csv_path):
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
        print("⚠️ Could not locate header row in cashflow.")
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

    print(f"✅ Cash flow statement JSON saved to {json_path}")

def process_balance_sheet_to_json(csv_path):
    df = pd.read_csv(csv_path, header=None)
    if df.empty:
        print(f"⚠️ Balance sheet CSV at {csv_path} is empty.")
        return

    year_row_idx = None
    for i, row in df.iterrows():
        year_candidates = [str(cell) for cell in row if re.search(r"20\d{2}", str(cell))]
        if len(year_candidates) >= 2:
            year_row_idx = i
            break

    if year_row_idx is None:
        print(f"⚠️ Could not find year header in balance sheet.")
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

    print(f"✅ Balance sheet JSON saved to {json_path}")


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

    print(f"✅ Cleaned JSON saved to {clean_json_path}")



postprocess_tables_to_json()