import pandas as pd

def preprocess_dataframe(df):
    """Preprocess the DataFrame: trim column headers, remove empty rows and columns, and promote header if needed."""
    # Trim column headers and check for placeholders like "Unnamed"
    df = df.rename(columns=lambda x: x.strip())
    placeholder_detected = any("Unnamed" in col or "column" in col.lower() for col in df.columns)
    
    if placeholder_detected:
        # Promote the first row to header if placeholders detected
        new_header = df.iloc[0]  # Grab the first row for the header
        df = df[1:]  # Take the data less the header row
        df.columns = new_header  # Set the header row as the df header
        df.reset_index(drop=True, inplace=True)
    
    # Remove all empty rows and columns
    df.dropna(how='all', inplace=True)  # Rows
    df.dropna(axis=1, how='all', inplace=True)  # Columns

    # Additional check and trim after promoting the first row to header
    df = df.rename(columns=lambda x: x.strip())
    df.columns = [col if not ("Unnamed" in str(col) or "column" in str(col).lower()) else "" for col in df.columns]

    # Reset index after dropping rows
    df.reset_index(drop=True, inplace=True)
    return df

def load_sheet_data(selected_sheets_info):
    """Load data for selected sheets with preprocessing."""
    loaded_data = {}
    for key, (file, sheet_name) in selected_sheets_info.items():
        # Load data from the sheet
        if file.type == 'text/csv':
            data = pd.read_csv(file)
        else:  # For Excel files
            data = pd.read_excel(file, sheet_name=sheet_name)
        # Preprocess the loaded data
        data = preprocess_dataframe(data)
        loaded_data[key] = data
    return loaded_data
