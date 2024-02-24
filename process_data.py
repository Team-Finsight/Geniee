import pandas as pd

def preprocess_dataframe(df):
    """Preprocess the DataFrame: trim column headers and remove empty rows."""
    # Trim column headers
    df = df.rename(columns=lambda x: x.strip())
    # Remove all empty rows at the beginning of the DataFrame
    df.dropna(how='all', inplace=True)
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
