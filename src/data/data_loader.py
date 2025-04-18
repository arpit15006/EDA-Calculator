import pandas as pd
import numpy as np
import os

def load_application_data(file_path='application_data.csv'):
    """
    Load the application data from CSV file

    Parameters:
    -----------
    file_path : str
        Path to the application data CSV file

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the application data
    """
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='latin1', engine='python')

def load_previous_application_data(file_path='previous_application.csv'):
    """
    Load the previous application data from CSV file

    Parameters:
    -----------
    file_path : str
        Path to the previous application data CSV file

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the previous application data
    """
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='latin1', engine='python')

def load_column_descriptions(file_path='columns_description.csv'):
    """
    Load the column descriptions from CSV file

    Parameters:
    -----------
    file_path : str
        Path to the column descriptions CSV file

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the column descriptions
    """
    # Try different encodings
    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']

    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding, engine='python')
        except UnicodeDecodeError:
            continue

    # If all encodings fail, try with error handling
    return pd.read_csv(file_path, encoding='latin1', engine='python', on_bad_lines='skip')

def get_basic_stats(df):
    """
    Get basic statistics of the dataframe

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze

    Returns:
    --------
    dict
        Dictionary containing basic statistics
    """
    stats = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes,
        'missing_values': df.isnull().sum(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2),
        'numeric_stats': df.describe(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'numeric_columns': df.select_dtypes(include=['number']).columns.tolist()
    }
    return stats

def merge_application_with_previous(app_df, prev_df):
    """
    Merge application data with previous application data

    Parameters:
    -----------
    app_df : pd.DataFrame
        Application data DataFrame
    prev_df : pd.DataFrame
        Previous application data DataFrame

    Returns:
    --------
    pd.DataFrame
        Merged DataFrame
    """
    # Aggregate previous applications data at client level
    prev_agg = prev_df.groupby('SK_ID_CURR').agg({
        'SK_ID_PREV': 'count',
        'AMT_CREDIT': ['mean', 'sum', 'max', 'min'],
        'AMT_ANNUITY': ['mean', 'sum', 'max', 'min'],
        'AMT_APPLICATION': ['mean', 'sum', 'max', 'min'],
        'AMT_DOWN_PAYMENT': ['mean', 'sum', 'max', 'min'],
        'DAYS_DECISION': ['mean', 'max', 'min'],
        'CNT_PAYMENT': ['mean', 'max', 'min'],
    })

    # Flatten the column names
    prev_agg.columns = ['_'.join(col).strip() for col in prev_agg.columns.values]
    prev_agg.reset_index(inplace=True)

    # Calculate approval rate
    approved = prev_df[prev_df['NAME_CONTRACT_STATUS'] == 'Approved'].groupby('SK_ID_CURR').size()
    total = prev_df.groupby('SK_ID_CURR').size()
    approval_rate = (approved / total).fillna(0)
    approval_rate = pd.DataFrame({'SK_ID_CURR': approval_rate.index, 'PREV_APPROVAL_RATE': approval_rate.values})

    # Merge with approval rate
    prev_agg = prev_agg.merge(approval_rate, on='SK_ID_CURR', how='left')

    # Merge with application data
    return app_df.merge(prev_agg, on='SK_ID_CURR', how='left')

if __name__ == "__main__":
    # Test the functions
    app_df = load_application_data()
    prev_df = load_previous_application_data()
    col_desc = load_column_descriptions()

    print("Application Data Shape:", app_df.shape)
    print("Previous Application Data Shape:", prev_df.shape)
    print("Column Descriptions Shape:", col_desc.shape)

    # Get basic stats for application data
    app_stats = get_basic_stats(app_df)
    print("\nApplication Data Missing Values (Top 10):")
    print(app_stats['missing_percentage'].sort_values(ascending=False).head(10))

    # Target distribution
    print("\nTarget Distribution:")
    print(app_df['TARGET'].value_counts(normalize=True) * 100)
