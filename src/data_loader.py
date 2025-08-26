# src/data_loader.py

import pandas as pd
import numpy as np
import re
from pathlib import Path

def load_census_data(data_path: Path, columns_path: Path) -> pd.DataFrame:
    """
    Loads the Census data, assigns clean column names, and performs
    essential initial cleaning.

    Args:
        data_path (Path): The file path to the main data file.
        columns_path (Path): The file path to the column names file.

    Returns:
        pd.DataFrame: A cleaned DataFrame ready for feature engineering.
    """
    # 1. Load and clean column names
    # The columns file is a single column of text.
    try:
        column_names = pd.read_csv(columns_path, header=None).iloc[:, 0].tolist()
        # Clean the names: lowercase, replace spaces/dashes with underscores
        clean_column_names = []
        for col in column_names:
            temp_col = col.strip().lower().replace(' ', '_').replace('-', '_')
            clean_col = re.sub(r"[^a-z0-9_]", "", temp_col)
            clean_column_names.append(clean_col)
            # Debugging output
            # print(f"Original: {col} | Cleaned: {clean_col}")
    except FileNotFoundError:
        print(f"Error: Columns file not found at {columns_path}")
        raise

    # 2. Load the main dataset without a header
    try:
        df = pd.read_csv(
            data_path, 
            header=None, 
            names=clean_column_names,
            sep=',',
            skipinitialspace=True # Handles whitespace after delimiters
        )
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        raise
        
    # 3. Perform essential, universal cleaning
    
    # Clean the target variable (income_label) into a binary 0/1 format
    # This column name comes from the data dictionary for the project.
    target_col = 'label'
    if target_col in df.columns:
        df[target_col] = df[target_col].apply(lambda x: 1 if '+' in str(x) else 0)
    else:
        # Handling the case where the column name might be slightly different
        # In a real project, this would be logged and investigated.
        print("Warning: Target column 'label' not found. Please verify column names.")

    # Standardize missing value placeholders to np.nan
    # The dataset uses '?' for missing values.
    df.replace('?', np.nan, inplace=True)

    # Strip whitespace from all object columns to prevent issues like ' Male' vs 'Male'
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    print("Data loaded and initial cleaning complete.")
    print(f"Dataset shape: {df.shape}")
    
    return df