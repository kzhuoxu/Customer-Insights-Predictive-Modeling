# src/feature_engineering.py

import pandas as pd
import numpy as np

from src import config

def map_coded_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps numeric codes in specific categorical columns to their string labels
    based on the census codebook.
    """
    print("Mapping numeric codes to string labels...")
    code_maps = {
        'veterans_benefits': {
            0: 'Not in universe', 1: 'Yes', 2: 'No'
        },
        'own_business_or_self_employed': {
            0: 'Not in business or self-employed',
            1: 'Own business - incorporated',
            2: 'Self-employed - not incorporated'
        }
    }
    for column, mapping in code_maps.items():
        if column in df.columns:
            df[column] = df[column].map(mapping)
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all feature engineering steps to the raw features DataFrame.

    This function takes the cleaned DataFrame from the data_loader and creates
    a suite of new features based on domain knowledge to improve model performance
    and interpretability.

    Args:
        df (pd.DataFrame): The input DataFrame containing the raw features.

    Returns:
        pd.DataFrame: The DataFrame with the new, engineered features added.
    """
    
    # Initial mapping of coded features to string labels for better interpretability.
    df = map_coded_features(df)
    
    # --- 1. Financial Profile Features ---
    # Log-transform income-related features to reduce skewness.
    for col in ['wage_per_hour', 'capital_gains', 'capital_losses', 'dividends_from_stocks']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: np.log1p(x) if x > 0 else 0)

    # Combine capital gains and losses into a single, more powerful feature.
    df['net_capital_gain'] = df['capital_gains'] - df['capital_losses']
    
    # Create a binary flag for investment activity, simplifying sparse data.
    df['has_investment_income'] = ((df['capital_gains'] > 0) | (df['dividends_from_stocks'] > 0)).astype(int)

    # --- 2. Employment and Work Status Features ---

    def _refined_employment_status(status_string: str) -> str:
        """Helper to group detailed employment status into clean categories."""
        if status_string == 'Full-time schedules':
            return 'full-time'
        elif status_string in ['PT for econ reasons usually FT', 
                               'PT for non-econ reasons usually FT', 
                               'PT for econ reasons usually PT']:
            return 'part-time'
        elif status_string in ['Unemployed full-time', 'Unemployed part- time']:
            return 'unemployed'
        else: # Catches 'Not in labor force', 'Disabled', Armed Forces categories etc.
            return 'not-in-labor-force'
    
    # Apply the function to create a more interpretable employment feature.
    df['employment_status'] = df['full_or_part_time_employment_stat'].apply(_refined_employment_status)

    # --- 3. Life Stage and Demographic Features ---

    # Create a squared term for age to help linear models capture non-linear relationships.
    df['age_squared'] = df['age'] ** 2

    # Group age into standard demographic bins for better interpretation and segmentation.
    def _age_group(age: int) -> str:
        if age < 18: return 'Under 18'
        if age <= 29: return '18-29'
        if age <= 44: return '30-44'
        if age <= 59: return '45-59'
        return '60+'
    df['age_group'] = df['age'].apply(_age_group)

    # Group detailed education levels into a clear, ordinal hierarchy.
    def _group_education(education_level: str) -> str:
        if education_level == 'Children': return 'Children'
        if education_level in ['Less than 1st grade', '1st 2nd 3rd or 4th grade', 
                               '5th or 6th grade', '7th and 8th grade', '9th grade', 
                               '10th grade', '11th grade', '12th grade no diploma']:
            return 'No-High-School'
        if education_level == 'High school graduate': return 'High-School-Graduate'
        if education_level in ['Some college but no degree', 
                               'Associates degree-occup /vocational', 
                               'Associates degree-academic program']:
            return 'Some-College'
        if education_level == 'Bachelors degree(BA AB BS)': return 'Bachelors-Degree'
        if education_level in ['Masters degree(MA MS MEng MEd MSW MBA)', 
                               'Prof school degree (MD DDS DVM LLB JD)',
                               'Doctorate degree(PhD EdD)']:
            return 'Advanced-Degree'
        return 'Other'
    df['education_group'] = df['education'].apply(_group_education)

    # --- 4. Interaction Features ---
    # Combine features to capture more complex relationships.

    # Interaction between marital status and age group.
    def _marital_age_interaction(row: pd.Series) -> str:
        is_married = row['marital_stat'] in ['Married-civilian spouse present', 'Married-A F spouse present']
        age_group = row['age_group']
        if is_married:
            return f'Married_{age_group}'
        else:
            return f'Not-Married_{age_group}'
    df['marital_age_interaction'] = df.apply(_marital_age_interaction, axis=1)

    # Interaction between education level and age group.
    def _education_age_interaction(row: pd.Series) -> str:
        has_degree = row['education_group'] in ['Bachelors-Degree', 'Advanced-Degree']
        age_group = row['age_group']
        if has_degree:
            return f'Degree_{age_group}'
        else:
            return f'No-Degree_{age_group}'
    df['education_age_interaction'] = df.apply(_education_age_interaction, axis=1)

    # --- 5. Immigration Status Features ---

    def _get_immigrant_status(row: pd.Series) -> str:
        """Determines if a person is a 1st gen, 2nd gen, or native-born resident."""
        is_self_foreign = row['country_of_birth_self'] != 'United-States'
        is_father_foreign = row['country_of_birth_father'] != 'United-States'
        is_mother_foreign = row['country_of_birth_mother'] != 'United-States'

        if is_self_foreign:
            return '1st_Gen_Immigrant'
        elif is_father_foreign or is_mother_foreign:
            return '2nd_Gen_Immigrant'
        else:
            return 'Native-Born'
    
    df['immigrant_status'] = df.apply(_get_immigrant_status, axis=1)

    # --- 6. Cleanup ---
    # Drop columns that won't be used for modeling to reduce noise and potential overfitting.
    cols_to_drop = [col for col in config.COLUMNS_TO_DROP if col in df.columns]
    if cols_to_drop:
        print(f"Dropping unused columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    print("Feature engineering complete. New columns added.")
    return df


if __name__ == '__main__':
    # This block allows you to test the feature engineering script directly.
    # It will only run when you execute `python src/feature_engineering.py` from the terminal.
    
    from data_loader import load_census_data
    from config import DATA_FILE, COLUMNS_FILE

    print("--- Running Test for feature_engineering.py ---")
    
    # Load the data
    initial_df = load_census_data(DATA_FILE, COLUMNS_FILE)
    
    # Keep only the features, as the function expects
    features_only_df = initial_df.drop(columns=['label'])
    
    # Run the feature engineering function
    featured_df = create_features(features_only_df)
    
    # Print the names of the new columns created
    new_cols = [col for col in featured_df.columns if col not in features_only_df.columns]
    print("\nNew columns created:")
    for col in new_cols:
        print(f"- {col}")
        
    print("\nSample of the 'education_group' column:")
    print(featured_df['education_group'].value_counts().head())

    print("\n--- Test Complete ---")