# src/preprocessing.py

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src import config # Import the config to get feature lists

def create_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """
    Creates the full preprocessing pipeline for the census data.

    This function defines the steps for handling numerical and categorical features,
    including imputation, scaling, and one-hot encoding. The specific columns
    for each transformer are pulled from the config.py file.

    Returns:
        ColumnTransformer: A scikit-learn ColumnTransformer object ready to be
                           integrated into a model training pipeline.
    """

    # --- Define the pipeline for NUMERICAL features ---
    # 1. Impute missing values with the median (robust to outliers).
    # 2. Scale features to have zero mean and unit variance.
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # --- Define the pipeline for CATEGORICAL features ---
    # 1. Impute missing values with the most frequent value.
    # 2. One-hot encode the categories.
    #    - min_frequency: Groups rare categories into an "infrequent" bucket.
    #    - handle_unknown='ignore': Prevents errors if a new category appears in test data.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(
            min_frequency=0.005, # Consolidate categories appearing in < 0.5% of records
            handle_unknown='ignore',
            sparse_output=False
        ))
    ])

    # --- Create the master ColumnTransformer ---
    # This preprocessor applies the correct transformation to each column type.
    # The 'remainder="passthrough"' argument ensures that any columns not
    # explicitly listed are kept in the DataFrame, which can be useful for debugging.
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    print("Preprocessing pipeline created successfully.")
    return preprocessor