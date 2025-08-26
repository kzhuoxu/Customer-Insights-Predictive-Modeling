# src/train.py

"""
This script orchestrates the entire training and evaluation pipeline for the
income classification model. It performs the following steps:
1.  Loads the cleaned data using the data_loader module.
2.  Applies feature engineering using the feature_engineering module.
3.  Splits the data into training and testing sets.
4.  Defines and trains the final, optimized LightGBM model pipeline.
5.  Evaluates the model's performance on the held-out test set.
6.  Saves the final trained model pipeline to a file for future use.
"""

import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, classification_report, ConfusionMatrixDisplay

# Import custom modules from the project
from src import config
from src.data_loader import load_census_data
from src.feature_engineering import create_features
from src.preprocessing import create_preprocessor

def run_training():
    """
    Executes the model training and evaluation pipeline.
    """
    # --- 1. Load Data ---
    print("Step 1/6: Loading data...")
    raw_df = load_census_data(
        data_path=config.DATA_FILE,
        columns_path=config.COLUMNS_FILE
    )

    # --- 2. Feature Engineering ---
    print("Step 2/6: Applying feature engineering...")
    X = raw_df.drop(columns=[config.TARGET_VARIABLE, 'weight'])
    y = raw_df[config.TARGET_VARIABLE]
    X_featured = create_features(X)

    # --- 3. Data Split ---
    print("Step 3/6: Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_featured, 
        y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE,
        stratify=y  # Crucial for maintaining class distribution
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # --- 4. Define Final Model Pipeline ---
    print("Step 4/6: Defining the final model pipeline...")
    
    print(f"Loading best hyperparameters from {config.HYPERPARAMETERS_FILE}")
    with open(config.HYPERPARAMETERS_FILE, 'r') as f:
        best_params_raw = json.load(f)

    # The keys from RandomizedSearchCV have a 'classifier__' prefix. We need to remove it.
    best_params = {key.replace('classifier__', ''): value for key, value in best_params_raw.items()}
    print("Best parameters loaded:", best_params)

    final_model = LGBMClassifier(
        random_state=config.RANDOM_STATE,
        is_unbalance=True,  # Use built-in imbalance handling
        n_jobs=-1,
        **best_params
    )
    
    NUMERICAL_COLS = X_featured.select_dtypes(include='number').columns.tolist()
    CATEGORICAL_COLS = X_featured.select_dtypes(include='object').columns.tolist()
    preprocessor = create_preprocessor(NUMERICAL_COLS, CATEGORICAL_COLS)

    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', final_model)
    ])

    # --- 5. Train the Final Model ---
    print("Step 5/6: Training the final model on the full training set...")
    final_pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # --- 6. Evaluate on Test Set ---
    print("\n--- Final Model Evaluation on Held-Out Test Set ---")
    y_pred_test = final_pipeline.predict(X_test)
    y_pred_proba_test = final_pipeline.predict_proba(X_test)[:, 1]

    final_roc_auc = roc_auc_score(y_test, y_pred_proba_test)
    print(f"Final ROC AUC on Test Set: {final_roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=['Income < $50k', 'Income > $50k']))

    # --- 7. Save the Final Model ---
    print(f"\nStep 7/7: Saving final model pipeline to {config.CLASSIFICATION_MODEL_FILE}...")
    joblib.dump(final_pipeline, config.CLASSIFICATION_MODEL_FILE)
    print("Model saved successfully.")

    # --- Optional: Display and Save Confusion Matrix ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_estimator(
        final_pipeline,
        X_test,
        y_test,
        display_labels=['Income < $50k', 'Income > $50k'],
        cmap='Blues',
        ax=ax
    )
    plt.title('Final Confusion Matrix on Test Data')
    # Save the figure to the outputs directory
    figure_path = config.OUTPUT_DIR / "final_confusion_matrix.png"
    plt.savefig(figure_path)
    print(f"Confusion matrix saved to {figure_path}")
    # plt.show() # Comment out or remove for a pure script

if __name__ == "__main__":
    # This block allows the script to be run from the command line.
    run_training()