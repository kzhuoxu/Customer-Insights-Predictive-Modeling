# src/segment.py

"""
This script executes the end-to-end customer segmentation pipeline.
It identifies distinct customer personas from census data using K-Means clustering
on a dimensionally-reduced feature space created by TruncatedSVD.
"""
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Import custom modules
from src import config
from src.data_loader import load_census_data
from src.feature_engineering import create_features
from src.segmentation_utils import choose_k, save_numerical_profile_heatmap, save_categorical_profile_heatmaps

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def run_segmentation():
    """Main function to run the entire segmentation pipeline."""
    # 1. Load and Engineer Data
    logging.info("Step 1/8: Loading and engineering data...")
    raw_df = load_census_data(config.DATA_FILE, config.COLUMNS_FILE)
    features_df = create_features(raw_df.drop(columns=[config.TARGET_VARIABLE, 'weight']))
    
    # 2. Select curated features for segmentation
    logging.info("Step 2/8: Selecting curated features for segmentation...")
    X_segment = features_df[config.SEGMENTATION_FEATURES].copy()
    numeric_cols = X_segment.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_segment.select_dtypes(include=['object', 'category']).columns.tolist()

    # 3. Build SVD Preprocessor and Pipeline
    logging.info("Step 3/8: Building the TruncatedSVD preprocessing and reduction pipeline...")
    svd_preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), numeric_cols),
            ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=True), categorical_cols)
        ],
        remainder='passthrough'
    )
    svd_pipeline = Pipeline(steps=[
        ('preprocessor', svd_preprocessor),
        ('svd', TruncatedSVD(n_components=config.N_COMPONENTS, random_state=config.RANDOM_STATE))
    ])

    # 4. Preprocess Data and Reduce Dimensions
    logging.info("Step 4/8: Preprocessing data and reducing dimensionality with SVD...")
    X_processed_svd = svd_pipeline.fit_transform(X_segment)
    logging.info(f"Shape of data after SVD: {X_processed_svd.shape}")
    
    # 5. Determine Optimal K
    logging.info("Step 5/8: Determining the optimal number of clusters (k)...")
    best_k = choose_k(X_processed_svd, config.K_CANDIDATES)
    best_k = 7  # Forcing k=5 based on prior knowledge; comment this line to use the chosen k

    # 6. Train Final K-Means Model
    logging.info(f"Step 6/8: Training final K-Means model with k={best_k}...")
    final_kmeans_model = KMeans(n_clusters=best_k, n_init='auto', random_state=config.RANDOM_STATE)
    X_segment['cluster'] = final_kmeans_model.fit_predict(X_processed_svd)

    # 7. Profile and Visualize Clusters
    logging.info("Step 7/8: Generating and saving cluster profiles...")
    # Visualize using the first 2 SVD components
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_processed_svd[:, 0], y=X_processed_svd[:, 1], hue=X_segment['cluster'], palette='viridis', alpha=0.5, s=20)
    plt.title(f'Customer Segments (k={best_k}) Visualized in 2D SVD Space')
    plt.xlabel('SVD Component 1'); plt.ylabel('SVD Component 2')
    plt.legend(title='Segment')
    plt.savefig(config.SVD_PLOT_FILE)
    logging.info(f"SVD visualization saved to {config.SVD_PLOT_FILE}")
    plt.close()

    # Generate and save detailed profile heatmaps
    save_numerical_profile_heatmap(X_segment, numeric_cols, config.NUMERICAL_PROFILE_PLOT_FILE)
    save_categorical_profile_heatmaps(X_segment, categorical_cols, config.PLOTS_DIR)

    # 8. Save Artifacts
    logging.info("Step 8/8: Saving all segmentation artifacts...")
    X_segment[['cluster']].to_csv(config.SEGMENTATION_OUTPUT_FILE)
    joblib.dump(final_kmeans_model, config.KMEANS_MODEL_FILE)
    joblib.dump(svd_pipeline, config.SEGMENTATION_PIPELINE_FILE) # Save the full SVD pipeline
    
    logging.info("\n--- Segmentation Pipeline Complete ---")
    
if __name__ == "__main__":
    run_segmentation()