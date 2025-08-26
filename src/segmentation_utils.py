# src/segmentation_utils.py

import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from src import config

def choose_k(X_processed: np.ndarray, k_list: list[int]) -> int:
    """
    Chooses the best k for K-Means based on Silhouette and Davies-Bouldin scores.
    This function iterates through candidate k values, fitting a MiniBatchKMeans model
    on a sample of the data to find the optimal number of clusters.
    """
    # Use a sample for speed, as defined in the config
    if X_processed.shape[0] > config.SAMPLE_SIZE_FOR_K_SELECTION:
        sample_indices = np.random.choice(X_processed.shape[0], config.SAMPLE_SIZE_FOR_K_SELECTION, replace=False)
        X_sample = X_processed[sample_indices, :]
    else:
        X_sample = X_processed

    metrics = []
    logging.info(f"Starting k-selection for k in {k_list} using a sample of {X_sample.shape[0]}...")
    for k in k_list:
        km = MiniBatchKMeans(n_clusters=k, n_init='auto', batch_size=config.KMEANS_BATCH_SIZE, random_state=config.RANDOM_STATE)
        labels = km.fit_predict(X_sample)
        sil = silhouette_score(X_sample, labels)
        dbi = davies_bouldin_score(X_sample, labels)
        metrics.append({"k": k, "silhouette": sil, "davies_bouldin": dbi})
        logging.info(f"  k={k}: Silhouette={sil:.4f}, Davies-Bouldin={dbi:.4f}")

    # --- Plotting the metrics for manual inspection ---
    metrics_df = pd.DataFrame(metrics)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Cluster Evaluation Metrics for K-Selection', fontsize=16)

    ax1.plot(metrics_df['k'], metrics_df['davies_bouldin'], 'bo-')
    ax1.set_xlabel('Number of Clusters (k)'); ax1.set_ylabel('Davies-Bouldin Score'); ax1.set_title('Davies-Bouldin Score (Lower is better)')

    ax2.plot(metrics_df['k'], metrics_df['silhouette'], 'ro-')
    ax2.set_xlabel('Number of Clusters (k)'); ax2.set_ylabel('Average Silhouette Score'); ax2.set_title('Silhouette Score (Higher is better)')
    
    plot_path = os.path.join(config.PLOTS_DIR, "k_selection_metrics.png")
    plt.savefig(plot_path)
    logging.info(f"K-selection metrics plot saved to {plot_path}")
    plt.close()
    
    # Sort to find the best k based on highest silhouette, then lowest Davies-Bouldin
    metrics_df_sorted = metrics_df.sort_values(by=['silhouette', 'davies_bouldin'], ascending=[False, True])
    best_k = int(metrics_df_sorted.iloc[0]['k'])
    logging.info(f"Optimal k selected based on metrics: {best_k}")
    return best_k

def save_numerical_profile_heatmap(df_with_labels: pd.DataFrame, numerical_cols: list[str], output_path: str):
    """Generates and saves a heatmap of the mean numerical features by cluster."""
    logging.info("Generating numerical profile heatmap...")
    # Calculate the mean of numerical features for each cluster
    numerical_summary = df_with_labels.groupby('cluster')[numerical_cols].mean()

    # Normalize the data for fair comparison in the heatmap
    scaler = StandardScaler()
    summary_scaled = scaler.fit_transform(numerical_summary.T)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        summary_scaled,
        annot=numerical_summary.T,
        fmt=".2f",
        cmap="coolwarm",
        yticklabels=numerical_summary.columns,
        xticklabels=numerical_summary.index,
        linewidths=.5
    )
    plt.title('Heatmap of Mean Numerical Features by Cluster (Centered & Scaled)', fontsize=16)
    plt.ylabel('Numerical Feature')
    plt.xlabel('Cluster')
    plt.tight_layout()
    plt.savefig(output_path)
    logging.info(f"Numerical profile heatmap saved to {output_path}")
    plt.close()

def save_categorical_profile_heatmaps(df_with_labels: pd.DataFrame, categorical_cols: list[str], output_dir: str):
    """Generates and saves heatmaps of categorical feature distributions for each cluster."""
    logging.info("Generating categorical profile heatmaps...")
    for col in categorical_cols:
        # Use crosstab to get a frequency table normalized by cluster
        contingency_table = pd.crosstab(
            index=df_with_labels['cluster'],
            columns=df_with_labels[col],
            normalize='index'
        )
        
        plt.figure(figsize=(12, max(7, len(contingency_table.columns) // 2))) # Adjust height for many categories
        sns.heatmap(contingency_table, annot=True, fmt=".1%", cmap="YlGnBu")
        plt.title(f"Proportion of '{col}' within each Cluster")
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f"categorical_dist_{col}.png")
        plt.savefig(output_path)
        plt.close()
    logging.info(f"Categorical profile heatmaps saved to {output_dir}")