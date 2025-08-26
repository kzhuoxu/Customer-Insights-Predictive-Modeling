# %% [markdown]
# # Segmentation
# Unsupervised Learning

# %%
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ## Load Data

# %%
notebook_path = Path.cwd()
project_root = notebook_path.parent
sys.path.insert(0, str(project_root))

# print(f"Project Root (added to sys.path): {project_root}")

# %%
from src import config
from src.data_loader import load_census_data

raw_df = load_census_data(
    data_path=config.DATA_FILE,
    columns_path=config.COLUMNS_FILE
)
raw_df.head()

# %%
print(config.TARGET_VARIABLE)

# %% [markdown]
# Separate the Target, Features and Weights

# %%
X = raw_df.drop(columns=[config.TARGET_VARIABLE, 'weight'])
y = raw_df[config.TARGET_VARIABLE]
weights = raw_df['weight']

# %% [markdown]
# Feature Engineering

# %%
from src.feature_engineering import create_features

X_featured = create_features(X)

print(X_featured.shape[1])
print(X_featured.columns.tolist())

# %% [markdown]
# ## Feature Selection

# %%
SEGMENTATION_FEATURES = [
    # Core Demographics
    'age', 'sex', 'race',
    # Household Composition
    'marital_stat', 'family_members_under_18', 'tax_filer_stat',
    # Education & Careers
    'education_group', 'major_occupation_code', 'major_industry_code',
    # Work Intensity
    'employment_status', 'weeks_worked_in_year',
    # Financial Profile
    'wage_per_hour', 'capital_gains', 'capital_losses', 'dividends_from_stocks',
    'has_investment_income', 'own_business_or_self_employed',
    # Nativity & Migration
    'citizenship', 'live_in_this_house_1_year_ago',
    # Veteran Status
    'veterans_benefits'
]

X_segment = X_featured[SEGMENTATION_FEATURES].copy()
SEG_NUMERICAL_COLS = X_segment.select_dtypes(include=np.number).columns.tolist()
SEG_CATEGORICAL_COLS = X_segment.select_dtypes(include=['object', 'category']).columns.tolist()

# %%
print(f"Numerical Features ({len(SEG_NUMERICAL_COLS)}): {SEG_NUMERICAL_COLS}")
print(f"Categorical Features ({len(SEG_CATEGORICAL_COLS)}): {SEG_CATEGORICAL_COLS}")

# %% [markdown]
# ## Preprocessing 
# ### with FAMD

# %%
import prince

famd_preprocessor = ColumnTransformer(
    [('scaler', StandardScaler(), SEG_NUMERICAL_COLS)],
    remainder='passthrough'
)
famd_preprocessor.set_output(transform='pandas')

N_COMPONENTS = 15

famd_pipeline = Pipeline(steps=[
    ('preprocessor', famd_preprocessor),
    ('famd', prince.FAMD(
        n_components=N_COMPONENTS,
        n_iter=5,
        random_state=config.RANDOM_STATE
    ))
])

print("Fitting FAMD and transforming data...")
X_processed_famd = famd_pipeline.fit_transform(X_segment)
print(f"Processed Feature Shape after FAMD: {X_processed_famd.shape}")


# %%
famd_instance = famd_pipeline.named_steps['famd']
famd_instance.eigenvalues_summary

# %%
eigenvalues = famd_instance.eigenvalues_
total_inertia = sum(eigenvalues)
explained_inertia = [eig / total_inertia for eig in eigenvalues]

feature_contributions = famd_instance.column_coordinates_

# --- Detailed Analysis for Top Components ---

print("="*60)
print("--- Decoding FAMD Component 0 (The Primary Axis) ---")
print(f"This component explains {explained_inertia[0]:.2%} of the total variance.")
print("Top 10 features most influential for Component 0:")
# We display the actual coordinate values to see the direction of influence (positive/negative)
# but sort by the absolute value to find the strongest influencers.
display(feature_contributions[0].reindex(feature_contributions[0].abs().sort_values(ascending=False).index).head(10).to_frame(name='Coordinate on Component 0'))


print("\n" + "="*60)
print("--- Decoding FAMD Component 1 (The Secondary Axis) ---")
print(f"This component explains {explained_inertia[1]:.2%} of the total variance.")
print("Top 10 features most influential for Component 1:")
display(feature_contributions[1].reindex(feature_contributions[1].abs().sort_values(ascending=False).index).head(10).to_frame(name='Coordinate on Component 1'))


max_val = feature_contributions.abs().max().max()

# Apply the background gradient styling
styled_contributions = feature_contributions.style.background_gradient(
    cmap='bwr', vmin=-max_val, vmax=max_val
).format("{:.3f}")

print("\nFull Contribution Matrix of Features on FAMD Components:")
display(styled_contributions)

# %% [markdown]
# Find Optimal Number of Clusters

# %%
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

sample_indices = np.random.choice(X_processed_famd.shape[0], config.SAMPLE_SIZE_FOR_K_SELECTION, replace=False)
X_sample_famd = X_processed_famd.iloc[sample_indices]

metrics = []
k_list = config.K_CANDIDATES
print(f"Starting k-selection for k in {k_list} on FAMD components...")
for k in k_list:
    km = MiniBatchKMeans(n_clusters=k,
                         n_init='auto',
                         batch_size=config.KMEANS_BATCH_SIZE,
                         random_state=config.RANDOM_STATE)
    labels = km.fit_predict(X_sample_famd)
    sil = silhouette_score(X_sample_famd, labels)
    dbi = davies_bouldin_score(X_sample_famd, labels)
    metrics.append({"k": k, "silhouette": sil, "davies_bouldin": dbi})
    print(f"  k={k}: Silhouette={sil:.4f}, Davies-Bouldin={dbi:.4f}")

metrics_df = pd.DataFrame(metrics)

# %%
# Plotting the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Cluster Evaluation Metrics on FAMD Components', fontsize=16)

# Davies-Bouldin Score Plot (Elbow Method) - Lower is better
ax1.plot(metrics_df['k'], metrics_df['davies_bouldin'], 'bo-')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Davies-Bouldin Score')
ax1.set_title('Davies-Bouldin Score for Optimal k')

# Silhouette Score Plot - Higher is better
ax2.plot(metrics_df['k'], metrics_df['silhouette'], 'ro-')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Average Silhouette Score')
ax2.set_title('Silhouette Analysis for Optimal k')

plt.show()


# %% [markdown]
# ### Compare with Truncated SVD for Dimension Reduction

# %%
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder

# %% [markdown]
# ### 1. Create the TruncatedSVD Preprocessing Pipeline
# This pipeline will scale numerical features and one-hot encode categorical features.

# %%
# Define the preprocessor for SVD
svd_preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), SEG_NUMERICAL_COLS),
        ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), SEG_CATEGORICAL_COLS)
    ],
    remainder='passthrough'
)

# Define the full pipeline with TruncatedSVD
N_COMPONENTS = 15 # Let's use the same number of components for a fair comparison

svd_pipeline = Pipeline(steps=[
    ('preprocessor', svd_preprocessor),
    ('svd', TruncatedSVD(
        n_components=N_COMPONENTS,
        random_state=config.RANDOM_STATE
    ))
])

# Fit the pipeline
print("Fitting TruncatedSVD pipeline...")
X_processed_svd = svd_pipeline.fit_transform(X_segment)
print(f"Shape of data after SVD: {X_processed_svd.shape}")

# %%
# Get the fitted preprocessor and SVD model from the pipeline
preprocessor_fitted = svd_pipeline.named_steps['preprocessor']
svd_model = svd_pipeline.named_steps['svd']

# Get the feature names after preprocessing (this is crucial)
# The get_feature_names_out() method provides the names of the scaled and OHE columns
feature_names_out = preprocessor_fitted.get_feature_names_out()

# Create a DataFrame of the component loadings (weights)
# The .components_ attribute has a shape of (n_components, n_features)
component_loadings = pd.DataFrame(
    svd_model.components_.T, # Transpose to have features as rows
    index=feature_names_out,
    columns=[f"Component {i}" for i in range(N_COMPONENTS)]
)

print("\n--- Decoding TruncatedSVD Component 0 ---")
print("Top 10 influential features (absolute value):")
display(component_loadings['Component 0'].reindex(component_loadings['Component 0'].abs().sort_values(ascending=False).index).head(10).to_frame())


print("\n--- Decoding TruncatedSVD Component 1 ---")
print("Top 10 influential features (absolute value):")
display(component_loadings['Component 1'].reindex(component_loadings['Component 1'].abs().sort_values(ascending=False).index).head(10).to_frame())

# %%
# --- Full Styled Loading Matrix ---
max_val = component_loadings.abs().max().max()

styled_svd_loadings = component_loadings.style.background_gradient(
    cmap='bwr', vmin=-max_val, vmax=max_val
).format("{:.3f}")

print("\nFull Loading Matrix for TruncatedSVD Components:")
display(styled_svd_loadings)


# %% [markdown]
# Find Optimal Number of Clusters

# %%
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

sample_indices = np.random.choice(X_processed_svd.shape[0], config.SAMPLE_SIZE_FOR_K_SELECTION, replace=False)
X_sample_svd = X_processed_svd[sample_indices]

metrics = []
k_list = config.K_CANDIDATES
print(f"Starting k-selection for k in {k_list} on SVD components...")
for k in k_list:
    km = MiniBatchKMeans(n_clusters=k,
                         n_init='auto',
                         batch_size=config.KMEANS_BATCH_SIZE,
                         random_state=config.RANDOM_STATE)
    labels = km.fit_predict(X_sample_svd)
    sil = silhouette_score(X_sample_svd, labels)
    dbi = davies_bouldin_score(X_sample_svd, labels)
    metrics.append({"k": k, "silhouette": sil, "davies_bouldin": dbi})
    print(f"  k={k}: Silhouette={sil:.4f}, Davies-Bouldin={dbi:.4f}")

metrics_df = pd.DataFrame(metrics)

# %%
# Plotting the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Cluster Evaluation Metrics on SVD Components', fontsize=16)

# Davies-Bouldin Score Plot (Elbow Method) - Lower is better
ax1.plot(metrics_df['k'], metrics_df['davies_bouldin'], 'bo-')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Davies-Bouldin Score')
ax1.set_title('Davies-Bouldin Score for Optimal k')

# Silhouette Score Plot - Higher is better
ax2.plot(metrics_df['k'], metrics_df['silhouette'], 'ro-')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Average Silhouette Score')
ax2.set_title('Silhouette Analysis for Optimal k')

plt.show()


# %%
OPTIMAL_K = 6 # Based on the analysis above

kmeans_final = KMeans(n_clusters=OPTIMAL_K, random_state=config.RANDOM_STATE, n_init='auto')
cluster_labels = kmeans_final.fit_predict(X_processed_svd)

# Add the cluster labels back to our original (pre-processed) segmentation data
X_segment['cluster'] = cluster_labels
print(f"Final cluster assignments (for k={OPTIMAL_K}):")
print(X_segment['cluster'].value_counts())

# %% [markdown]
# Interprete the Result

# %%
# Calculate the mean of numerical features for each cluster
numerical_summary = X_segment.groupby('cluster')[SEG_NUMERICAL_COLS].mean()
# Calculate the mean for the total population for comparison
total_mean = X_segment[SEG_NUMERICAL_COLS].mean()
numerical_summary.loc['Total_Mean'] = total_mean

# Display the summary table, transposing for easier reading
print("Mean of Numerical Features by Cluster:")
display(numerical_summary.T)

# %%
numerical_summary.T

# %%
# Normalize the data for fair comparison in the heatmap
scaler = StandardScaler()

# Ensure all column names and index are strings
numerical_summary.columns = numerical_summary.columns.astype(str)
numerical_summary.index = numerical_summary.index.astype(str)

summary_scaled = scaler.fit_transform(numerical_summary.T)  # Transpose to scale features
plt.figure(figsize=(12, 8))
sns.heatmap(
    summary_scaled,
    annot=numerical_summary.T,
    fmt=".1f",
    cmap="coolwarm",
    yticklabels=numerical_summary.columns,
    xticklabels=numerical_summary.index
)
plt.title('Heatmap of Mean Numerical Features by Cluster (Centered & Scaled)', fontsize=16)
plt.ylabel('Numerical Feature')
plt.xlabel('Cluster')
plt.show()


# %%
# Analyze the distribution of categorical features across clusters
for col in SEG_CATEGORICAL_COLS:
    # Use crosstab to get a frequency table
    contingency_table = pd.crosstab(
        index=X_segment['cluster'],
        columns=X_segment[col],
        normalize='index' # Normalize by row to get percentages
    )
    print(f"\n----- Distribution of '{col}' by Cluster -----")
    # Plotting the distributions
    plt.figure(figsize=(12, 7))
    # Use a diverging color palette to highlight differences from a central point
    sns.heatmap(contingency_table, annot=True, fmt=".2%", cmap="YlGnBu")
    plt.title(f"Proportion of '{col}' within each Cluster")
    plt.show()
    # plt.savefig(os.path.join(config.FIGURE_DIR, f"categorical_distribution_{col}.png"))


