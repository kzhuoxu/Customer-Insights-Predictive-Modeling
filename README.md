# Take-Home Project: Income Classification & Customer Segmentation

This repository contains solutions for two machine learning tasks using US Census data:
1. **Binary Classification**: Predicting income levels (>$50K vs ≤$50K)
2. **Customer Segmentation**: Creating marketing personas using unsupervised clustering

## Project Structure

```
├── data/                          # Raw census data files
├── src/                           # Source code modules
│   ├── config.py                  # Configuration and file paths
│   ├── data_loader.py             # Data loading and initial cleaning
│   ├── feature_engineering.py     # Feature creation and transformations
│   ├── preprocessing.py           # ML preprocessing pipelines
│   ├── train.py                   # Classification model training
│   ├── segment.py                 # Segmentation model execution
│   └── segmentation_utils.py      # Clustering utilities
├── models/                        # Trained model artifacts
├── outputs/                       # Generated results and visualizations
├── plots/                         # Analysis plots and charts
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Setup Instructions

### 1. Environment Setup
```bash
# Clone repository (if applicable)
git clone https://github.com/kzhuoxu/Customer-Insights-Predictive-Modeling/tree/main

# Install required dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
Ensure the following files are present in the `data/` directory:
- `census-bureau.data` - Main dataset
- `census-bureau.columns` - Column names file

## Execution Instructions

### Classification Model Training

To train the income prediction model:

```bash
cd TakeHomeProject_new
python -m src.train
```

**What this does:**
- Loads and preprocesses census data
- Applies feature engineering transformations
- Trains a LightGBM classifier with optimized hyperparameters
- Evaluates model performance on held-out test set
- Saves trained model to `models/income_classifier_lgbm.joblib`

**Expected Output:**
- Model performance metrics (ROC AUC, F1-score, classification report)
- Confusion matrix visualization
- Trained model artifact

### Customer Segmentation Analysis

To generate customer segments:

```bash
cd TakeHomeProject_new
python -m src.segment
```

**What this does:**
- Preprocesses data for clustering analysis
- Applies dimensionality reduction using TruncatedSVD
- Determines optimal number of clusters using silhouette analysis
- Performs K-means clustering
- Generates cluster profiles and visualizations
- Saves segmentation results to `outputs/customer_segments.csv`

**Expected Output:**
- Optimal cluster count determination
- Customer segment assignments
- Cluster profile visualizations and heatmaps
- Saved clustering model and preprocessor

## Key Dependencies

- **Python 3.8+**
- **Core ML Libraries**: scikit-learn, lightgbm, pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Utilities**: joblib (model serialization)

See `requirements.txt` for complete dependency list with versions.

## Output Files

After successful execution, the following key files will be generated:

**Classification:**
- `models/income_classifier_lgbm.joblib` - Trained classification model
- `outputs/final_confusion_matrix.png` - Model performance visualization

**Segmentation:**
- `outputs/customer_segments.csv` - Customer records with cluster assignments
- `models/kmeans_model.joblib` - Trained clustering model
- `models/segmentation_pipeline.joblib` - Complete preprocessing pipeline
- `plots/` - Various cluster profile visualizations

## Notes

- All scripts use reproducible random seeds (RANDOM_STATE = 42)
- Models are optimized for business metrics (ROC AUC for classification, silhouette score for clustering)
- The codebase is modular and configuration-driven for easy maintenance and experimentation

For detailed analysis, methodology, and business recommendations, please refer to the accompanying project report.