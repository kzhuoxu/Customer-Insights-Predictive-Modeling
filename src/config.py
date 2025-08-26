# src/config.py

from pathlib import Path

# --- 1. DEFINE THE PROJECT'S ROOT DIRECTORY AS THE ANCHOR ---
# This is the single most important line for robust path management.
# __file__ is the path to this config.py file.
# .resolve() makes it an absolute path.
# .parent gives us the 'src' directory.
# .parent again gives us the project's root directory ('jpmc_take_home/').
BASE_DIR = Path(__file__).resolve().parent.parent

# --- 2. BUILD ALL OTHER PATHS FROM THE BASE DIRECTORY ---
# This ensures all paths are absolute and work regardless of where
# you run your scripts from (root, src/, notebooks/, etc.).

# --- Input Data Paths ---
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "census-bureau.data"
COLUMNS_FILE = DATA_DIR / "census-bureau.columns"

COLUMNS_TO_DROP = [
    'detailed_industry_recode',
    'detailed_occupation_recode',
    'year'
]

# --- Output & Artifact Paths ---
# Create the output directories if they don't already exist.
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Hyperparameter Tuning Artifacts
HYPERPARAMETERS_FILE = MODEL_DIR / "best_hyperparameters.json"
HYPERPARAMETERS_FILE_alt = MODEL_DIR / "best_hyperparameters_f1_macro.json"

# Classification Model Artifact
CLASSIFICATION_MODEL_FILE = MODEL_DIR / "income_classifier_lgbm.joblib"

# Segmentation Artifacts
SEGMENTATION_OUTPUT_FILE = OUTPUT_DIR / "customer_segments.csv"
CLUSTER_PROFILES_FILE = OUTPUT_DIR / "cluster_profiles.csv"
KMEANS_MODEL_FILE = MODEL_DIR / "kmeans_model.joblib"
SEGMENTATION_PREPROCESSOR_FILE = MODEL_DIR / "segmentation_preprocessor.joblib"
SVD_PLOT_FILE = OUTPUT_DIR / "segmentation_svd_visualization.png"
NUMERICAL_PROFILE_PLOT_FILE = PLOTS_DIR / "numerical_profile_heatmap.png"
SEGMENTATION_PIPELINE_FILE = MODEL_DIR / "segmentation_pipeline.joblib"

# --- 3. MODELING & FEATURE PARAMETERS ---

# --- General ---
RANDOM_STATE = 42  # For reproducibility

# --- Classification ---
TARGET_VARIABLE = 'label'  # The binary income label
TEST_SIZE = 0.2    # 20% of data reserved for testing

# --- Segmentation ---
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

# Parameters for the k-selection process
K_CANDIDATES = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
SAMPLE_SIZE_FOR_K_SELECTION = 30000
N_COMPONENTS = 15  # For TruncatedSVD
# Parameters for the final MiniBatchKMeans model
KMEANS_BATCH_SIZE = 4096