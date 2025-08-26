# %% [markdown]
# # Classification

# %%
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown]
# Load Data

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
# Data Split for Training

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_featured, 
    y, 
    test_size=config.TEST_SIZE, 
    random_state=config.RANDOM_STATE,
    stratify=y # To maintain class distribution in splits
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"\nTraining set target distribution:\n{y_train.value_counts(normalize=True)}")
print(f"\nTest set target distribution:\n{y_test.value_counts(normalize=True)}")

# %% [markdown]
# Preprocessing

# %%
from src.preprocessing import create_preprocessor

NUMERICAL_COLS = X_featured.select_dtypes(include='number').columns.tolist()
CATEGORICAL_COLS = X_featured.select_dtypes(include='object').columns.tolist()
print(f"Numerical columns: {NUMERICAL_COLS}")
print(f"Categorical columns: {CATEGORICAL_COLS}")

preprocessor = create_preprocessor(NUMERICAL_COLS, CATEGORICAL_COLS)

# Fit the preprocessor ONLY on the training data
preprocessor.fit(X_train)

# Transform both the training and testing data
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# The output is a NumPy array, ready for the model
print(f"\nShape of processed training data: {X_train_processed.shape}")
print(f"Shape of processed testing data: {X_test_processed.shape}")

# %%
# get the feature names after preprocessing
def get_feature_names(preprocessor):
    feature_names = []
    
    # Numerical features
    if 'num' in preprocessor.named_transformers_:
        num_features = preprocessor.named_transformers_['num'].feature_names_in_.tolist()
        feature_names.extend(num_features)
    
    # Categorical features
    if 'cat' in preprocessor.named_transformers_:
        cat_transformer = preprocessor.named_transformers_['cat']
        if hasattr(cat_transformer, 'get_feature_names_out'):
            cat_features = cat_transformer.get_feature_names_out(preprocessor.transformers_[1][2])
            feature_names.extend(cat_features)
        else:
            # Fallback if get_feature_names_out is not available
            cat_features = preprocessor.transformers_[1][2]
            feature_names.extend(cat_features)
    
    return feature_names    
feature_names = get_feature_names(preprocessor)
print(f"\nTotal number of features after preprocessing: {len(feature_names)}")
print(f"First 10 feature names: {feature_names[:10]}")

# %% [markdown]
# ## Baseline
# Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# The full pipeline by combining our preprocessor with the classifier
lr_pipeline_standard = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=config.RANDOM_STATE, max_iter=1000))
])

# Perform 5-fold cross-validation
cv_auc_standard = cross_val_score(
    lr_pipeline_standard, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1
)

cv_f1_standard = cross_val_score(
    lr_pipeline_standard, X_train, y_train, cv=5, scoring='f1_weighted', n_jobs=-1
)

# To see how the model performs for underrepresented classes
cv_f1_macro = cross_val_score(
    lr_pipeline_standard, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1
)

print("\nResults for Standard Logistic Regression (5-fold CV):")
print(f"  Mean ROC AUC: {np.mean(cv_auc_standard):.4f} (Std: {np.std(cv_auc_standard):.4f})")
print(f"  Mean F1-Score (Weighted): {np.mean(cv_f1_standard):.4f} (Std: {np.std(cv_f1_standard):.4f})")
print(f"  Mean F1-Score (Macro): {np.mean(cv_f1_macro):.4f} (Std: {np.std(cv_f1_macro):.4f})")

# %% [markdown]
# ## Tree-Based Models

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import time

models_to_test = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=config.RANDOM_STATE,
        class_weight='balanced'
    ),
    "Random Forest": RandomForestClassifier(
        random_state=config.RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1  
    ),
    "LightGBM": LGBMClassifier(
        random_state=config.RANDOM_STATE,
        is_unbalance=True, 
        n_jobs=-1
    )
}

results = {}

for name, model in models_to_test.items():
    start_time = time.time()
    print(f"\nTraining {name}...")

    # Create the full Scikit-Learn pipeline
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Perform 5-fold cross-validation for ROC AUC and F1-Score
    cv_auc = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1_weighted', n_jobs=-1)
    cv_macro_f1 = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Store the results
    results[name] = {
        "Mean ROC AUC": cv_auc.mean(),
        "Std ROC AUC": cv_auc.std(),
        "Mean F1 (Weighted)": cv_f1.mean(),
        "Std F1 (Weighted)": cv_f1.std(),
        "Mean F1 (Macro)": cv_macro_f1.mean(),
        "Std F1 (Macro)": cv_macro_f1.std(),
        "Training Time (s)": training_time
    }
    
    print(f"{name} trained in {training_time:.2f} seconds.")

# %%
lightgbm_smote_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=config.RANDOM_STATE)),
    ('classifier', LGBMClassifier(random_state=config.RANDOM_STATE, n_jobs=-1))
])

start_time = time.time()

cv_auc = cross_val_score(lightgbm_smote_pipeline, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
cv_f1 = cross_val_score(lightgbm_smote_pipeline, X_train, y_train, cv=5, scoring='f1_weighted', n_jobs=-1)
cv_macro_f1 = cross_val_score(lightgbm_smote_pipeline, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1)

end_time = time.time()
training_time = end_time - start_time

results["LightGBM_with_SMOTE"] = {
    "Mean ROC AUC": cv_auc.mean(),
    "Std ROC AUC": cv_auc.std(),
    "Mean F1 (Weighted)": cv_f1.mean(),
    "Std F1 (Weighted)": cv_f1.std(),
    "Mean F1 (Macro)": cv_macro_f1.mean(),
    "Std F1 (Macro)": cv_macro_f1.std(),
    "Training Time (s)": training_time
}

# %%
results_df = pd.DataFrame(results).T # .T transposes the DataFrame
results_df.sort_values(by='Mean ROC AUC', ascending=False, inplace=True)

print("\nModel Performance Results (sorted by Mean ROC AUC):")
display(results_df)

# %% [markdown]
# Tuned hyperparameters for the best model (LightGBM)

# %%
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Tuned hyperparameters for the best model (LightGBM)
lgbm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(
        random_state=config.RANDOM_STATE,
        is_unbalance=True, # Continue handling class imbalance
        n_jobs=-1
    ))
])

# Define the parameter distribution to sample from
param_dist = {
    'classifier__n_estimators': randint(100, 1000),
    'classifier__learning_rate': uniform(0.01, 0.2),
    'classifier__num_leaves': randint(20, 50),
    'classifier__max_depth': randint(5, 20),
    'classifier__reg_alpha': uniform(0.1, 0.9), # L1 regularization
    'classifier__reg_lambda': uniform(0.1, 0.9)  # L2 regularization
}

n_iter_search = 25

random_search = RandomizedSearchCV(
    lgbm_pipeline,
    param_distributions=param_dist,
    n_iter=n_iter_search,
    cv=5, # 5-fold cross-validation
    scoring='roc_auc', # primary metric
    random_state=config.RANDOM_STATE,
    n_jobs=-1,
    # verbose=2 # show progress
)

# Fit the search on the TRAINING data
random_search.fit(X_train, y_train)

# %%
import json

print("Best cross-validated ROC AUC score from search: {:.4f}".format(random_search.best_score_))
print("\nBest parameters found:")
print(random_search.best_params_)

print(f"\nSaving best parameters to: {config.HYPERPARAMETERS_FILE}")
with open(config.HYPERPARAMETERS_FILE, 'w') as f:
    json.dump(random_search.best_params_, f, indent=4) # indent=4 makes it human-readable

print("Parameters saved successfully.")

# Compare to the default LightGBM score from the fine-tuning results
default_lgbm_score = results_df.loc['LightGBM', 'Mean ROC AUC']
print(f"\nDefault LightGBM score was: {default_lgbm_score:.4f}")
improvement = (random_search.best_score_ - default_lgbm_score) / default_lgbm_score * 100
print(f"Improvement from tuning: {improvement:.2f}%")

# %%
from sklearn.metrics import roc_auc_score, classification_report, ConfusionMatrixDisplay

best_model = random_search.best_estimator_

# Make predictions on the unseen test data
y_pred_test = best_model.predict(X_test)
y_pred_proba_test = best_model.predict_proba(X_test)[:, 1] # Probabilities for the positive class

# --- Calculate and Print Final Metrics ---
final_roc_auc = roc_auc_score(y_test, y_pred_proba_test)
print(f"Final ROC AUC on Test Set: {final_roc_auc:.4f}")

print("\n--- Classification Report on Test Set ---")
print(classification_report(y_test, y_pred_test, target_names=['Income < $50k', 'Income > $50k']))


# %% [markdown]
# Experiment with f1_macro

# %%
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Tuned hyperparameters for the best model (LightGBM)
lgbm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(
        random_state=config.RANDOM_STATE,
        is_unbalance=True, # Continue handling class imbalance
        n_jobs=-1
    ))
])

# Define the parameter distribution to sample from
param_dist = {
    'classifier__n_estimators': randint(100, 1000),
    'classifier__learning_rate': uniform(0.01, 0.2),
    'classifier__num_leaves': randint(20, 50),
    'classifier__max_depth': randint(5, 20),
    'classifier__reg_alpha': uniform(0.1, 0.9), # L1 regularization
    'classifier__reg_lambda': uniform(0.1, 0.9)  # L2 regularization
}

n_iter_search = 25

random_search = RandomizedSearchCV(
    lgbm_pipeline,
    param_distributions=param_dist,
    n_iter=n_iter_search,
    cv=5, # 5-fold cross-validation
    scoring='f1_macro', # alternative metric
    random_state=config.RANDOM_STATE,
    n_jobs=-1,
    # verbose=2 # show progress
)

# Fit the search on the TRAINING data
random_search.fit(X_train, y_train)

# %%
import json

print("Best cross-validated f1 macro score from search: {:.4f}".format(random_search.best_score_))
print("\nBest parameters found:")
print(random_search.best_params_)

print(f"\nSaving best parameters to: {config.HYPERPARAMETERS_FILE_alt}")
with open(config.HYPERPARAMETERS_FILE_alt, 'w') as f:
    json.dump(random_search.best_params_, f, indent=4) # indent=4 makes it human-readable

print("Parameters saved successfully.")

# Compare to the default LightGBM score 
default_lgbm_score = results_df.loc['LightGBM', 'Mean F1 Macro']
print(f"\nDefault LightGBM score was: {default_lgbm_score:.4f}")
improvement = (random_search.best_score_ - default_lgbm_score) / default_lgbm_score * 100
print(f"Improvement from tuning: {improvement:.2f}%")

# %%



