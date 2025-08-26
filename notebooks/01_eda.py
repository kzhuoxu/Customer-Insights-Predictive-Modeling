# %%
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

# %% [markdown]
# # Objective 1
# As a data scientist, you are tasked by your retail business client with identifying two
# groups of people for marketing purposes:
# 
# - People who earn an income of less than $50,000 and those who earn more than
# $50,000. 
# - To assist in this pursuit, Walmart has developed a means of accessing 40
# different demographic and employment related variables for any person they are
# interested in marketing to. 
# - Additionally, Walmart has been able to compile a
# dataset that provides gold labels for a variety of observations of these 40
# variables within the population. 
# - Using the dataset given, train and validate a
# classifier that predicts this outcome.

# %% [markdown]
# ## Data
# [Census-Income (KDD)](https://archive.ics.uci.edu/dataset/117/census+income+kdd)
# weighted census data extracted from the 1994 and 1995
# 
# Each line of the data set (censusbureau.data) contains 
# - 40 demographic and employment related variables as well as 
# - a weight for the observation 
#     - The weight indicates the relative distribution of people in the general population that each record represents due to stratified sampling
# - a label for each observation, 
#     - which indicates whether a particular population component had an income that is greater than or less than $50k

# %% [markdown]
# ### 1. Load the data

# %%
with open('../data/census-bureau.columns', 'r') as f:
    columns = f.read().splitlines()
    print(columns)

df = pd.read_csv('../data/census-bureau.data', header=None, names=columns)
print(df.shape)
print(df.info())
df.head()

# %%
print(df.shape)
df.info()
df.head()

# %%
# Separate the weight and label columns for the dataset
weights = df['weight']
labels = df['label']
features = df.drop(columns=['weight', 'label'])
features.head()

# %%
# One-hot encode label column
labels = labels.map({'- 50000.': 0, '50000+.': 1})

# %% [markdown]
# ### 2. EDA

# %% [markdown]
# #### Label 

# %%
# see if there is class imbalance
df['label'].value_counts(normalize=True)

# %%
# Inspect label distribution using stratified sampling weights
print(f"Percentage of people that earn {df['label'].unique()[0]} among all population (with weights): {weights[df['label'] == df['label'].unique()[0]].sum() / weights.sum():.4f}.")

# %% [markdown]
# #### Numeric Features

# %%
# Insepect the number of unique values for each feature
features.nunique().sort_values(ascending=True)

# %%
# Identify numeric features
numeric_cols = features.nunique()[features.nunique() > 52].index.tolist()
print(numeric_cols)

# %%
# Is there any missing data in numeric features?
features[numeric_cols].isnull().sum()

# %%
features['major industry code'].value_counts()

# %%
features['tax filer stat'].value_counts()

# %%
# Plot pairplot for numeric features and color by label
plt.figure(figsize=(12, 10))
sns.pairplot(df[numeric_cols + ['label']], hue='label', diag_kind='hist', plot_kws={'alpha': 0.5})
plt.suptitle('Pairplot for Numeric Features Colored by Label', y=1.02)
plt.show()

# %% [markdown]
# Anlaysis for the pairplot:
# - On the diagonals for `wage per hour`, `capital gains`, `capital losses`, and `dividends` from stocks: the histograms are overwhelmingly dominated by a single bar at zero. 

# %%
# Plot correlation heatmap for numeric features
plt.figure(figsize=(10, 8))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap for Numeric Features')
plt.show()

# %% [markdown]
# #### Categorical Features

# %%
# Identify one-hot encoded features: those that are not numeric
categorical_cols = [ col for col in features.columns if col not in numeric_cols ]
print(len(categorical_cols))
features[categorical_cols].nunique().sort_values(ascending=True)

# %%
# Understanding the reason for "Not in universe": histogram for categorical features
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=col, hue='label')
    plt.title(f'Countplot of {col} by Label')
    plt.xticks(rotation=45)
    plt.show()

# %%
features['citizenship'].value_counts()

# %% [markdown]
# some categorical features have numerical value labels, use codebook to map them to string labels
# - https://cps.ipums.org/cps/resources/codebooks/cpsmar96.pdf
# - VETYN veterans_benefits: 0 1 2
#     - page 128
#     - 0 .Not in universe: missing data?
#     - 1 .Yes
#     - 2 .No 
# - SEOTR own_business_or_self_employed: 0 1 2
#     - page 206
#     - 0 .Not in universe: missing data?
#     - 1 .Yes
#     - 2 .No 
# - NOEMP num_persons_worked_for_employer: 0 1 2 3 4 5 6
#     - page 203
#     - 0 .Not in universe
#     - 1 .Under 10
#     - 2 .10 - 24
#     - 3 .25 - 99
#     - 4 .100 - 499
#     - 5 .500 - 999
#     - 6 .1000+ 

# %%
features[categorical_cols].nunique().sum()

# %%
# Is there any missing data in categorical features? show only those with missing values
features[categorical_cols].isnull().sum()[features[categorical_cols].isnull().sum() > 0]

# %%
# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False, 
                        min_frequency=500,
                        drop='first',  # drop first to avoid dummy variable trap
                        handle_unknown='ignore') # fill with nan

features_encoded = encoder.fit_transform(features[categorical_cols])
encoded_cols = encoder.get_feature_names_out(categorical_cols)
features_encoded = pd.DataFrame(features_encoded, columns=encoded_cols)
features_encoded.head()

# %% [markdown]
# #### Feature Engineering

# %% [markdown]
# 1. How is this person doing financially?
#     - own business or self employed  
#     - num persons worked for employer   
#     - tax filer stat           
#     - full or part time employment stat   
#     - weeks worked in year
#     - capital losses
#     - capital gains
#     - wage per hour
#     - dividends from stocks    

# %%
# net capital gain and loss
features['net capital gains'] = features['capital gains'] - features['capital losses']
# has dividends or not
features['has dividends'] = (features['dividends from stocks'] > 0).astype(int)
# total income
features['total income'] = features['weeks worked in year'] * features['wage per hour'] * 8 * 5 \
                            + features['net capital gains'] \
                            + features['dividends from stocks']

# %%
features['full or part time employment stat'].value_counts()

# %%
# Employment Status:
def refined_employment_status(x):
    # Group 1: Currently Employed, Full-Time
    if x == 'Full-time schedules':
        return 'full-time'
    
    # Group 2: Currently Employed, Part-Time (regardless of the reason)
    elif x in ['PT for econ reasons usually FT', 
               'PT for non-econ reasons usually FT', 
               'PT for econ reasons usually PT']:
        return 'part-time'
        
    # Group 3: Currently Unemployed (actively seeking work)
    elif x in ['Unemployed full-time', 'Unemployed part- time']:
        return 'unemployed'
        
    # Group 4: Not in the labor force (not working, not seeking work)
    else: # This will catch 'Not in labor force', 'Disabled', etc.
        return 'not-in-labor-force'
    
features['refined employment status'] = features['full or part time employment stat'].apply(refined_employment_status)
features['refined employment status'].value_counts()

# %%
features['refined employment status from hours worked'] = features['weeks worked in year'].apply(
    lambda x: "full-time" if x > 50 else "part-time" if x > 0 else "not-in-labor-force"
)
# compare the two refined employment status
pd.crosstab(features['refined employment status'], features['refined employment status from hours worked'])

# %% [markdown]
# To create a robust feature for employment, we explored two definitions: one based on the respondent's self-identified employment status and another based on the literal number of weeks worked. 
# 
# A crosstabulation (see table above) revealed significant discrepancies between the two approaches. 
# The self-identified status proved to be a more nuanced and powerful feature. For example, it successfully isolated a distinct 'unemployed' population that the hours-worked metric could not. It also highlighted a large cohort of individuals (~36,000) who worked a full year but do not identify as being in the formal labor force, suggesting a more complex employment situation.
# Given its ability to capture these critical subtleties that are highly relevant for assessing financial stability, we have proceeded with the refined employment status feature for our final model.

# %% [markdown]
# 2. Life Stages
#      - Age
#      - education         
#      - enroll in edu inst last wk   
#      - marital stat 
#      - detailed household summary in household 
#      - family members under 18   

# %%
# Age squared
features['age_squared'] = features['age'] ** 2
# Age grouping
def age_grouping(age):
    if age < 18:
        return 'under_18'
    elif 18 <= age < 30:
        return '18-29'
    elif 30 <= age < 45:
        return '30-44'
    elif 45 <= age < 60:
        return '45-59'
    elif 60 <= age < 75:
        return '60-74'
    else:
        return '75_and_above'
features['age_group'] = features['age'].apply(age_grouping)
features['age_group'].value_counts()

# %%
features['marital stat'].value_counts()

# %%
# marital status and age interaction
def marital_age_interaction(row):
    if row['marital stat'] in ['Married-civilian spouse present', 'Married-A F spouse present']:
        if row['age'] < 30:
            return 'young_married'
        elif 30 <= row['age'] < 60:
            return 'midage_married'
        else:
            return 'senior_married'
    else:
        if row['age'] < 30:
            return 'young_not_married'
        elif 30 <= row['age'] < 60:
            return 'midage_not_married'
        else:
            return 'senior_not_married'
features['marital_age_group'] = features.apply(marital_age_interaction, axis=1)
features['marital_age_group'].value_counts()

# %%
features['education'].value_counts()

# %%
# education level grouping
def group_education(education_level):
    # Group 1: Children
    if education_level == 'Children':
        return 'Children'

    # Group 2: No High School Diploma
    elif education_level in ['Less than 1st grade', '1st 2nd 3rd or 4th grade', 
                             '5th or 6th grade', '7th and 8th grade', 
                             '9th grade', '10th grade', '11th grade', 
                             '12th grade no diploma']:
        return 'No-High-School'

    # Group 3: High School Graduate
    elif education_level == 'High school graduate':
        return 'High-School-Graduate'

    # Group 4: Some College (including Associates)
    elif education_level in ['Some college but no degree', 
                             'Associates degree-occup /vocational', 
                             'Associates degree-academic program']:
        return 'Some-College'

    # Group 5: Bachelors Degree
    elif education_level == 'Bachelors degree(BA AB BS)':
        return 'Bachelors-Degree'

    # Group 6: Advanced Degree
    elif education_level in ['Masters degree(MA MS MEng MEd MSW MBA)', 
                             'Prof school degree (MD DDS DVM LLB JD)',
                             'Doctorate degree(PhD EdD)']:
        return 'Advanced-Degree'
    
    # Fallback for any unexpected values
    else:
        return 'Other'

features['education_group'] = features['education'].apply(group_education)
print(features['education_group'].value_counts())

# %%
# education and age interaction
def education_age_interaction(row):
    if row['education'] in ['Bachelors', 'Masters', 'Doctorate', 'Prof-school']:
        if row['age'] < 30:
            return 'young_high_edu'
        elif 30 <= row['age'] < 60:
            return 'midage_high_edu'
        else:
            return 'senior_high_edu'
    else:
        if row['age'] < 30:
            return 'young_low_edu'
        elif 30 <= row['age'] < 60:
            return 'midage_low_edu'
        else:
            return 'senior_low_edu'

# %% [markdown]
# 3. Immigration Status
#     - migration prev res in sunbelt    
#     - state of previous residence   
#     - citizenship      
#     - race
#     - hispanic origin 
#     - region of previous residence       
#     - country of birth father
#     - country of birth mother
#     - country of birth self
#     - migration code-change in reg
#     - migration code-move within reg
#     - migration code-change in msa          

# %%
# first generation immigrant & second generation
    # - country of birth father
    # - country of birth mother
    # - country of birth self
features['immigrant_status'] = features.apply(lambda row: 
    'first_gen' 
        if row['country of birth self'] != 'United-States' 
            and row['country of birth father'] != 'United-States' 
            and row['country of birth mother'] != 'United-States'
        else ('second_gen' 
            if row['country of birth father'] != 'United-States' 
                or row['country of birth mother'] != 'United-States' 
            else 'native'), axis=1)
features['immigrant_status'].value_counts()

# %%
# write this feature engineering pipeline into a function for later use
def feature_engineering(features):
    # net capital gain and loss
    features['net capital gains'] = features['capital gains'] - features['capital losses']
    # has dividends or not
    features['has dividends'] = (features['dividends from stocks'] > 0).astype(int)
    # total income
    features['total income'] = features['weeks worked in year'] * features['wage per hour'] * 8 * 5 \
                                + features['net capital gains'] \
                                + features['dividends from stocks']

    # refined employment status
    def refined_employment_status(x):
        if x == 'Full-time schedules':
            return 'full-time'
        elif x in ['PT for econ reasons usually FT', 
                   'PT for non-econ reasons usually FT', 
                   'PT for econ reasons usually PT']:
            return 'part-time'
        elif x in ['Unemployed full-time', 'Unemployed part- time']:
            return 'unemployed'
        else:
            return 'not-in-labor-force'
        
    features['refined employment status'] = features['full or part time employment stat'].apply(refined_employment_status)

    # Age squared
    features['age_squared'] = features['age'] ** 2
    # Age grouping
    def age_grouping(age):
        if age < 18:
            return 'under_18'
        elif 18 <= age < 30:
            return '18-29'
        elif 30 <= age < 45:
            return '30-44'
        elif 45 <= age < 60:
            return '45-59'
        elif 60 <= age < 75:
            return '60-74'
        else:
            return '75_and_above'
    features['age_group'] = features['age'].apply(age_grouping)

    # marital status and age interaction
    def marital_age_interaction(row):
        if row['marital stat'] in ['Married-civilian spouse present', 'Married-A F spouse present']:
            if row['age'] < 30:
                return 'young_married'
            elif 30 <= row['age'] < 60:
                return 'midage_married'
            else:
                return 'senior_married'
        else:
            if row['age'] < 30:
                return 'young_not_married'
            elif 30 <= row['age'] < 60:
                return 'midage_not_married'
            else:
                return 'senior_not_married'
    features['marital_age_interaction'] = features.apply(marital_age_interaction, axis=1)
    
    # education level grouping
    def group_education(education_level):
        # Group 1: Children
        if education_level == 'Children':
            return 'Children'

        # Group 2: No High School Diploma
        elif education_level in ['Less than 1st grade', '1st 2nd 3rd or 4th grade', 
                                '5th or 6th grade', '7th and 8th grade', 
                                '9th grade', '10th grade', '11th grade', 
                                '12th grade no diploma']:
            return 'No-High-School'

        # Group 3: High School Graduate
        elif education_level == 'High school graduate':
            return 'High-School-Graduate'

        # Group 4: Some College (including Associates)
        elif education_level in ['Some college but no degree', 
                                'Associates degree-occup /vocational', 
                                'Associates degree-academic program']:
            return 'Some-College'

        # Group 5: Bachelors Degree
        elif education_level == 'Bachelors degree(BA AB BS)':
            return 'Bachelors-Degree'

        # Group 6: Advanced Degree
        elif education_level in ['Masters degree(MA MS MEng MEd MSW MBA)', 
                                'Prof school degree (MD DDS DVM LLB JD)',
                                'Doctorate degree(PhD EdD)']:
            return 'Advanced-Degree'
        
        # Fallback for any unexpected values
        else:
            return 'Other'

    features['education_group'] = features['education'].apply(group_education)
    print(features['education_group'].value_counts())
    
    # education and age interaction
    def education_age_interaction(row):
        if row['education'] in ['Bachelors', 'Masters', 'Doctorate', 'Prof-school']:
            if row['age'] < 30:
                return 'young_high_edu'
            elif 30 <= row['age'] < 60:
                return 'midage_high_edu'
            else:
                return 'senior_high_edu'
        else:
            if row['age'] < 30:
                return 'young_low_edu'
            elif 30 <= row['age'] < 60:
                return 'midage_low_edu'
            else:
                return 'senior_low_edu'

    return features

# %% [markdown]
# ### 3. Create Training and Test Data Sets

# %%
X = features_encoded
y = labels

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
# ### 4. Model Training

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import config
from sklearn.preprocessing import preprocessor


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

# %%



