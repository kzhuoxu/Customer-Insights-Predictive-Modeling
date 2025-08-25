# JPMC Risk Program Senior Associate - Take-Home Project

This repository contains the solution for the take-home project, which involves two key tasks:
1.  **Supervised Classification:** Predicting whether an individual's income is greater than $50k based on US Census data.
2.  **Unsupervised Segmentation:** Identifying distinct customer personas from the same dataset for targeted marketing.

The project is structured as a reproducible Python application, emphasizing a clean, modular, and well-documented workflow. The final analysis and business recommendations are detailed in the accompanying `Project_Report.pdf`.

---

## Project Structure

The repository is organized into the following directories:

-   `data/`: Contains the raw census data and column information.
-   `notebooks/`: Includes the initial exploratory data analysis (EDA) and model prototyping.
-   `src/`: Contains the core, modularized Python source code for the project pipeline.
    -   `config.py`: Centralized configuration for file paths and model parameters.
    -   `data_loader.py`: Handles loading and initial cleaning of the dataset.
    -   `feature_engineering.py`: A library of functions for creating new features and preparing data for modeling.
    -   `train.py`: Script for training, evaluating, and saving the final classification model.
    -   `segment.py`: Script for generating and profiling customer segments.
-   `requirements.txt`: A list of all necessary Python packages to run the project.
-   `README.md`: This file.

---

## How to Run

### 1. Setup

First, clone the repository and install the required dependencies.

```bash
# Clone the project
git clone <your-repo-url>
cd jpmc_take_home

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Classification Model

To train the income prediction model, run the `train.py` script. This script will perform the full pipeline: load data, apply feature engineering, train a model (e.g., Gradient Boosting Classifier), evaluate it, and save the trained model artifact to disk.

```bash
python src/train.py
```
The script will print the final evaluation metrics (e.g., ROC AUC, F1-Score, and the confusion matrix) to the console.

### 3. Running the Segmentation Model

To generate the customer segments, run the `segment.py` script. This will preprocess the data, determine the optimal number of clusters, run K-Means, and save a CSV file containing the original data with an assigned cluster label for each individual.

```bash
python src/segment.py
```
This script will also print a summary profile of each generated customer segment to the console, describing their key demographic and employment characteristics.

---

## Core Methodologies

A brief overview of the key decisions and techniques used in this project:

*   **Feature Engineering:** A significant focus was placed on creating intuitive, powerful features from the raw data. This included:
    *   Grouping highly granular categorical variables (like `education` and `occupation`) into logical, lower-cardinality tiers.
    *   Creating composite features to capture concepts like `employment_status` and `investment_profile`.
    *   Validating feature definitions through comparative analysis (e.g., `crosstab`).
*   **Classification:** A Gradient Boosting Machine was selected as the final model due to its high performance on tabular data and its robustness to varied feature types. The model was evaluated on key business metrics like ROC AUC and F1-score to account for class imbalance.
*   **Segmentation:** K-Means clustering was used to segment the population. The optimal number of clusters was determined using the Elbow Method and Silhouette Analysis. Each resulting segment was profiled to create actionable business personas.

For a detailed walkthrough of the analysis, findings, and business recommendations, please see `Project_Report.pdf`.