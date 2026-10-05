# Customer Churn Classification with Random Forest & XGBoost

A machine learning project using the IBM Telco Customer Churn dataset to predict whether a customer is likely to churn.

## Dataset

Recommended dataset:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Download the CSV and place it at:

`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

The project expects the standard IBM Telco dataset columns.

## Project workflow

1. Load and inspect the raw dataset.
2. Clean `TotalCharges`.
3. Remove `customerID` because it is an identifier.
4. Convert `Churn` from Yes/No to 1/0.
5. Split the data using stratification.
6. One-hot encode categorical features.
7. Train Random Forest.
8. Train XGBoost.
9. Compare Accuracy, Precision, Recall, F1 and ROC-AUC.
10. Save both complete pipelines as `.pkl` files.
11. Use XGBoost in the Flask prediction application.

## Setup

Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install packages:

```powershell
pip install -r requirements.txt
```

## Run preprocessing

```powershell
python src/data_preprocessing.py
```

This creates:

`data/processed/churn_cleaned.csv`

## Train the models

```powershell
python src/train_models.py
```

This creates:

- `models/random_forest.pkl`
- `models/xgboost.pkl`

## Test a prediction

```powershell
python src/predict.py
```

## Run the Flask app

```powershell
python app.py
```

Then open:

`http://127.0.0.1:5000`

## Main concepts

- Binary Classification
- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- One-Hot Encoding
- Train-Test Split
- Stratification
- Random Forest Classifier
- XGBoost Classifier
- Class Imbalance
- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Feature Importance
- Flask
