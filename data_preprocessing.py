"""
Data preprocessing for the IBM Telco Customer Churn dataset.

Input:
    data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv

Output:
    data/processed/churn_cleaned.csv
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "churn_cleaned.csv"


def load_and_clean_data(path=RAW_PATH):
    df = pd.read_csv(path)

    # TotalCharges contains blank strings for some new customers.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # CustomerID is an identifier, not a predictive feature.
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Convert target to binary.
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    return df


def main():
    df = load_and_clean_data()
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("Cleaned dataset saved to:", PROCESSED_PATH)
    print("Shape:", df.shape)
    print("\nChurn distribution:")
    print(df["Churn"].value_counts())
    print("\nMissing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])


if __name__ == "__main__":
    main()
