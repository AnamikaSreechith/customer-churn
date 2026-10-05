"""
Predict churn for a new customer using the trained XGBoost model.

Run:
    python src/predict.py
"""

from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "xgboost.pkl"


def predict_customer(customer_data):
    model = joblib.load(MODEL_PATH)
    customer_df = pd.DataFrame([customer_data])

    prediction = int(model.predict(customer_df)[0])
    probability = float(model.predict_proba(customer_df)[0][1])

    return {
        "prediction": "Will Churn" if prediction == 1 else "Will Stay",
        "churn_probability": round(probability * 100, 2),
    }


if __name__ == "__main__":
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 55.5,
        "TotalCharges": 111.0,
    }

    print(predict_customer(sample_customer))
