from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "xgboost.pkl"

app = Flask(__name__)


def get_value(name):
    value = request.form.get(name, "")
    return value


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        try:
            customer = {
                "gender": get_value("gender"),
                "SeniorCitizen": int(get_value("SeniorCitizen")),
                "Partner": get_value("Partner"),
                "Dependents": get_value("Dependents"),
                "tenure": float(get_value("tenure")),
                "PhoneService": get_value("PhoneService"),
                "MultipleLines": get_value("MultipleLines"),
                "InternetService": get_value("InternetService"),
                "OnlineSecurity": get_value("OnlineSecurity"),
                "OnlineBackup": get_value("OnlineBackup"),
                "DeviceProtection": get_value("DeviceProtection"),
                "TechSupport": get_value("TechSupport"),
                "StreamingTV": get_value("StreamingTV"),
                "StreamingMovies": get_value("StreamingMovies"),
                "Contract": get_value("Contract"),
                "PaperlessBilling": get_value("PaperlessBilling"),
                "PaymentMethod": get_value("PaymentMethod"),
                "MonthlyCharges": float(get_value("MonthlyCharges")),
                "TotalCharges": float(get_value("TotalCharges")),
            }

            model = joblib.load(MODEL_PATH)
            df = pd.DataFrame([customer])

            prediction = int(model.predict(df)[0])
            probability = float(model.predict_proba(df)[0][1]) * 100

            result = {
                "label": "Likely to Churn" if prediction == 1 else "Likely to Stay",
                "probability": f"{probability:.2f}%",
            }

        except FileNotFoundError:
            error = "Model not found. Run the preprocessing and training scripts first."
        except Exception as exc:
            error = f"Please check your inputs. Error: {exc}"

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True)
