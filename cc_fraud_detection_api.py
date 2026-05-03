from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Model path
MODEL_PATH = "xgb_bp_model.pkl"

# Verify model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Load trained model
model = joblib.load(MODEL_PATH)

metrics = {
    "Accuracy": "99.97%",
    "Precision": "95.40%",
    "Recall": "84.69%",
    "F1 Score": "89.73%"
}

# Initialize FastAPI
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Predict fraudulent transactions using tuned XGBoost model",
    version="1.0"
)

# Prediction threshold
FRAUD_THRESHOLD = 0.75


# Input schema
class TransactionData(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    Time: float


@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API is running successfully"
    }


@app.post("/predict")
def predict(transaction: TransactionData):
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([transaction.dict()])

        # # Apply scaling manually
        input_data["scaled_amount"] = input_data["Amount"]
        input_data["scaled_time"] = input_data["Time"]

        # # Drop original columns
        input_data.drop(["Time", "Amount"], axis=1, inplace=True)

        # Reorder columns to match training
        expected_order = [
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8',
            'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15',
            'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22',
            'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
            'scaled_amount', 'scaled_time'
        ]

        input_data = input_data[expected_order]

        # Predict probability
        probability = model.predict_proba(input_data)[0][1]
        print("probability:\n",probability)

        # Apply threshold
        prediction = int(probability > FRAUD_THRESHOLD)
        print("prediction:\n",prediction)

        return {
            "fraud_probability": round(float(probability), 4),
            "threshold": FRAUD_THRESHOLD,
            "is_fraud": prediction,
            "result": "Fraudulent Transaction" if prediction == 1 else "Legitimate Transaction"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
    

@app.get("/metrics")
def get_metrics():
    return metrics



    # run command
    # uvicorn filename_without_py:fastapi_object
    # uvicorn cc_fraud_detection_api:app --reload
    # http://127.0.0.1:8000/docs

    # Live
    # https://ml-credit-card-fraud.onrender.com