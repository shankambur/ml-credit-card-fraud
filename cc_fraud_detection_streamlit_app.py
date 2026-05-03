import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/")
print("Running using API ",API_URL)
# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

# -----------------------------
# Sample Transactions
# -----------------------------
fraud_sample = {
    "V1": -2.3122265423263,
    "V2": 1.95199201064158,
    "V3": -1.60985073229769,
    "V4": 3.9979055875468,
    "V5": -0.522187864667764,
    "V6": -1.42654531920595,
    "V7": -2.53738730624579,
    "V8": 1.39165724829804,
    "V9": -2.77008927719433,
    "V10": -2.77227214465915,
    "V11": 3.20203320709635,
    "V12": -2.89990738849473,
    "V13": -0.595221881324605,
    "V14": -4.28925378244217,
    "V15": 0.389724120274487,
    "V16": -1.14074717980657,
    "V17": -2.83005567450437,
    "V18": -0.0168224681808257,
    "V19": 0.416955705037907,
    "V20": 0.126910559061474,
    "V21": 0.517232370861764,
    "V22": -0.0350493686052974,
    "V23": -0.465211076182388,
    "V24": 0.320198198514526,
    "V25": 0.0445191674731724,
    "V26": 0.177839798284401,
    "V27": 0.261145002567677,
    "V28": -0.143275874698919,
    "Amount": -0.35322939296682354,
    "Time": -1.9880335064229064
}

legit_sample = {
    "V1": -1.359807,
    "V2": -0.072781,
    "V3": 2.536346,
    "V4": 1.378155,
    "V5": -0.338321,
    "V6": 0.462388,
    "V7": 0.239599,
    "V8": 0.098698,
    "V9": 0.363787,
    "V10": 0.090794,
    "V11": -0.5516,
    "V12": -0.617801,
    "V13": -0.99139,
    "V14": -0.311169,
    "V15": 1.468177,
    "V16": -0.470401,
    "V17": 0.207971,
    "V18": 0.025791,
    "V19": 0.403993,
    "V20": 0.251412,
    "V21": -0.018307,
    "V22": 0.277838,
    "V23": -0.110474,
    "V24": 0.066928,
    "V25": 0.128539,
    "V26": -0.189115,
    "V27": 0.133558,
    "V28": -0.021053,
    "Amount": 0.244964,
    "Time": -1.996583
}

# -----------------------------
# Title
# -----------------------------

st.title("💳 Credit Card Fraud Detection App")
st.write("Paste transaction JSON, load sample transactions, and predict fraud.")


# -----------------------------
# API Health Check
# -----------------------------
try:
    health_response = requests.get(f"{API_URL}/")

    if health_response.status_code == 200:
        st.success("✅ FastAPI backend is running")
    else:
        st.warning("⚠️ FastAPI backend responded unexpectedly")

except:
    st.error("❌ FastAPI backend is not running. Please start the API server.")

# -----------------------------
# Fetch Model Metrics
# -----------------------------
try:
    metrics_response = requests.get(f"{API_URL}/metrics")

    if metrics_response.status_code == 200:
        model_metrics = metrics_response.json()
    else:
        model_metrics = None

except:
    model_metrics = None

# -----------------------------
# Model Metrics Dashboard
# -----------------------------
if model_metrics:
    st.subheader("📊 Model Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
     st.markdown("**Accuracy**")
     st.write(model_metrics.get("Accuracy", "Pending"))

    with col2:
     st.markdown("**Precision**")
     st.write(model_metrics.get("Precision", "Pending"))

    with col3:
     st.markdown("**Recall**")
     st.write(model_metrics.get("Recall", "Pending"))

    with col4:
     st.markdown("**F1 Score**")
     st.write(model_metrics.get("F1 Score", "Pending"))


# -----------------------------
# Session State
# -----------------------------
if "json_input" not in st.session_state:
    st.session_state.json_input = ""

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# -----------------------------
# Sample Buttons
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Load Fraud Sample"):
        st.session_state.json_input = json.dumps(fraud_sample, indent=2)

with col2:
    if st.button("Load Legitimate Sample"):
        st.session_state.json_input = json.dumps(legit_sample, indent=2)

with col3:
    if st.button("Clear JSON"):
        st.session_state.json_input = ""

# -----------------------------
# JSON Input
# -----------------------------
json_input = st.text_area(
    "Paste transaction JSON here:",
    key="json_input",
    height=350
)


# Default feature structure
features = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]

# Initialize session state
if "input_data" not in st.session_state:
    print("Initialize session state")
    st.session_state.input_data = {feature: 0.0 for feature in features}


    
# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict Fraud"):
    print("stream button predict")

    try:
        parsed_data = json.loads(json_input)

        # Remove Class if included
        parsed_data.pop("Class", None)

        # Validate required fields
        missing_fields = [f for f in features if f not in parsed_data]

        if missing_fields:
            st.error(f"Missing fields: {missing_fields}")
        else:
            st.session_state.input_data = {
                feature: float(parsed_data[feature]) for feature in features
            }
            st.success("JSON data loaded successfully!")

    except Exception as e:
        st.error(f"Invalid JSON format: {str(e)}")


    print("parsed_data successfully:\n",parsed_data)
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=parsed_data
        )

        if response.status_code == 200:
            result = response.json()
            st.subheader("Prediction Result")
            fraud_probability = float(result["fraud_probability"])
            st.write(f"**Fraud Probability:** {fraud_probability:.4f}")
            # Progress bar expects 0–100 integer
            st.progress(int(fraud_probability * 100))
            # -----------------------------
            # Risk Level Classification
            # -----------------------------
            if fraud_probability >= 0.75:
                risk_level = "🔴 High Risk"
            elif fraud_probability >= 0.40:
                risk_level = "🟠 Moderate Risk"
            else:
                risk_level = "🟢 Low Risk"

            st.write(f"**Risk Level:** {risk_level}")
            st.session_state.prediction_history.append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Prediction": result["result"],
                "Fraud Probability": result["fraud_probability"],
                "Threshold": result["threshold"],
                "Risk Level": risk_level
            })

            st.write(f"**Threshold:** {result['threshold']}")
            st.write(f"**Prediction:** {result['result']}")

            if result["is_fraud"] == 1:
                st.error("⚠️ Fraudulent Transaction Detected!")
            else:
                st.success("✅ Legitimate Transaction")

        else:
            st.error(f"API Error: {response.text}")

    except Exception as e:
        st.error(f"Connection failed: {str(e)}")


# -----------------------------
# Prediction History
# -----------------------------
if st.session_state.prediction_history:
    st.subheader("📜 Prediction History")

    # Convert history to DataFrame
    history_df = pd.DataFrame(st.session_state.prediction_history)

    # Display table
    st.table(history_df)


if st.button("Clear Prediction History"):
    st.session_state.prediction_history = []
    st.rerun()