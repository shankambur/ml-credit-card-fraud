# 💳 Credit Card Fraud Detection System

An end-to-end Machine Learning fraud detection application built using:

* XGBoost (Tuned Model)
* FastAPI (Backend API)
* Streamlit (Frontend Dashboard)

---

## 🚀 Project Overview

This project detects fraudulent credit card transactions using a tuned XGBoost classifier trained on highly imbalanced transaction data.

The system includes:

### Backend:

* FastAPI deployment
* Fraud prediction API
* Health check endpoint

### Frontend:

* Streamlit dashboard
* Fraud/legitimate sample loaders
* Real-time predictions
* Risk classification
* Prediction history
* Model performance dashboard

---

## 📊 Final Model Performance (Threshold = 0.75)

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 99.97% |
| Precision | 95.40% |
| Recall    | 84.69% |
| F1 Score  | 89.73% |

---

## 🛠️ Tech Stack

* Python
* XGBoost
* Scikit-learn
* FastAPI
* Streamlit
* Pandas
* NumPy
* SHAP
* Joblib

---

## 📂 Project Structure

```bash
ml-credit-card-fraud/
│
├── cc_fraud_detection_api.py
├── cc_fraud_detection_streamlit_app.py
├── xgb_bp_model.pkl
├── requirements.txt
├── start.sh
└── README.md
```

---

## ▶️ Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start API

```bash
uvicorn cc_fraud_detection_api:app --reload
```

### 3. Start Streamlit App

```bash
streamlit run cc_fraud_detection_streamlit_app.py
```

---

## 🌐 Deployment

### start.sh

```bash
#!/bin/bash
uvicorn cc_fraud_detection_api:app --host 0.0.0.0 --port 8000 &
streamlit run cc_fraud_detection_streamlit_app.py --server.port $PORT --server.address 0.0.0.0
```

## API endpoints
API_URL = https://ml-credit-card-fraud.onrender.com/docs
### POST /predict


---

## 🎯 Key Features

* Fraud probability scoring
* Risk classification
* Fraud threshold tuning
* API deployment
* Interactive UI
* Prediction history
* Model metrics dashboard
* Demo-ready sample transactions

---

## 📌 Future Enhancements

* SHAP explainability
* Batch CSV prediction
* Docker containerization
* Cloud deployment (Render/AWS)
* Authentication
* Monitoring dashboards

---

## 👨‍💻 Author

Built as part of advanced AI/ML engineering portfolio development.
