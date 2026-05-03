#!/bin/bash
uvicorn cc_fraud_detection_api:app --host 0.0.0.0 --port 8000 &
streamlit run cc_fraud_detection_streamlit_app.py --server.port 8501 --server.address 0.0.0.0