import streamlit as st
import requests
import os

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")

st.title("Credit Risk Predictor")

age = st.number_input("Age", 18, 100, 28)
income = st.number_input("Income", 0, step=1000, value=50000)
loan = st.number_input("Loan Amount", 0, step=1000, value=15000)
rate = st.number_input("Interest Rate (%)", 0.0, value=12.5)
ratio = st.number_input("Loan % of Income", 0.0, value=0.3)

if st.button("Predict Risk"):
    payload = {
        "person_age": age,
        "person_income": income,
        "loan_amnt": loan,
        "loan_int_rate": rate,
        "loan_percent_income": ratio
    }

    with st.spinner("Predicting..."):
        API_URL = os.getenv("API_URL", "http://localhost:8000")

        res = requests.post(f"{API_URL}/predict", json=payload)


    if res.status_code == 200:
        risk = res.json()["default_risk"] * 100
        st.success(f"Predicted Default Risk: {risk:.2f}%")
    else:
        st.error("API Error — Is FastAPI running?")
