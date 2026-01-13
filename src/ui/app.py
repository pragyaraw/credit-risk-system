import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")

BASE = Path(__file__).resolve().parent
bundle = joblib.load(BASE / "../../models/xgboost_bundle.pkl")
model = bundle["model"]
features = bundle["features"]

st.title("Credit Risk Predictor")

age = st.number_input("Age", min_value=18, max_value=100, value=28)
income = st.number_input("Income", min_value=0, value=50000)
loan_amt = st.number_input("Loan Amount", min_value=0, value=15000)
int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.5)
loan_pct = st.number_input("Loan % of Income", min_value=0.0, max_value=1.0, value=0.3)

payload = {
    "person_age": age,
    "person_income": income,
    "loan_amnt": loan_amt,
    "loan_int_rate": int_rate,
    "loan_percent_income": loan_pct,
}

def predict_local(payload):
    df = pd.DataFrame([payload])
    df = pd.get_dummies(df)

    for col in features:
        if col not in df.columns:
            df[col] = 0

    df = df[features]
    return model.predict_proba(df)[0][1]

if st.button("Predict Risk"):

    # 🔒 Input Validation
    if age < 18 or age > 100:
        st.error("Age must be between 18 and 100.")
        st.stop()

    if income <= 0:
        st.error("Income must be positive.")
        st.stop()

    if loan_amount <= 0:
        st.error("Loan amount must be positive.")
        st.stop()

    if interest_rate < 0 or interest_rate > 100:
        st.error("Interest rate must be between 0 and 100.")
        st.stop()

    if loan_pct <= 0 or loan_pct > 1:
        st.error("Loan % of income must be between 0 and 1.")
        st.stop()

    if loan_amount > income * 5:
        st.warning("Loan amount is very high relative to income.")

    # 📤 Build payload only if validation passed
    payload = {
        "person_age": age,
        "person_income": income,
        "loan_amnt": loan_amount,
        "loan_int_rate": interest_rate,
        "loan_percent_income": loan_pct
    }

    # 🔮 Predict
    with st.spinner("Predicting..."):
        prob = predict_local(payload)
        st.success(f"Predicted Default Risk: {prob*100:.2f}%")
