from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent

bundle = joblib.load(BASE / "../../models/xgboost_bundle.pkl")
model = bundle["model"]
features = bundle["features"]

app = FastAPI(title="Credit Risk API")

class Applicant(BaseModel):
    person_age: int
    person_income: float
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float

@app.post("/predict")
def predict_risk(applicant: Applicant):

    if applicant.person_age < 18 or applicant.person_age > 100:
        raise HTTPException(status_code=400, detail="Age must be between 18 and 100")

    if applicant.person_income <= 0:
        raise HTTPException(status_code=400, detail="Income must be positive")

    if applicant.loan_amnt <= 0:
        raise HTTPException(status_code=400, detail="Loan amount must be positive")

    if applicant.loan_int_rate < 0 or applicant.loan_int_rate > 100:
        raise HTTPException(status_code=400, detail="Interest rate must be between 0 and 100")

    if applicant.loan_percent_income <= 0 or applicant.loan_percent_income > 1:
        raise HTTPException(status_code=400, detail="Loan % of income must be between 0 and 1")

    # Align columns with training features
    for col in features:
        if col not in data.columns:
            data[col] = 0

    data = data[features]

    prob = model.predict_proba(data)[0][1]
    return {"default_risk": float(prob)}

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port)
