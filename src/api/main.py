from fastapi import FastAPI
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
    data = pd.DataFrame([applicant.dict()])
    data = pd.get_dummies(data)

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
