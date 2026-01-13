import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from pathlib import Path
import joblib

BASE = Path(__file__).resolve().parent

# Load data
df = pd.read_csv(BASE / "../../data/credit_risk_dataset.csv")

# Target
y = df["loan_status"]
X = df.drop(columns=["loan_status"])

# One-hot encode categoricals
X = pd.get_dummies(X, drop_first=True)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Handle imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)

print("Training model...")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

roc = roc_auc_score(y_val, y_prob)
f1 = f1_score(y_val, y_pred)

print(f"ROC-AUC: {roc:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save model + feature names
feature_names = X.columns.tolist()

bundle = {
    "model": model,
    "features": feature_names
}

bundle_path = BASE / "../../models/xgboost_bundle.pkl"
joblib.dump(bundle, bundle_path)

print(f"Model and features saved to {bundle_path}")
