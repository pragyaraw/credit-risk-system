import pandas as pd
from ydata_profiling import ProfileReport
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

print("Loading dataset...")
df = pd.read_csv(BASE_DIR / "../../data/credit_risk_dataset.csv")

print("Generating profiling report...")
profile = ProfileReport(df, title="Credit Risk Report", explorative=True)
profile.to_file(BASE_DIR / "../../reports/profiling_report.html")

print("EDA complete. Report saved to reports/profiling_report.html")
