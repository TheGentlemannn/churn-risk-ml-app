import joblib
import pandas as pd

from config import MODEL_PATH


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python src/train.py` first.")
    return joblib.load(MODEL_PATH)


def predict_churn_probability(customer: dict) -> float:
    model = load_model()
    input_df = pd.DataFrame([customer])
    return float(model.predict_proba(input_df)[:, 1][0])
