import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "outputs" / "models" / "best_model.pkl"

def load_prediction_model(model_path=MODEL_PATH):
    return joblib.load(model_path)

def predict_transaction(transaction, model, threshold=0.5):
    # transaction is expected to be a DataFrame row or dict
    if isinstance(transaction, dict):
        df_trans = pd.DataFrame([transaction])
    else:
        df_trans = transaction
        
    prob = model.predict_proba(df_trans)[:, 1][0]
    
    prediction = 1 if prob >= threshold else 0
    
    if prob < 0.2:
        risk_level = "Low"
    elif prob < threshold:
        risk_level = "Medium"
    else:
        risk_level = "High"
        
    return {
        "prediction": prediction,
        "fraud_probability": prob,
        "risk_level": risk_level
    }
