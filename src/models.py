from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from src.preprocessing import get_preprocessing_pipeline
import joblib
from pathlib import Path

def get_logistic_regression():
    return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)

def get_random_forest():
    return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)

def get_xgboost(scale_pos_weight):
    return XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False, eval_metric="logloss")

def get_isolation_forest(contamination):
    return IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)

def build_model_pipeline(model, use_smote=False):
    pipeline = get_preprocessing_pipeline(use_smote=use_smote)
    pipeline.steps.append(('classifier', model))
    return pipeline

def save_model(model, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)

def load_model(filepath):
    return joblib.load(filepath)
