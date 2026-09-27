import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def load_data(data_path):
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please add the dataset.")
    df = pd.read_csv(data_path)
    return df

def basic_preprocessing(df):
    # Check for duplicates and drop them if any
    df = df.drop_duplicates()
    return df

def get_train_test_split(df):
    X = df.drop(columns=['Class'])
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test

def get_preprocessing_pipeline(use_smote=False):
    steps = [('scaler', StandardScaler())]
    if use_smote:
        steps.append(('smote', SMOTE(random_state=42)))
    return ImbPipeline(steps)
