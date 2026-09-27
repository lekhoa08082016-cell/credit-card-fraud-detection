import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import sys

# Add src to path so we can import modules if needed
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))
from src.prediction import load_prediction_model, predict_transaction

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

MODEL_PATH = BASE_DIR / "outputs" / "models" / "best_model.pkl"
RESULTS_DIR = BASE_DIR / "outputs" / "results"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"

st.title("Credit Card Fraud Detection")
st.markdown("This application detects fraudulent credit card transactions using Machine Learning.")

tab1, tab2, tab3 = st.tabs(["Fraud Prediction", "Model Performance", "Explainability"])

with tab1:
    st.header("Fraud Prediction")
    uploaded_file = st.file_uploader("Upload CSV of transactions", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())
        
        if st.button("Run Prediction"):
            try:
                model = load_prediction_model(MODEL_PATH)
                predictions = []
                
                # Assume threshold 0.5 for now, or read from optimal threshold config
                optimal_threshold = 0.5 
                
                for _, row in df.iterrows():
                    # We might need to drop 'Class' if it exists in the uploaded CSV
                    if 'Class' in row.index:
                        trans = pd.DataFrame([row.drop('Class')])
                    else:
                        trans = pd.DataFrame([row])
                        
                    pred_res = predict_transaction(trans, model, threshold=optimal_threshold)
                    predictions.append(pred_res)
                    
                df_res = pd.DataFrame(predictions)
                df_final = pd.concat([df.reset_index(drop=True), df_res], axis=1)
                
                st.write("Prediction Results:")
                st.dataframe(df_final)
                
                total_trans = len(df_final)
                pred_fraud = df_final['prediction'].sum()
                pred_normal = total_trans - pred_fraud
                fraud_rate = (pred_fraud / total_trans) * 100 if total_trans > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Transactions", total_trans)
                col2.metric("Predicted Normal", pred_normal)
                col3.metric("Predicted Fraud", pred_fraud)
                col4.metric("Fraud Rate", f"{fraud_rate:.2f}%")
                
                csv = df_final.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv',
                )
            except Exception as e:
                st.error(f"Error running prediction: {str(e)}. Make sure the model is trained and saved.")

with tab2:
    st.header("Model Performance")
    
    metrics_path = RESULTS_DIR / "model_comparison.csv"
    if metrics_path.exists():
        df_metrics = pd.read_csv(metrics_path)
        st.write("Evaluation Metrics:")
        st.dataframe(df_metrics)
    else:
        st.warning("Metrics file not found. Please run the notebook first.")
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Precision-Recall Curve")
        pr_curve_path = FIGURES_DIR / "precision_recall_curve.png"
        if pr_curve_path.exists():
            st.image(Image.open(pr_curve_path))
        else:
            st.write("Image not found.")
            
    with col2:
        st.subheader("Confusion Matrix")
        # Could be named based on best model, e.g., confusion_matrix_XGBoost.png
        # We will try to show one of them
        cm_files = list(FIGURES_DIR.glob("confusion_matrix_*.png"))
        if cm_files:
            st.image(Image.open(cm_files[0]))
        else:
            st.write("Image not found.")
            
    st.subheader("Threshold Cost Curve")
    tc_curve_path = FIGURES_DIR / "threshold_cost_curve.png"
    if tc_curve_path.exists():
        st.image(Image.open(tc_curve_path))
    else:
        st.write("Image not found.")

with tab3:
    st.header("Explainability")
    
    st.subheader("Feature Importance")
    fi_files = list(FIGURES_DIR.glob("*_feature_importance.png"))
    if fi_files:
        st.image(Image.open(fi_files[0]))
    else:
        st.write("Image not found.")
        
    st.subheader("SHAP Summary")
    shap_sum_path = FIGURES_DIR / "shap_summary.png"
    if shap_sum_path.exists():
        st.image(Image.open(shap_sum_path))
    else:
        st.write("Image not found.")
        
    st.subheader("Local Explanations")
    col1, col2, col3 = st.columns(3)
    
    local1 = FIGURES_DIR / "shap_local_1.png"
    local2 = FIGURES_DIR / "shap_local_2.png"
    local3 = FIGURES_DIR / "shap_local_3.png"
    
    if local1.exists():
        col1.image(Image.open(local1), caption="Transaction 1")
    if local2.exists():
        col2.image(Image.open(local2), caption="Transaction 2")
    if local3.exists():
        col3.image(Image.open(local3), caption="Transaction 3")
