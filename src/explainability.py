import shap
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd

def generate_shap_explanations(model, X_train, X_test, feature_names, save_dir):
    # SHAP explainer
    # Tree explainer for tree-based models, otherwise KernelExplainer or generic Explainer
    # We assume model is the classifier (not the full pipeline). The X_train and X_test should be preprocessed.
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    except Exception:
        # Fallback for non-tree models, might be slow, so we take a sample
        explainer = shap.Explainer(model, X_train.sample(100, random_state=42))
        shap_values = explainer(X_test.sample(100, random_state=42))
        
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Global summary plot
    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X_test, feature_names=feature_names, show=False)
    else:
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'shap_summary.png')
    plt.close()
    
def generate_local_explanation(explainer, shap_values, X_sample, feature_names, index, save_path):
    # Depending on model type and SHAP version
    # Not fully implemented without knowing exact SHAP shapes, but conceptually:
    
    plt.figure()
    
    if isinstance(shap_values, list): # TreeExplainer with classification
        shap.plots.waterfall(shap.Explanation(values=shap_values[1][index], 
                                              base_values=explainer.expected_value[1], 
                                              data=X_sample.iloc[index], 
                                              feature_names=feature_names),
                             show=False)
    elif hasattr(shap_values, 'values'):
        # For new SHAP api
        shap.plots.waterfall(shap_values[index], show=False)
    
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()
