import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path

# Assume FN cost > FP cost
COST_FN = 10
COST_FP = 1

def calculate_cost(y_true, y_pred, cost_fn=COST_FN, cost_fp=COST_FP):
    cm = confusion_matrix(y_true, y_pred)
    # cm: [[TN, FP], [FN, TP]]
    if cm.shape == (2, 2):
        fn = cm[1, 0]
        fp = cm[0, 1]
    else:
        # Handle cases where only one class is present in y_true
        fn = 0
        fp = 0
        
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    return total_cost, fn, fp

def threshold_optimization(y_true, y_prob, thresholds=None, save_path=None):
    if thresholds is None:
        thresholds = np.arange(0.10, 0.95, 0.05)
        
    results = []
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        total_cost, fn, fp = calculate_cost(y_true, y_pred)
        
        results.append({
            'Threshold': t,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'FP': fp,
            'FN': fn,
            'Total Cost': total_cost
        })
        
    df_results = pd.DataFrame(results)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(save_path, index=False)
        
    return df_results

def plot_threshold_cost_curve(df_results, save_path=None):
    plt.figure(figsize=(10, 6))
    
    plt.plot(df_results['Threshold'], df_results['Total Cost'], marker='o', linewidth=2, color='red')
    
    best_idx = df_results['Total Cost'].idxmin()
    best_threshold = df_results.loc[best_idx, 'Threshold']
    min_cost = df_results.loc[best_idx, 'Total Cost']
    
    plt.axvline(x=best_threshold, color='black', linestyle='--', label=f'Optimal Threshold: {best_threshold:.2f}')
    
    plt.xlabel('Threshold')
    plt.ylabel('Total Business Cost')
    plt.title('Threshold Optimization vs Total Cost')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()
    
    return best_threshold, min_cost
