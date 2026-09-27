import nbformat as nbf

nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell("# Đề tài: Phát hiện gian lận giao dịch bằng Machine Learning\n\n## 1. Problem Definition\nPhát hiện các giao dịch có nguy cơ gian lận trong bối cảnh thanh toán điện tử / e-commerce.\n- Class = 0: Giao dịch bình thường\n- Class = 1: Giao dịch gian lận\n\nMục tiêu là giảm số giao dịch Fraud bị bỏ sót và cân bằng giữa False Positive (cảnh báo nhầm) và False Negative (bỏ sót). Không đánh giá mô hình chỉ bằng Accuracy vì dữ liệu bị mất cân bằng nghiêm trọng."),
    
    nbf.v4.new_markdown_cell("## 2. Import Libraries"),
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Thêm đường dẫn để import custom modules
BASE_DIR = Path().resolve().parent
sys.path.append(str(BASE_DIR))

from src.preprocessing import get_train_test_split, basic_preprocessing, get_preprocessing_pipeline
from src.models import get_logistic_regression, get_random_forest, get_xgboost, get_isolation_forest, build_model_pipeline, save_model
from src.evaluation import evaluate_model, plot_precision_recall_curve, plot_roc_curve, plot_confusion_matrix
from src.cost_analysis import threshold_optimization, plot_threshold_cost_curve
from src.explainability import generate_shap_explanations, generate_local_explanation

from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')"""),

    nbf.v4.new_markdown_cell("## 3. Load Dataset\nSử dụng đường dẫn tương đối (relative path) bằng Pathlib."),
    nbf.v4.new_code_cell("""DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
if not DATA_PATH.exists():
    print(f"Dataset chưa được đặt tại {DATA_PATH}. Hãy thêm dataset vào thư mục này rồi chạy lại.")
else:
    df = pd.read_csv(DATA_PATH)
    print("Loaded dataset successfully!")"""),
    
    nbf.v4.new_markdown_cell("## 4. Data Overview"),
    nbf.v4.new_code_cell("""if 'df' in locals():
    display(df.head())
    print("Shape:", df.shape)
    df.info()
    display(df.describe())"""),
    
    nbf.v4.new_markdown_cell("## 5. Data Cleaning\nKiểm tra missing values và duplicate."),
    nbf.v4.new_code_cell("""if 'df' in locals():
    print("Missing values:\\n", df.isnull().sum().max())
    print("Duplicate rows:", df.duplicated().sum())
    df = basic_preprocessing(df)
    print("Shape after dropping duplicates:", df.shape)"""),
    
    nbf.v4.new_markdown_cell("## 6. Class Imbalance Analysis\nKiểm tra sự mất cân bằng giữa giao dịch bình thường và gian lận."),
    nbf.v4.new_code_cell("""if 'df' in locals():
    class_counts = df["Class"].value_counts()
    class_ratios = df["Class"].value_counts(normalize=True)
    print("Số giao dịch Normal:", class_counts[0])
    print("Số giao dịch Fraud:", class_counts[1])
    print("Tỷ lệ Normal: {:.4%}".format(class_ratios[0]))
    print("Tỷ lệ Fraud: {:.4%}".format(class_ratios[1]))
    print(f"Imbalance ratio: 1 Fraud / {int(class_counts[0]/class_counts[1])} Normal")"""),
    
    nbf.v4.new_markdown_cell("**Giải thích:** Dữ liệu bị mất cân bằng nghiêm trọng. Do tỷ lệ Fraud rất nhỏ, nếu model dự đoán toàn bộ là Normal thì vẫn có Accuracy > 99%. Do đó, ta không dùng Accuracy làm tiêu chí đánh giá chính."),
    
    nbf.v4.new_markdown_cell("## 7. Exploratory Data Analysis (EDA)\nTạo các biểu đồ phân phối và tương quan."),
    nbf.v4.new_code_cell("""if 'df' in locals():
    fig_dir = BASE_DIR / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Phân phối Class
    plt.figure()
    sns.countplot(x='Class', data=df)
    plt.title('Class Distribution')
    plt.savefig(fig_dir / 'class_distribution.png')
    plt.close()
    
    # 2. Phân phối Amount
    plt.figure()
    sns.histplot(df[df['Amount'] < 1000]['Amount'], bins=50)
    plt.title('Amount Distribution (Amount < 1000)')
    plt.savefig(fig_dir / 'amount_distribution.png')
    plt.close()
    
    # 3. Amount theo Class
    plt.figure()
    sns.boxplot(x='Class', y='Amount', data=df[df['Amount'] < 1000])
    plt.title('Amount by Class')
    plt.savefig(fig_dir / 'amount_by_class.png')
    plt.close()
    
    # 4. Time theo Class
    plt.figure(figsize=(12, 6))
    sns.histplot(df[df['Class'] == 0]['Time'], bins=50, color='b', stat='density', alpha=0.5, label='Normal')
    sns.histplot(df[df['Class'] == 1]['Time'], bins=50, color='r', stat='density', alpha=0.5, label='Fraud')
    plt.title('Time Distribution by Class')
    plt.legend()
    plt.savefig(fig_dir / 'time_distribution.png')
    plt.close()
    
    # 5. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), cmap='coolwarm', cbar=True)
    plt.title('Correlation Heatmap')
    plt.savefig(fig_dir / 'correlation_heatmap.png')
    plt.close()
    print("EDA figures saved to outputs/figures/")"""),
    
    nbf.v4.new_markdown_cell("## 8. Data Preprocessing & 9. Train/Test Split\nSử dụng stratify=y để giữ tỷ lệ Class trong tập train và test."),
    nbf.v4.new_code_cell("""if 'df' in locals():
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    print("Train shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)
    print("Train Fraud Rate: {:.4%}".format(y_train.mean()))
    print("Test Fraud Rate: {:.4%}".format(y_test.mean()))"""),
    
    nbf.v4.new_markdown_cell("## 10. Handling Class Imbalance\nKhông SMOTE trên toàn bộ dataset trước khi split vì sẽ gây Data Leakage. Test set phải giữ nguyên phân phối thực tế. SMOTE chỉ được áp dụng trên X_train hoặc bên trong imblearn Pipeline (Stratified CV)."),
    
    nbf.v4.new_markdown_cell("## 11. Logistic Regression, 12. Random Forest, 13. Isolation Forest, 14. XGBoost\nKhởi tạo các mô hình và huấn luyện."),
    nbf.v4.new_code_cell("""if 'df' in locals():
    models = {
        'Logistic Regression': build_model_pipeline(get_logistic_regression(), use_smote=False),
        'Random Forest': build_model_pipeline(get_random_forest(), use_smote=False),
        'XGBoost': build_model_pipeline(get_xgboost(scale_pos_weight=(y_train==0).sum()/(y_train==1).sum()), use_smote=False)
    }
    
    # Huấn luyện và dự đoán
    results = []
    trained_models = {}
    
    for name, pipeline in models.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        metrics = evaluate_model(y_test, y_pred, y_prob)
        metrics['Model'] = name
        results.append(metrics)
        
        # Plot curves & CM
        plot_precision_recall_curve(y_test, y_prob, name, BASE_DIR / 'outputs' / 'figures' / f'precision_recall_curve_{name.replace(" ", "_")}.png')
        plot_roc_curve(y_test, y_prob, name, BASE_DIR / 'outputs' / 'figures' / f'roc_curve_{name.replace(" ", "_")}.png')
        plot_confusion_matrix(y_test, y_pred, name, BASE_DIR / 'outputs' / 'figures' / f'confusion_matrix_{name.replace(" ", "_")}.png')
    
    # Isolation Forest
    print("Training Isolation Forest...")
    contamination = (y_train == 1).sum() / len(y_train) # giả định bằng tỷ lệ fraud trong tập train
    iso_forest = build_model_pipeline(get_isolation_forest(contamination))
    iso_forest.fit(X_train)
    # y_pred_iso is -1 (anomaly/fraud) and 1 (normal)
    y_pred_iso = iso_forest.predict(X_test)
    y_pred_iso = np.where(y_pred_iso == -1, 1, 0)
    
    metrics_iso = evaluate_model(y_test, y_pred_iso) # Isolation forest doesn't output prob by default easily
    metrics_iso['Model'] = 'Isolation Forest'
    results.append(metrics_iso)
    plot_confusion_matrix(y_test, y_pred_iso, 'Isolation Forest', BASE_DIR / 'outputs' / 'figures' / 'confusion_matrix_Isolation_Forest.png')"""),
    
    nbf.v4.new_markdown_cell("## 15. Model Comparison\nSo sánh các model qua Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC."),
    nbf.v4.new_code_cell("""if 'df' in locals():
    df_results = pd.DataFrame(results)
    res_dir = BASE_DIR / 'outputs' / 'results'
    res_dir.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(res_dir / 'model_comparison.csv', index=False)
    display(df_results)
    
    # Chọn model tốt nhất (Ví dụ dựa vào PR-AUC hoặc F1)
    # Giả sử XGBoost là model tốt nhất sau khi chạy
    best_model_name = 'XGBoost'
    best_model = trained_models[best_model_name]
    save_model(best_model, BASE_DIR / 'outputs' / 'models' / 'best_model.pkl')
    print(f"Saved {best_model_name} as best model.")"""),
    
    nbf.v4.new_markdown_cell("## 16. Precision-Recall Curve & 17. ROC Curve & 18. Confusion Matrix\n(Đã được lưu tự động trong thư mục `outputs/figures/`). Trong bài toán mất cân bằng, PR-AUC quan trọng hơn ROC-AUC vì PR-AUC tập trung vào Positive class (Fraud)."),
    
    nbf.v4.new_markdown_cell("## 19. Cost Matrix & 20. Threshold Optimization\nGiả định chi phí False Negative cao hơn False Positive. Ta thử nhiều threshold để tối ưu Total Cost."),
    nbf.v4.new_code_cell("""if 'df' in locals():
    y_prob_best = best_model.predict_proba(X_test)[:, 1]
    df_thresholds = threshold_optimization(y_test, y_prob_best, save_path=res_dir / 'threshold_analysis.csv')
    best_th, min_cost = plot_threshold_cost_curve(df_thresholds, BASE_DIR / 'outputs' / 'figures' / 'threshold_cost_curve.png')
    display(df_thresholds.loc[df_thresholds['Total Cost'].idxmin():df_thresholds['Total Cost'].idxmin()])"""),
    
    nbf.v4.new_markdown_cell("## 21. Feature Importance\nLấy feature importance từ mô hình cây."),
    nbf.v4.new_code_cell("""if 'df' in locals():
    xgb = best_model.named_steps['classifier']
    importances = xgb.feature_importances_
    features = X_train.columns
    
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importances)[::-1][:20]
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.title('Feature Importances (XGBoost)')
    plt.gca().invert_yaxis()
    plt.savefig(BASE_DIR / 'outputs' / 'figures' / 'xgboost_feature_importance.png')
    plt.close()
    print("Feature importance saved.")"""),
    
    nbf.v4.new_markdown_cell("## 22. SHAP / LIME Explainability\nSử dụng SHAP để giải thích mô hình."),
    nbf.v4.new_code_cell("""if 'df' in locals():
    import shap
    shap.initjs()
    X_train_scaled = pd.DataFrame(best_model.named_steps['scaler'].transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(best_model.named_steps['scaler'].transform(X_test), columns=X_test.columns)
    classifier = best_model.named_steps['classifier']
    
    # Tạo Summary Plot
    generate_shap_explanations(classifier, X_train_scaled, X_test_scaled, X_train.columns, BASE_DIR / 'outputs' / 'figures')
    
    # Tạo Local Explanations
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_test_scaled)
    
    # Fraud examples
    fraud_indices = np.where(y_test.values == 1)[0][:2]
    # Normal example
    normal_indices = np.where(y_test.values == 0)[0][:1]
    indices_to_explain = np.concatenate([fraud_indices, normal_indices])
    
    for i, idx in enumerate(indices_to_explain):
        print(f"Generating explanation for transaction idx: {idx}, Actual Class: {y_test.values[idx]}")
        generate_local_explanation(explainer, shap_values, X_test_scaled, X_train.columns, idx, BASE_DIR / 'outputs' / 'figures' / f'shap_local_{i+1}.png')
    print("SHAP plots saved.")"""),
    
    nbf.v4.new_markdown_cell("## 23. Learning Curve & 24. Overfitting Analysis"),
    nbf.v4.new_code_cell("""if 'df' in locals():
    # Learning Curve for Logistic Regression
    train_sizes, train_scores, test_scores = learning_curve(
        models['Logistic Regression'], X_train, y_train, cv=5, n_jobs=-1, scoring='f1', train_sizes=np.linspace(0.1, 1.0, 5)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    plt.figure()
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score (F1)")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score (F1)")
    plt.title("Learning Curve (Logistic Regression)")
    plt.xlabel("Training examples")
    plt.ylabel("F1 Score")
    plt.legend(loc="best")
    plt.savefig(BASE_DIR / 'outputs' / 'figures' / 'learning_curve_logistic.png')
    plt.close()
    print("Learning curve saved.")"""),
    
    nbf.v4.new_markdown_cell("**Overfitting Analysis:** So sánh Training score và Validation score để xem liệu có hiện tượng overfitting (Train cao, Val thấp) hay underfitting (Cả hai đều thấp) hay không."),
    
    nbf.v4.new_markdown_cell("## 25. E-commerce Business Application\nKết quả mô hình hỗ trợ Payment Gateway từ chối hoặc yêu cầu xác minh đối với giao dịch rủi ro cao. False Positive làm ảnh hưởng trải nghiệm khách hàng, False Negative gây thất thoát trực tiếp."),
    nbf.v4.new_markdown_cell("## 26. Limitations\n- Dataset bị ẩn danh hóa (V1-V28).\n- Cost Matrix chỉ là giả định.\n- Pattern gian lận thay đổi theo thời gian, mô hình cần retrain liên tục."),
    nbf.v4.new_markdown_cell("## 27. Conclusion\nĐã xây dựng pipeline từ dữ liệu đến deploy ứng dụng phân tích Fraud bằng SHAP. Cần tối ưu threshold phụ thuộc cost doanh nghiệp.")
]

with open(r'c:\Users\PC\BD_BT3\fraud_detection\notebooks\fraud_detection.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook created successfully!")
