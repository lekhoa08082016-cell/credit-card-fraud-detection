# Credit Card Fraud Detection

## 1. Tên project
Phát hiện gian lận giao dịch bằng Machine Learning (Fraud Detection).

## 2. Mục tiêu
Phát hiện các giao dịch thẻ tín dụng có nguy cơ gian lận nhằm giảm tổn thất cho doanh nghiệp, cân bằng giữa cảnh báo sai (False Positive) và bỏ sót (False Negative). Không sử dụng Accuracy làm tiêu chí chính.

## 3. Dataset
Credit Card Fraud Detection.
Dataset được lưu tại `data/creditcard.csv` (Cần tự thêm vào thư mục `data/`).

## 4. Công nghệ sử dụng
- Python, Pandas, Numpy, Scikit-learn, Imbalanced-learn, XGBoost.
- Jupyter Notebook để phân tích.
- Streamlit để demo web app.
- SHAP để Explainability.

## 5. Cấu trúc project
```text
fraud_detection/
├── data/ (đặt creditcard.csv vào đây)
├── notebooks/ (chứa Jupyter Notebook)
├── src/ (các mã nguồn xử lý, models)
├── outputs/ (mô hình, kết quả, biểu đồ)
├── report/ (báo cáo)
├── app.py (Streamlit Web App)
└── requirements.txt
```

## 6. Cách cài đặt
```bash
pip install -r requirements.txt
```

## 7. Cách chạy Notebook
Mở Jupyter Notebook hoặc VS Code, chạy file `notebooks/fraud_detection.ipynb` từ đầu đến cuối.

## 8. Cách chạy Streamlit
```bash
python -m streamlit run app.py
```

## 9. Kết quả
Các model được so sánh bao gồm Logistic Regression, Random Forest, Isolation Forest và XGBoost. PR-AUC và Cost Matrix được dùng để chọn threshold tối ưu. (Xem thêm trong phần outputs/results/).

## 10. Ứng dụng e-commerce
Hệ thống này hỗ trợ Payment Gateway từ chối hoặc yêu cầu xác minh (OTP, KYC) đối với các giao dịch có Fraud Probability cao, giúp bảo vệ nền tảng e-commerce khỏi Chargeback.

## 11. Hạn chế
- Dataset ẩn danh (V1-V28) nên khó giải thích theo domain cụ thể.
- Môi trường thực tế có data drift nên model cần retrain liên tục.
- Cost Matrix trong demo chỉ là giả định học thuật.

## 12. Hướng dẫn deploy
Tham khảo `DEPLOY_CHECKLIST.md` để triển khai lên Streamlit Community Cloud.
