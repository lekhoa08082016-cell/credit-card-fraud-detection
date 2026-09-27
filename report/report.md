# BÁO CÁO: PHÁT HIỆN GIAN LẬN GIAO DỊCH (FRAUD DETECTION)

## 1. Giới thiệu bài toán
Trong môi trường thanh toán điện tử, gian lận thẻ tín dụng gây tổn thất khổng lồ. Bài toán yêu cầu xây dựng mô hình Machine Learning phát hiện giao dịch gian lận (Class=1) giữa hàng trăm ngàn giao dịch bình thường (Class=0).

## 2. Dataset
- Credit Card Fraud Detection (Kaggle).
- Chứa các biến V1 đến V28 đã qua PCA ẩn danh, Time và Amount.
- Mất cân bằng vô cùng nghiêm trọng.

## 3. Tiền xử lý
- Loại bỏ các dòng duplicate.
- Chia tập Train/Test bằng `stratify=y` trước khi xử lý mất cân bằng, đảm bảo Test set phản ánh đúng thực tế.
- StandardScaler áp dụng trên Pipeline.

## 4. Class imbalance
- Dữ liệu bị lệch (Imbalance Ratio cực lớn).
- Áp dụng SMOTE (chỉ trên tập Train) hoặc `class_weight='balanced'` để giúp mô hình không bị "mù" với class Fraud.
- Data Leakage được phòng ngừa bằng cách đưa SMOTE vào trong pipeline của `imblearn`.

## 5. Các thuật toán
- Logistic Regression
- Random Forest
- Isolation Forest (Anomaly detection)
- XGBoost (sử dụng scale_pos_weight)

## 6. Kết quả
Dự kiến XGBoost hoặc Random Forest sẽ đem lại hiệu suất tốt nhất. Accuracy không được dùng để so sánh mà phải dựa trên F1-score và PR-AUC.

## 7. Precision-Recall
Trong khi ROC-AUC dễ bị "ảo tưởng" bởi TN cao, PR-AUC thực tế hơn khi tập trung vào lớp thiểu số (Fraud). PR curve cho thấy sự đánh đổi giữa Precision và Recall.

## 8. Cost Matrix
Cost_FN (Bỏ sót Fraud) được định nghĩa lớn gấp 10 lần Cost_FP (Khóa nhầm khách hàng). Điều này phản ánh thực tế doanh nghiệp thà hy sinh một chút trải nghiệm khách (yêu cầu thêm OTP) còn hơn mất tiền thật.

## 9. Threshold Optimization
Không dùng mức cutoff 0.5. Mức cutoff tối ưu (ví dụ 0.8 hay 0.9) được xác định bằng cách vẽ Threshold vs Total Cost, chọn điểm có Total Cost thấp nhất.

## 10. SHAP/LIME
SHAP được dùng để tính toán mức độ đóng góp (Feature Importance) theo giá trị SHAP ở mức global, và vẽ local explanation (Waterfall plot) giải thích vì sao giao dịch đó bị đánh dấu là Fraud, dựa trên các feature như V14, V17...

## 11. Overfitting
Sử dụng Learning Curve để quan sát khoảng cách giữa Training Score và CV Score. Nếu Training rất cao và CV thấp, có hiện tượng Overfitting. Các biện pháp Regularization đã được cài đặt qua tham số mô hình.

## 12. Ứng dụng thực tế trong e-commerce
- Payment Gateway có thể dùng API này làm Risk Engine.
- Điểm Fraud Probability < threshold: duyệt ngay. Điểm ở mức Medium: chạy rule 3D-Secure. Điểm cao: Reject thẳng.
- Bảo vệ doanh nghiệp khỏi Chargeback.

## 13. Hạn chế
- Thiếu thông tin domain thực để tạo ra hand-crafted feature tốt.
- Cost giả định, không áp dụng cho mọi doanh nghiệp.
- Môi trường thực tế có Data Drift liên tục.

## 14. Kết luận
Dự án đã đáp ứng đầy đủ Pipeline chuẩn cho Fraud Detection, từ đánh giá, xử lý imbalance đến Cost Matrix và Explainability. Ứng dụng sẵn sàng demo bằng Streamlit.
