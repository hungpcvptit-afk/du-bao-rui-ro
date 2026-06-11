# 🛡️ Hệ thống Ứng dụng Web Phát hiện Gian lận Giao dịch (Streamlit App)

Ứng dụng web tương tác trực quan được phát triển trên nền tảng **Streamlit** giúp chuyển đổi mô hình huấn luyện từ mã nguồn Jupyter Notebook (`phat_hien_giao_dich_gian_lan.ipynb`) thành một sản phẩm công nghệ hoàn chỉnh cho phép các chuyên viên quản trị rủi ro chấm điểm tín dụng và phát hiện gian lận tự động theo thời gian thực.

## ✨ Tính năng chính của ứng dụng
- **Cấu hình động tham số AI:** Tùy biến linh hoạt bộ tham số cho thuật toán `RandomForestClassifier` (độ sâu, số lượng cây quyết định, trọng số mất cân bằng lớp) ngay từ giao diện.
- **Tổng quan dữ liệu khoa học:** Thống kê mô tả và theo dõi chi tiết cấu trúc dữ liệu đầu vào.
- **Trực quan hóa đồ họa cao cấp:** Sử dụng `Plotly` dựng ma trận tương quan và phân phối biến phân lớp trực quan.
- **Báo cáo kiểm định minh bạch:** Hiển thị chi tiết ma trận nhầm lẫn (Confusion Matrix), biểu đồ đo lường mức độ quan trọng của biến thuộc tính (`Feature Importance`) cùng bộ chỉ số khoa học: Accuracy, Precision, Recall, F1-Score.
- **Hai chế độ vận hành dự báo linh hoạt:**
  1. Nhập trực tiếp thông số của một giao dịch đơn lẻ thông qua biểu mẫu động.
  2. Tải tệp dữ liệu hàng loạt (`.csv` / `.xlsx`) cấu trúc kiểm thử để dự báo đồng thời và xuất báo cáo kết quả tức thì.

## 📁 Cấu trúc dữ liệu đầu vào yêu cầu
Để hệ thống hoạt động chính xác, tệp dữ liệu huấn luyện hoặc tệp tin dự báo hàng loạt cần tuân thủ theo định dạng của `dataset1 (1).csv`:
- **Biến đầu vào ($X$):** Gồm 14 cột đặc trưng định lượng liên tục được định danh lần lượt tự động từ `X_1`, `X_2`, `X_3`, ..., `X_14`.
- **Biến phân lớp mục tiêu ($y$):** Cột mang tên `default` chứa giá trị nhị phân nguyên (`0`: Giao dịch bình thường; `1`: Giao dịch gian lận/rủi ro cao). *Lưu ý: Đối với chế độ tải file dự báo hàng loạt ở Tab 4, cột `default` này không bắt buộc phải có sẵn.*

## ⚙️ Hướng dẫn cài đặt và khởi chạy hệ thống

### Bước 1: Khởi tạo môi trường và tải các thư viện phụ thuộc
Mở terminal tại thư mục chứa các file mã nguồn của dự án và chạy câu lệnh sau:
```bash
pip install -r requirements.txt
