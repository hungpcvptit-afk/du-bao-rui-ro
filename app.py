import streamlit as io_st  # Tránh trùng tên gốc khi cấu hình rộng
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import io

# -----------------------------------------------------------------------------
# LỆNH STREAMLIT ĐẦU TIÊN
# -----------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Hệ thống Phát hiện Gian lận Giao dịch",
    page_icon="🛡️"
)

# -----------------------------------------------------------------------------
# CÁC HÀM CACHE DÙNG CHUNG
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(file_bytes, file_name):
    """Nạp dữ liệu từ bytes để đảm bảo khả năng hashable của Streamlit cache."""
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return None
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}")
        return None

# -----------------------------------------------------------------------------
# SIDEBAR - VÙNG CẤU HÌNH
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # 1. Tải file dữ liệu huấn luyện
    uploaded_file = st.file_uploader(
        "Tải lên dữ liệu huấn luyện (.csv, .xlsx)", 
        type=["csv", "xlsx"],
        help="Chọn tệp dữ liệu mẫu chứa các biến X_1 đến X_14 và biến mục tiêu 'default'"
    )
    
    st.divider()
    st.subheader("Tham số mô hình AI")
    st.caption("Thuật toán: RandomForestClassifier")
    
    # Các siêu tham số trích xuất và tối ưu từ cấu trúc hệ thống
    n_estimators = st.slider(
        "Số lượng cây (n_estimators)", 
        min_value=10, max_value=300, value=100, step=10,
        help="Số lượng cây quyết định trong rừng độc lập."
    )
    
    max_depth = st.slider(
        "Độ sâu tối đa (max_depth)", 
        min_value=2, max_value=30, value=15, step=1,
        help="Độ sâu tối đa của từng cây quyết định để tránh overfitting."
    )
    
    min_samples_split = st.number_input(
        "Mẫu tối thiểu để tách nút", 
        min_value=2, max_value=20, value=2, step=1,
        help="Số lượng mẫu tối thiểu cần thiết để phân tách một nút nội bộ."
    )
    
    random_state = st.number_input(
        "Trạng thái ngẫu nhiên (random_state)", 
        value=42, step=1,
        help="Đảm bảo tính tái lập của kết quả huấn luyện mô hình."
    )
    
    with st.expander("⚙️ Tham số nâng cao"):
        criterion = st.selectbox(
            "Tiêu chí đánh giá (criterion)", 
            options=["gini", "entropy", "log_loss"], index=0,
            help="Hàm đo lường chất lượng phân tách phân lớp."
        )
        class_weight = st.selectbox(
            "Trọng số phân lớp (class_weight)", 
            options=[None, "balanced", "balanced_subsample"], index=0,
            help="Chế độ xử lý dữ liệu mất cân bằng giữa giao dịch thường và gian lận."
        )

    st.divider()
    # Nút bấm hành động duy nhất để kích hoạt huấn luyện
    btn_train = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# HEADER - VÙNG ĐỊNH HƯỚNG
# -----------------------------------------------------------------------------
st.title("🛡️ Hệ thống Phát hiện Gian lận Giao dịch Tài chính")
st.caption("Ứng dụng hỗ trợ phân tích rủi ro tín dụng và phát hiện tự động các giao dịch bất thường dựa trên mô hình Học máy Random Forest.")

if uploaded_file is None:
    st.info("👋 Vui lòng tải lên tệp dữ liệu (.csv hoặc .xlsx) ở thanh Sidebar bên trái để bắt đầu cấu hình hệ thống.")
    st.stop()

# Đọc dữ liệu khi đã upload thành công
file_bytes = uploaded_file.getvalue()
df_raw = load_data(file_bytes, uploaded_file.name)

if df_raw is None:
    st.error("Không thể đọc tệp dữ liệu. Vui lòng kiểm tra lại định dạng file.")
    st.stop()

st.caption(f"📊 Đang sử dụng tệp: `{uploaded_file.name}` | Quy mô: {df_raw.shape[0]} dòng, {df_raw.shape[1]} cột")
st.divider()

# Khai báo các cột đặc trưng theo thiết kế của bài toán
features = [f"X_{i}" for i in range(1, 15)]
target = "default"

# Kiểm tra tính toàn vẹn của Schema dữ liệu đầu vào
missing_cols = [col for col in features + [target] if col not in df_raw.columns]
if missing_cols:
    st.error(f"Dữ liệu thiếu các cột cấu trúc bắt buộc sau: {missing_cols}")
    st.stop()

# -----------------------------------------------------------------------------
# KHỐI HUẤN LUYỆN (Chạy khi bấm nút, lưu vào session_state)
# -----------------------------------------------------------------------------
if btn_train:
    with st.spinner("⏳ Hệ thống đang tiến hành tiền xử lý và huấn luyện mô hình..."):
        try:
            X = df_raw[features]
            y = df_raw[target]
            
            # Phân tách tập huấn luyện và kiểm thử
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)
            
            # Chuẩn hóa dữ liệu đặc trưng
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Khởi tạo và huấn luyện mô hình
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state,
                criterion=criterion,
                class_weight=class_weight,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            
            # Dự đoán thu thập chỉ số
            y_pred = model.predict(X_test_scaled)
            y_probs = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Lưu trữ toàn bộ trạng thái vào session_state
            st.session_state["trained_model"] = model
            st.session_state["scaler"] = scaler
            st.session_state["eval_metrics"] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "y_test": y_test.tolist(),
                "y_pred": y_pred.tolist(),
                "feature_importances": model.feature_importances_.tolist()
            }
            st.success("✅ Huấn luyện mô hình thành công! Hãy chuyển sang các Tab bên dưới để xem kết quả chi tiết.")
        except Exception as e:
            st.error(f"Đã xảy ra lỗi trong quá trình huấn luyện: {e}")

# -----------------------------------------------------------------------------
# KHỞI TẠO CÁC TABS CHỨC NĂNG VÙNG CHÍNH
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🎯 Kết quả & Kiểm định", 
    "🔮 Dự báo thực tế"
])

# -----------------------------------------------------------------------------
# TAB 1: TỔNG QUAN DỮ LIỆU
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Phân tích cấu trúc file dữ liệu mẫu")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Số lượng bản ghi (Dòng)", f"{df_raw.shape[0]:,}")
    with col_m2:
        st.metric("Số lượng thuộc tính (Cột)", f"{df_raw.shape[1]}")
    with col_m3:
        file_size_mb = len(file_bytes) / (1024 * 1024)
        st.metric("Dung lượng tệp tin", f"{file_size_mb:.2f} MB")
        
    st.write("### 🔍 Xem trước 5 hàng dữ liệu đầu tiên")
    st.dataframe(df_raw.head(5), use_container_width=True)
    
    st.write("### 📉 Chỉ số thống kê mô tả của các biến mô hình")
    # Chỉ hiển thị thống kê mô tả đối với các biến đầu vào và biến mục tiêu
    st.dataframe(df_raw[features + [target]].describe().T, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: TRỰC QUAN HÓA DỮ LIỆU
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Biểu đồ phân phối thuộc tính đặc trưng")
    
    # Ưu tiên hiển thị biến mục tiêu trước
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        df_target_counts = df_raw[target].value_counts().reset_index()
        df_target_counts.columns = ["Trạng thái", "Số lượng"]
        df_target_counts["Trạng thái"] = df_target_counts["Trạng thái"].map({0: "0 (Bình thường)", 1: "1 (Gian lận/Rủi ro)"})
        fig_target = px.bar(
            df_target_counts, x="Trạng thái", y="Số lượng",
            title=f"Phân phối biến mục tiêu ({target})",
            color="Trạng thái", color_discrete_sequence=px.colors.qualitative.Set2,
            height=350
        )
        st.plotly_chart(fig_target, use_container_width=True)
        
    with col_g2:
        # Lựa chọn hiển thị động các biến phân tích nếu danh sách dài
        selected_feature = st.selectbox("Chọn biến đầu vào để xem phân phối mẫu:", options=features, index=0)
        fig_feat = px.histogram(
            df_raw, x=selected_feature, color=target,
            title=f"Biểu đồ phân phối biến {selected_feature} theo Nhãn mục tiêu",
            barmode="overlay", marginal="box",
            color_discrete_sequence=["#2b5c8f", "#d95f02"],
            height=350
        )
        st.plotly_chart(fig_feat, use_container_width=True)
        
    st.divider()
    st.write("### 🗺️ Ma trận tương quan các biến số")
    corr_matrix = df_raw[features + [target]].corr()
    fig_corr = px.imshow(
        corr_matrix, text_auto=".2f",
        title="Ma trận tương quan Pearson",
        color_continuous_scale="RdBu_r", aspect="auto", height=550
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Đánh giá độ chính xác thuật toán")
    
    if "eval_metrics" not in st.session_state:
        st.info("📢 Chưa tìm thấy dữ liệu mô hình. Vui lòng thiết lập cấu hình và bấm nút **'Huấn luyện mô hình'** ở thanh Sidebar trái.")
    else:
        metrics = st.session_state["eval_metrics"]
        
        # Hiển thị các chỉ số vô hướng quan trọng bằng các thẻ Metric
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        c_m1.metric("Độ chính xác (Accuracy)", f"{metrics['accuracy']:.4f}")
        c_m2.metric("Độ chuẩn xác (Precision)", f"{metrics['precision']:.4f}")
        c_m3.metric("Độ nhạy (Recall)", f"{metrics['recall']:.4f}")
        c_m4.metric("Chỉ số F1-Score", f"{metrics['f1']:.4f}")
        
        st.divider()
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.write("#### 🧱 Ma trận nhầm lẫn (Confusion Matrix)")
            cm = confusion_matrix(metrics["y_test"], metrics["y_pred"])
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Nhãn dự đoán", y="Nhãn thực tế"),
                x=["0 (Bình thường)", "1 (Gian lận)"],
                y=["0 (Bình thường)", "1 (Gian lận)"],
                color_continuous_scale="Blues", height=350
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_res2:
            st.write("#### 📊 Độ quan trọng của các biến thuộc tính (Feature Importance)")
            df_imp = pd.DataFrame({
                "Thuộc tính": features,
                "Độ quan trọng": metrics["feature_importances"]
            }).sort_values(by="Độ quan trọng", ascending=True)
            
            fig_imp = px.bar(
                df_imp, x="Độ quan trọng", y="Thuộc tính", orientation="h",
                title="Mức độ đóng góp quyết định của các biến số",
                color="Độ quan trọng", color_continuous_scale="Viridis", height=350
            )
            st.plotly_chart(fig_imp, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: SỬ DỤNG MÔ HÌNH (DỰ BÁO THỰC TẾ)
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("Chạy dự toán rủi ro giao dịch mới")
    
    if "trained_model" not in st.session_state:
        st.info("📢 Tính năng yêu cầu mô hình AI đã hoàn thiện fit dữ liệu. Vui lòng nhấn nút kích hoạt ở Sidebar trước.")
    else:
        model = st.session_state["trained_model"]
        scaler = st.session_state["scaler"]
        
        mode = st.radio("Lựa chọn phương thức nhập dữ liệu:", options=["Nhập trực tiếp qua form mẫu", "Dự báo hàng loạt theo file kiểm thử"])
        
        if mode == "Nhập trực tiếp qua form mẫu":
            st.write("✍️ *Điền các thông số kỹ thuật của giao dịch cụ thể dưới đây (Giá trị mặc định được lấy từ Trung vị dữ liệu):*")
            
            # Tạo form nhập dữ liệu động theo danh sách biến
            with st.form("single_prediction_form"):
                cols_form = st.columns(3)
                input_data = {}
                
                for idx, feat in enumerate(features):
                    # Phân bổ tuần tự các input field vào 3 cột thiết kế form cân đối
                    col_target = cols_form[idx % 3]
                    default_val = float(df_raw[feat].median())
                    min_val = float(df_raw[feat].min())
                    max_val = float(df_raw[feat].max())
                    
                    input_data[feat] = col_target.number_input(
                        f"Giá trị {feat}",
                        min_value=min_val - abs(min_val)*0.5,
                        max_value=max_val + abs(max_val)*0.5,
                        value=default_val,
                        format="%.5f"
                    )
                
                submit_predict = st.form_submit_button("🔍 Phân tích rủi ro", type="primary")
                
            if submit_predict:
                # Chuyển đổi dữ liệu input sang cấu trúc DataFrame chuẩn hóa giống lúc train
                df_single = pd.DataFrame([input_data])
                df_single_scaled = scaler.transform(df_single)
                
                prediction = model.predict(df_single_scaled)[0]
                proba = model.predict_proba(df_single_scaled)[0][1]
                
                st.subheader("🎯 Kết quả phân tích đánh giá:")
                if prediction == 1:
                    st.error(f"🚨 CẢNH BÁO: Giao dịch được hệ thống phân loại thuộc nhóm **RỦI RO GIAN LẬN** (Nhãn: 1).")
                else:
                    st.success(f"✅ AN TOÀN: Giao dịch được xác minh thuộc nhóm **BÌNH THƯỜNG** (Nhãn: 0).")
                    
                st.metric(label="Xác suất rủi ro dự đoán (Probability)", value=f"{proba*100:.2f} %")
                
        elif mode == "Dự báo hàng loạt theo file kiểm thử":
            st.write("📂 Tải lên file chứa cấu trúc các cột đặc trưng từ `X_1` tới `X_14` để tiến hành chấm điểm tự động toàn tập.")
            
            test_file = st.file_uploader("Tải tệp tin cần dự báo (.csv, .xlsx)", type=["csv", "xlsx"], key="bulk_predict")
            
            if test_file is not None:
                test_bytes = test_file.getvalue()
                df_test = load_data(test_bytes, test_file.name)
                
                if df_test is not None:
                    # Kiểm tra sự tồn tại đầy đủ của các biến đầu vào
                    missing_test_cols = [c for c in features if c not in df_test.columns]
                    if missing_test_cols:
                        st.error(f"Tệp tải lên không hợp lệ, thiếu các cột đặc trưng sau: {missing_test_cols}")
                    else:
                        # Thực hiện lọc đúng danh sách và thứ tự cột đặc trưng ban đầu
                        X_new = df_test[features]
                        X_new_scaled = scaler.transform(X_new)
                        
                        # Chạy dự báo hàng loạt
                        bulk_preds = model.predict(X_new_scaled)
                        bulk_probs = model.predict_proba(X_new_scaled)[:, 1]
                        
                        df_result = df_test.copy()
                        df_result["Dự đoán (default)"] = bulk_preds
                        df_result["Xác suất gian lận"] = bulk_probs
                        
                        st.success(f"🎉 Chấm điểm hoàn tất cho {df_result.shape[0]} dòng dữ liệu giao dịch mới.")
                        
                        # Thống kê nhanh kết quả vừa dự báo
                        count_fraud = int(np.sum(bulk_preds == 1))
                        st.warning(f"Hệ thống phát hiện **{count_fraud}** trường hợp nghi ngờ gian lận (Chiếm {(count_fraud/len(bulk_preds))*100:.2f}%).")
                        
                        st.write("#### 📋 Bảng kết quả tổng hợp chi tiết")
                        st.dataframe(df_result, use_container_width=True)
                        
                        # Cho phép người dùng kết xuất kết quả đầu ra thành file Excel/CSV
                        csv_buffer = df_result.to_csv(index=False, encoding="utf-8-sig")
                        st.download_button(
                            label="📥 Tải xuống tệp kết quả dự báo (CSV)",
                            data=csv_buffer,
                            file_name="ket_qua_du_bao_gian_lan.csv",
                            mime="text/csv"
                        )
      
