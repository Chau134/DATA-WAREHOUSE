import os
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.decomposition import PCA

from preprocess import process_edf_file, process_single_file

# Defer Keras/TensorFlow import until needed (lazy import)
_KERAS_AVAILABLE = None
def is_keras_available() -> bool:
    """Return True if Keras/TensorFlow can be imported. Cache result."""
    global _KERAS_AVAILABLE
    if _KERAS_AVAILABLE is not None:
        return _KERAS_AVAILABLE
    try:
        # import lazily to avoid requiring TF at module import time
        import keras  # type: ignore
        _KERAS_AVAILABLE = True
    except Exception:
        _KERAS_AVAILABLE = False
    return _KERAS_AVAILABLE

try:
    from models.kmeans_predict import predict_cluster, predict_cluster_with_pca
    KMEANS_AVAILABLE = True
except ImportError:
    KMEANS_AVAILABLE = False
    predict_cluster = None
    predict_cluster_with_pca = None


LABEL_TO_NAME = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "R",
}

FEATURE_COLUMNS_19 = [
    "Mean",
    "Variance",
    "Skewness",
    "Kurtosis",
    "Power_Delta",
    "Power_Theta",
    "Power_Alpha",
    "Power_Beta",
    "Theta_Alpha_Ratio",
    "Spectral_Entropy",
    "Rolling_Var",
    "Delta_diff",
    "Theta_diff",
    "Alpha_diff",
    "Beta_diff",
    "Delta_Theta_ratio",
    "Alpha_Beta_ratio",
    "Rolling_Mean_Delta",
    "Rolling_Mean_Theta",
]


def list_model_files(models_dir: Path):
    """Return a list of classifier model files (.pkl and .h5) in models_dir, excluding scalers/encoders/kmeans artifacts.
    
    If Keras is not available, .h5 files are excluded.
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []
    files = []
    # include .h5 and .pkl but exclude common non-model artifacts
    for p in sorted(models_dir.iterdir()):
        if p.suffix.lower() == '.h5':
            # Include .h5 files only if Keras is available
            if is_keras_available():
                files.append(p)
        elif p.suffix.lower() == '.pkl':
            name = p.stem.lower()
            if any(x in name for x in ('scaler', 'encoder', 'kmeans', 'pca')):
                continue
            files.append(p)
    return files


def show_pdf_bytes(pdf_bytes: bytes, height: int = 600):
    """Render PDF bytes inside Streamlit using an iframe (base64 embedded)."""
    if not pdf_bytes:
        st.info("Không có PDF để hiển thị.")
        return
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="{height}" type="application/pdf"></iframe>'
    components.html(pdf_display, height=height)



def find_scaler_for_model(model_path: Path):
    models_dir = Path(model_path).parent
    # common naming: <modelname>_scaler.pkl or *_scaler.pkl
    candidate = models_dir / f"{Path(model_path).stem}_scaler.pkl"
    if candidate.exists():
        return candidate
    # fallback: any scaler file
    scalers = sorted(models_dir.glob("*scaler*.pkl"))
    return scalers[0] if scalers else None


def find_encoder_for_model(model_path: Path):
    models_dir = Path(model_path).parent
    candidate = models_dir / f"{Path(model_path).stem}_encoder.pkl"
    if candidate.exists():
        return candidate
    encs = sorted(models_dir.glob("*encoder*.pkl"))
    return encs[0] if encs else None


def preprocess_pair(edf_path: str, tsv_path: str):
    """Process EDF/TSV pair and return aligned feature matrix and labels."""
    X_test, y_true = process_single_file(edf_path, tsv_path)
    return np.asarray(X_test), np.asarray(y_true)

def generate_decision_support_report(y_pred: np.ndarray) -> str:
    """Create a simple decision support markdown report from predicted stages."""
    from collections import Counter

    if y_pred is None:
        return "Không có dự đoán để tạo báo cáo."

    total = len(y_pred)
    stage_dist = Counter(map(int, y_pred))
    report = []

    # phân bố giai đoạn (Dùng chữ in đậm thay vì Heading)
    report.append("**Phân bố giai đoạn ngủ:**")
    for stage_id in sorted(stage_dist.keys()):
        count = stage_dist[stage_id]
        pct = 100 * count / total
        stage_name = LABEL_TO_NAME.get(int(stage_id), f"Unknown({stage_id})")
        report.append(f"- {stage_name}: {count} epoch ({pct:.1f}%)")

    # giai đoạn dominant
    dominant_stage_id = max(stage_dist, key=stage_dist.get)
    dominant_stage_name = LABEL_TO_NAME.get(int(dominant_stage_id), str(dominant_stage_id))
    dominant_pct = 100 * stage_dist[dominant_stage_id] / total
    report.append(f"\n**Giai đoạn chiếm ưu thế:** {dominant_stage_name} ({dominant_pct:.1f}%)")

    # nhận xét
    report.append("\n**Nhận xét:**")
    if dominant_stage_id == 0:  # Wake
        report.append("- Người này tỉnh nhiều hoặc khó ngủ. Có thể cần tư vấn về vệ sinh giấc ngủ.")
    elif dominant_stage_id in [1, 2, 3]:  # N1, N2, N3
        report.append(f"- Giai đoạn {dominant_stage_name} chiếm ưu thế, ngủ sâu hợp lý.")
        if stage_dist.get(3, 0) > 0:  # N3
            report.append("- Có ngủ sâu (N3), phục hồi tốt.")
        else:
            report.append("- Thiếu ngủ sâu (N3), có thể cần cải thiện chất lượng giấc ngủ.")
    elif dominant_stage_id == 4:  # REM
        report.append("- Giai đoạn REM (mơ) chiếm ưu thế, có thể cần tư vấn về rối loạn giấc ngủ.")

    # kiến nghị
    report.append("\n**Kiến nghị:**")
    if stage_dist.get(3, 0) == 0:
        report.append("- Không phát hiện ngủ sâu (N3) - xem xét các biện pháp cải thiện giấc ngủ.")
    if dominant_stage_id == 0 and stage_dist.get(0, 0) > total * 0.5:
        report.append("- Tỉnh nhiều (>50%) - khuyến cáo tư vấn bác sĩ.")
    if stage_dist.get(4, 0) > 0:
        rem_pct = 100 * stage_dist[4] / total
        if rem_pct < 15:
            report.append("- REM dưới 15% - có thể liên quan đến stress hoặc rối loạn giấc ngủ.")

    return "\n".join(report)

def predict_with_model(model_path: Path, X: np.ndarray):
    is_keras_model = model_path.suffix.lower() == ".h5" or "ann" in model_path.stem.lower()

    if is_keras_model:
        if not is_keras_available():
            raise ImportError(
                f"Keras/TensorFlow not installed. Cannot load ANN model '{model_path.name}'. "
                f"Please use a .pkl model (Logistic, Random Forest, or XGBoost) instead. "
                f"Install with: pip install tensorflow"
            )
        # import only when needed
        from keras.models import load_model
        try:
            model = load_model(model_path, compile=False)
        except TypeError as e:
            # Known incompatibility: some Keras versions save a Layer config
            # containing 'quantization_config' which older/newer deserializers don't accept.
            msg = str(e)
            if 'quantization_config' in msg or 'Unrecognized keyword arguments passed to' in msg:
                try:
                    import h5py, json, re, shutil, tempfile

                    tmp_path = Path(tempfile.mktemp(suffix='.h5'))
                    shutil.copy2(model_path, tmp_path)

                    with h5py.File(tmp_path, 'r+') as fh:
                        # model_config may be stored as an attribute or a dataset
                        if 'model_config' in fh.attrs:
                            raw = fh.attrs['model_config']
                            if isinstance(raw, bytes):
                                raw = raw.decode('utf-8')
                            cleaned = re.sub(r'\s*"quantization_config"\s*:\s*null,?', '', raw)
                            cleaned = re.sub(r',\s*}', '}', cleaned)
                            fh.attrs['model_config'] = cleaned
                        elif 'model_config' in fh:
                            raw = fh['model_config'][()]
                            if isinstance(raw, bytes):
                                raw = raw.decode('utf-8')
                            cleaned = re.sub(r'\s*"quantization_config"\s*:\s*null,?', '', raw)
                            cleaned = re.sub(r',\s*}', '}', cleaned)
                            del fh['model_config']
                            fh.create_dataset('model_config', data=cleaned)

                    model = load_model(tmp_path, compile=False)
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass
                except Exception:
                    # fallthrough to re-raise original error
                    raise
            else:
                raise
    else:
        model = joblib.load(model_path)

    model_n_features = getattr(model, "n_features_in_", None)

    if model_n_features is not None and X.shape[1] != int(model_n_features):
        raise ValueError(
            f"Model `{model_path.name}` can {int(model_n_features)} features, "
            f"nhung preprocess hien tai tao {X.shape[1]} features."
        )

    scaler_path = find_scaler_for_model(model_path)
    encoder_path = find_encoder_for_model(model_path) if is_keras_model else None
    X_input = X
    X_df = None
    if X.shape[1] == len(FEATURE_COLUMNS_19):
        X_df = pd.DataFrame(X, columns=FEATURE_COLUMNS_19)

    if scaler_path is not None:
        scaler = joblib.load(scaler_path)
        scaler_n_features = getattr(scaler, "n_features_in_", None)
        if scaler_n_features is not None and X.shape[1] != int(scaler_n_features):
            raise ValueError(
                f"Scaler `{scaler_path.name}` can {int(scaler_n_features)} features, "
                f"nhung preprocess hien tai tao {X.shape[1]} features."
            )
        feature_names_in = getattr(scaler, "feature_names_in_", None)
        if feature_names_in is not None and X_df is not None:
            X_input = scaler.transform(X_df[list(feature_names_in)])
        else:
            X_input = scaler.transform(X)

    if is_keras_model:
        if encoder_path is None:
            raise ValueError(f"Không tìm thấy encoder cho model ANN `{model_path.name}`.")
        encoder = joblib.load(encoder_path)
        y_prob = model.predict(X_input, verbose=0)
        y_pred_idx = np.argmax(y_prob, axis=1)
        if hasattr(encoder, "classes_"):
            y_pred = np.asarray(encoder.classes_)[y_pred_idx].astype(int)
        else:
            y_pred = np.asarray(y_pred_idx).astype(int)
    else:
        y_pred = model.predict(X_input)
        y_pred = np.asarray(y_pred).astype(int)

    return y_pred, scaler_path


def predict_cluster_info(X: np.ndarray, models_dir: Path):
    """Predict cluster and get cluster info."""
    if not KMEANS_AVAILABLE:
        return None, None, None
    try:
        cluster_labels, pca_coords, cluster_stats = predict_cluster_with_pca(
            X, str(models_dir), n_components=2
        )
        return cluster_labels, pca_coords, cluster_stats
    except Exception as e:
        return None, None, None


def main():
    st.set_page_config(page_title="Sleep Stage Test Evaluator", layout="wide")
    
    if "is_initialized" not in st.session_state:
        init_placeholder = st.empty() # Tạo một khung trống
        with init_placeholder.container():
            with st.spinner("Đang khởi tạo hệ thống và tải cấu hình mô hình..."):
                import time
                time.sleep(1) 
        init_placeholder.empty() 
        st.session_state.is_initialized = True
    
    st.title("Sleep Stage Test Evaluator")
    st.caption("Upload EDF/TSV, chọn nhiều mô hình (Classification & Clustering) để chạy và so sánh.")

    # Simple styling for a cleaner look
    st.markdown(
        """
        <style>
        .main .block-container{max-width:1200px;padding:1rem 2rem}
        .stButton>button{border-radius:6px}
        </style>
        """,
        unsafe_allow_html=True,
    )

    default_models_dir = Path("models")

    with st.sidebar:
        st.header("Cấu hình")
        models_dir = Path(st.text_input("Thư mục chứa model", str(default_models_dir)))

        # Show warning if keras not available
        if not is_keras_available():
            st.warning("Keras/TensorFlow chưa cài — model ANN (.h5) sẽ không khả dụng.")
        
    if not models_dir.exists():
        st.error(f"Không tìm thấy thư mục model: {models_dir}")
        return

    model_files = list_model_files(models_dir)
    
    # Tạo danh sách các lựa chọn (Bao gồm cả file model và tên đại diện cho K-Means)
    options_list = list(model_files)
    if KMEANS_AVAILABLE:
        options_list.append("K-Means Clustering")

    if not options_list:
        st.warning("Không tìm thấy model nào hợp lệ trong thư mục models.")
        return

    col1, col2 = st.columns(2)
    with col1:
        uploaded_edf = st.file_uploader(
            "Upload file EDF", 
            type=["edf"],
            accept_multiple_files=False,
            help="Chỉ phân tích 1 file EDF mỗi lần. Nếu bạn tải lên file khác, ứng dụng sẽ tự động thay thế file cũ."
        )
    with col2:
        uploaded_tsv = st.file_uploader(
            "Upload file TSV (nhãn) (Optional)", 
            type=["tsv"],
            accept_multiple_files=False,
            help="Chỉ tải lên 1 file TSV tương ứng với EDF bên cạnh. Tải lên file mới sẽ ghi đè file cũ."
        )

    with st.sidebar:
        selected_models = st.multiselect(
            "Chọn các model cần so sánh",
            options=options_list,
            format_func=lambda p: p if isinstance(p, str) else p.name,
            default=[options_list[0]] if options_list else None
        )

    if uploaded_edf is None:
        st.info("Hãy upload file EDF để tiền xử lý và dự đoán.")
        return

    if st.button("Tiền xử lý và Dự đoán", type="primary"):
        if not selected_models:
            st.error("Vui lòng chọn ít nhất một model.")
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            edf_path = temp_dir_path / uploaded_edf.name
            edf_path.write_bytes(uploaded_edf.getbuffer())

            tsv_path = None
            if uploaded_tsv is not None:
                tsv_path = temp_dir_path / uploaded_tsv.name
                tsv_path.write_bytes(uploaded_tsv.getbuffer())

            with st.spinner("Đang tiền xử lý EDF..."):
                if tsv_path is not None:
                    X_test, y_true = preprocess_pair(str(edf_path), str(tsv_path))
                else:
                    X_test = process_edf_file(str(edf_path))
                    y_true = None

            if X_test.size == 0:
                if tsv_path is not None:
                    st.warning(
                        "TSV không tạo được sample có nhãn phù hợp; chuyển sang chế độ chỉ EDF."
                    )
                    X_test = process_edf_file(str(edf_path))
                    y_true = None
                if X_test.size == 0:
                    st.error("Không tạo được sample từ file EDF này.")
                    return

            st.markdown("---")
            
            # Biến lưu trữ kết quả của tất cả model
            model_results = {}
            
            with st.spinner("Đang chạy các model..."):
                for model_option in selected_models:
                    if isinstance(model_option, str) and model_option == "K-Means Clustering":
                        try:
                            cluster_labels, distances, cluster_stats = predict_cluster(X_test, str(models_dir))
                            scaler_path = Path(models_dir) / 'kmeans_scaler.pkl'
                            model_results["K-Means"] = {
                                "type": "clustering",
                                "cluster_labels": cluster_labels,
                                "cluster_stats": cluster_stats,
                                "scaler_path": scaler_path
                            }
                        except Exception as exc:
                            st.error(f"Lỗi khi chạy K-means: {exc}")
                    else:
                        # Classification model
                        try:
                            y_pred, scaler_path = predict_with_model(model_option, X_test)
                            model_results[model_option.name] = {
                                "type": "classification",
                                "y_pred": y_pred,
                                "scaler_path": scaler_path
                            }
                        except ValueError as exc:
                            st.error(f"Lỗi ở model {model_option.name}: {exc}")
                            continue
            
            if not model_results:
                return

            st.header("Kết quả Dự đoán & So sánh")
            st.metric("Tổng số epoch (test)", len(X_test))

            # 1. BẢNG SO SÁNH TỔNG QUAN
            if y_true is not None:
                has_classifiers = any(res["type"] == "classification" for res in model_results.values())
                if has_classifiers:
                    st.subheader("Bảng so sánh hiệu suất (Các mô hình Classification)")
                    metrics_data = []
                    for m_name, res in model_results.items():
                        if res["type"] == "classification":
                            acc = accuracy_score(y_true, res["y_pred"])
                            f1 = f1_score(y_true, res["y_pred"], average="macro")
                            metrics_data.append({
                                "Tên Model": m_name, 
                                "Accuracy": f"{acc:.4f}", 
                                "Macro F1": f"{f1:.4f}"
                            })
                    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

            # 2. CHI TIẾT TỪNG MODEL VÀ NHÃN THỰC TẾ
            st.subheader("Chi tiết theo từng mô hình")
            
            tab_names = []
            if y_true is not None:
                tab_names.append("Nhãn thực tế")
            tab_names.extend(list(model_results.keys()))
            
            tabs = st.tabs(tab_names)
            tab_idx = 0
            
            # ==========================================
            # TAB 0: HIỂN THỊ NHÃN THỰC TẾ
            # ==========================================
            if y_true is not None:
                with tabs[tab_idx]:
                    st.caption("Dữ liệu gốc được gán nhãn bởi chuyên gia y tế (Từ file TSV).")
                    st.markdown("**Kết luận hỗ trợ quyết định (Dựa trên thực tế)**")
                    st.markdown(generate_decision_support_report(y_true))
                    
                    true_unique, true_counts = np.unique(y_true, return_counts=True)
                    true_stage_names = [LABEL_TO_NAME.get(int(s), str(s)) for s in true_unique]

                    col_chart1, col_chart2 = st.columns(2)
                    with col_chart1:
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(3.5, 2.5))
                        ax.bar(true_stage_names, true_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(true_unique)])
                        ax.set_title("Phân bố thực tế", fontsize=10)
                        ax.tick_params(axis='both', labelsize=8)
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=False)
                    with col_chart2:
                        fig, ax = plt.subplots(figsize=(3.5, 2.5))
                        pcts = [100 * c / len(y_true) for c in true_counts]
                        ax.pie(pcts, labels=true_stage_names, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
                        ax.set_title("Tỷ lệ % (Thực tế)", fontsize=10)
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=False)
                tab_idx += 1

            # ==========================================
            # CÁC TABS TIẾP THEO: HIỂN THỊ TỪNG MODEL
            # ==========================================
            for m_name, res in model_results.items():
                with tabs[tab_idx]:
                    if res["type"] == "classification":
                        y_pred_m = res["y_pred"]
                        scaler_m = res["scaler_path"]
                        st.caption(f"Loại: Phân loại có giám sát | Scaler: {scaler_m.name if scaler_m else 'Không'}")
                        
                        st.markdown("**Kết luận hỗ trợ quyết định (Dựa trên dự đoán)**")
                        st.markdown(generate_decision_support_report(y_pred_m))

                        unique, counts = np.unique(y_pred_m, return_counts=True)
                        stage_names = [LABEL_TO_NAME.get(int(s), str(s)) for s in unique]

                        col_chart1, col_chart2 = st.columns(2)
                        with col_chart1:
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(3.5, 2.5))
                            ax.bar(stage_names, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(unique)])
                            ax.set_title("Phân bố dự đoán", fontsize=10)
                            ax.tick_params(axis='both', labelsize=8)
                            fig.tight_layout()
                            st.pyplot(fig, use_container_width=False)
                        with col_chart2:
                            fig, ax = plt.subplots(figsize=(3.5, 2.5))
                            pcts = [100 * c / len(y_pred_m) for c in counts]
                            ax.pie(pcts, labels=stage_names, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
                            ax.set_title("Tỷ lệ % (Dự đoán)", fontsize=10)
                            fig.tight_layout()
                            st.pyplot(fig, use_container_width=False)

                        if y_true is not None:
                            st.markdown("---")
                            st.markdown("**Đối chiếu với Nhãn thực tế**")
                            label_order = sorted(LABEL_TO_NAME.keys())
                            
                            c_tab1, c_tab2 = st.tabs(["Confusion Matrix", "Classification Report"])
                            with c_tab1:
                                cm = confusion_matrix(y_true, y_pred_m, labels=label_order)
                                cm_df = pd.DataFrame(
                                    cm,
                                    index=[f"True_{LABEL_TO_NAME[i]}" for i in label_order],
                                    columns=[f"Pred_{LABEL_TO_NAME[i]}" for i in label_order],
                                )
                                st.dataframe(cm_df, use_container_width=True)
                            with c_tab2:
                                report_dict = classification_report(y_true, y_pred_m, labels=label_order, 
                                                                   target_names=[LABEL_TO_NAME[i] for i in label_order], 
                                                                   output_dict=True, zero_division=0)
                                st.dataframe(pd.DataFrame(report_dict).transpose(), use_container_width=True)
                    
                    elif res["type"] == "clustering":
                        cluster_labels = res["cluster_labels"]
                        cluster_stats = res["cluster_stats"]
                        scaler_m = res["scaler_path"]
                        
                        st.caption(f"Loại: Gom cụm không giám sát (K-Means) | Scaler: {scaler_m.name if scaler_m else 'Không'}")
                        
                        if cluster_labels is None or cluster_stats is None:
                            st.warning("Không có dữ liệu cụm để hiển thị.")
                            continue

                        if y_true is not None:
                            st.markdown("**So sánh phân bố: Cụm dự đoán vs. Nhãn thực**")
                            col_p1, col_p2 = st.columns(2)
                            with col_p1:
                                import matplotlib.pyplot as plt
                                fig1, ax1 = plt.subplots(figsize=(3.5, 2.5))
                                cluster_names = [f"Cụm {c}" for c in cluster_stats['clusters']]
                                ax1.pie(cluster_stats['percentages'], labels=cluster_names, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
                                ax1.set_title("Phân bố theo Cụm", fontsize=10)
                                fig1.tight_layout()
                                st.pyplot(fig1, use_container_width=False)
                            with col_p2:
                                fig2, ax2 = plt.subplots(figsize=(3.5, 2.5))
                                t_unique, t_counts = np.unique(y_true, return_counts=True)
                                t_names = [LABEL_TO_NAME.get(int(s), str(s)) for s in t_unique]
                                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                                ax2.pie(t_counts, labels=t_names, autopct='%1.1f%%', startangle=90, colors=colors[:len(t_unique)], textprops={'fontsize': 8})
                                ax2.set_title("Phân bố Nhãn thực", fontsize=10)
                                fig2.tight_layout()
                                st.pyplot(fig2, use_container_width=False)
                        else:
                            st.markdown("**Phân bố theo cụm:**")
                            col_c1, col_c2 = st.columns(2)
                            with col_c1:
                                for c_id, count, pct in zip(cluster_stats['clusters'], cluster_stats['counts'], cluster_stats['percentages']):
                                    st.write(f"- Cụm {c_id}: {count} epoch ({pct:.1f}%)")
                            with col_c2:
                                import matplotlib.pyplot as plt
                                fig, ax = plt.subplots(figsize=(3.5, 2.5))
                                cluster_names = [f"Cụm {c}" for c in cluster_stats['clusters']]
                                ax.pie(cluster_stats['percentages'], labels=cluster_names, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
                                ax.set_title("Phân bố cụm", fontsize=10)
                                fig.tight_layout()
                                st.pyplot(fig, use_container_width=False)

                        # Trực quan hóa PCA
                        try:
                            scaler = joblib.load(scaler_m)
                            expected_features = int(getattr(scaler, 'n_features_in_', X_test.shape[1]))
                            Xp = X_test[:, :expected_features] if X_test.shape[1] > expected_features else X_test
                            Xs = scaler.transform(Xp)
                            from sklearn.decomposition import PCA
                            coords = PCA(n_components=2).fit_transform(Xs)
                            
                            st.markdown("**Trực quan hóa PCA**")
                            if y_true is not None:
                                col_pca1, col_pca2 = st.columns(2)
                                with col_pca1:
                                    fig_a, ax_a = plt.subplots(figsize=(4, 3))
                                    for c_id in np.unique(cluster_labels):
                                        m = cluster_labels == c_id
                                        ax_a.scatter(coords[m, 0], coords[m, 1], label=f"Cụm {c_id}", s=10, alpha=0.6)
                                    ax_a.set_title("Cụm dự đoán", fontsize=10)
                                    ax_a.legend(loc='best', fontsize=7)
                                    ax_a.tick_params(axis='both', labelsize=8)
                                    fig_a.tight_layout()
                                    st.pyplot(fig_a, use_container_width=False)
                                with col_pca2:
                                    fig_b, ax_b = plt.subplots(figsize=(4, 3))
                                    for s_id in np.unique(y_true):
                                        m = y_true == s_id
                                        s_name = LABEL_TO_NAME.get(int(s_id), str(s_id))
                                        ax_b.scatter(coords[m, 0], coords[m, 1], label=s_name, s=10, alpha=0.6)
                                    ax_b.set_title("Nhãn thực tế", fontsize=10)
                                    ax_b.legend(loc='best', fontsize=7)
                                    ax_b.tick_params(axis='both', labelsize=8)
                                    fig_b.tight_layout()
                                    st.pyplot(fig_b, use_container_width=False)
                            else:
                                fig, ax = plt.subplots(figsize=(5, 3.5))
                                for c_id in np.unique(cluster_labels):
                                    m = cluster_labels == c_id
                                    ax.scatter(coords[m, 0], coords[m, 1], label=f"Cụm {c_id}", s=15, alpha=0.7)
                                ax.set_title("PCA 2D - Clusters", fontsize=10)
                                ax.legend(loc='best', fontsize=8)
                                ax.tick_params(axis='both', labelsize=8)
                                fig.tight_layout()
                                st.pyplot(fig, use_container_width=False)
                        except Exception as e:
                            st.warning(f"Không thể hiển thị PCA: {e}")

                        if y_true is not None:
                            st.markdown("---")
                            st.markdown("**Bảng đối chiếu Nhãn vs Cụm**")
                            ct = pd.crosstab(pd.Series(y_true, name='Thực tế'), pd.Series(cluster_labels, name='Cụm dự đoán'))
                            st.dataframe(ct, use_container_width=True)

                tab_idx += 1

            # 3. TẠO DATAFRAME TỔNG HỢP CHO TẤT CẢ CÁC MODEL VÀ DOWNLOAD
            st.markdown("---")
            st.subheader("Dữ liệu xuất (Kết hợp tất cả mô hình)")
            base_dict = {
                "epoch_index": np.arange(len(X_test)),
            }
            if y_true is not None:
                base_dict["y_true"] = y_true
                base_dict["true_stage"] = [LABEL_TO_NAME.get(int(v), str(v)) for v in y_true]

            for m_name, res in model_results.items():
                if res["type"] == "classification":
                    base_dict[f"y_pred_{m_name}"] = res["y_pred"]
                    base_dict[f"stage_{m_name}"] = [LABEL_TO_NAME.get(int(v), str(v)) for v in res["y_pred"]]
                else:
                    base_dict["cluster_K-Means"] = res["cluster_labels"]
            
            pred_df = pd.DataFrame(base_dict)
            download_name = f"predictions_{edf_path.stem}_multimodel.csv"

            st.dataframe(pred_df, use_container_width=True)
            csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Tải về kết quả (CSV)",
                data=csv_bytes,
                file_name=download_name,
                mime="text/csv",
            )

if __name__ == "__main__":
    main()