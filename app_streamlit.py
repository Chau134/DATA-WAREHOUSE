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

# Try to import keras (needed only for ANN model)
try:
    from keras.models import load_model
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    def load_model(*args, **kwargs):
        raise ImportError("Keras/TensorFlow not installed. Cannot load .h5 models. Use .pkl models instead.")

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
            # Only include .h5 files if keras is available
            if KERAS_AVAILABLE:
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
    """Create a simple decision support markdown report from predicted stages.

    Returns a markdown string.
    """
    from collections import Counter

    if y_pred is None:
        return "Không có dự đoán để tạo báo cáo."

    total = len(y_pred)
    stage_dist = Counter(map(int, y_pred))
    report = []

    # phân bố giai đoạn
    report.append("## Phân bố giai đoạn ngủ:")
    for stage_id in sorted(stage_dist.keys()):
        count = stage_dist[stage_id]
        pct = 100 * count / total
        stage_name = LABEL_TO_NAME.get(int(stage_id), f"Unknown({stage_id})")
        report.append(f"- **{stage_name}**: {count} epoch ({pct:.1f}%)")

    # giai đoạn dominant
    dominant_stage_id = max(stage_dist, key=stage_dist.get)
    dominant_stage_name = LABEL_TO_NAME.get(int(dominant_stage_id), str(dominant_stage_id))
    dominant_pct = 100 * stage_dist[dominant_stage_id] / total
    report.append(f"\n## Giai đoạn chiếm ưu thế: **{dominant_stage_name}** ({dominant_pct:.1f}%)")

    # nhận xét
    report.append("\n## Nhận xét:")
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
    report.append("\n## Kiến nghị:")
    if stage_dist.get(3, 0) == 0:
        report.append("- ⚠️ Không phát hiện ngủ sâu (N3) - xem xét các biện pháp cải thiện giấc ngủ.")
    if dominant_stage_id == 0 and stage_dist.get(0, 0) > total * 0.5:
        report.append("- ⚠️ Tỉnh nhiều (>50%) - khuyến cáo tư vấn bác sĩ.")
    if stage_dist.get(4, 0) > 0:
        rem_pct = 100 * stage_dist[4] / total
        if rem_pct < 15:
            report.append("- ℹ️ REM dưới 15% - có thể liên quan đến stress hoặc rối loạn giấc ngủ.")

    return "\n".join(report)


def predict_with_model(model_path: Path, X: np.ndarray):
    is_keras_model = model_path.suffix.lower() == ".h5" or "ann" in model_path.stem.lower()

    if is_keras_model:
        if not KERAS_AVAILABLE:
            raise ImportError(
                f"Keras/TensorFlow not installed. Cannot load ANN model '{model_path.name}'. "
                f"Please use a .pkl model (Logistic, Random Forest, or XGBoost) instead. "
                f"Install with: pip install tensorflow"
            )
        model = load_model(model_path, compile=False)
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
            raise ValueError(f"Khong tim thay encoder cho ANN model `{model_path.name}`.")
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
    st.title("Sleep Stage Test Evaluator")
    st.caption("Upload EDF/TSV, chay preprocess, du doan bang model .pkl va doi chieu voi nhan bac si.")

    

    default_models_dir = Path("models")

    with st.sidebar:
        st.header("Cau hinh")
        models_dir = Path(st.text_input("Thu muc model", str(default_models_dir)))
        mode = st.selectbox("Chon chuc nang", ["Classification", "Clustering"])
        
        # Show warning if keras not available
        if not KERAS_AVAILABLE:
            st.warning("⚠️ Keras/TensorFlow not installed - ANN model unavailable. Use Logistic, Random Forest, or XGBoost instead.")
        
    if not models_dir.exists():
        st.error(f"Khong tim thay thu muc model: {models_dir}")
        return

    model_files = list_model_files(models_dir)
    if not model_files:
        st.warning("Khong tim thay model .pkl hop le trong thu muc models.")
        return

    col1, col2 = st.columns(2)
    with col1:
        uploaded_edf = st.file_uploader("Upload file EDF", type=["edf"])
    with col2:
        uploaded_tsv = st.file_uploader("Upload file TSV label", type=["tsv"])

    # Model selection only for classification mode
    selected_model = None
    cluster_labels = None
    cluster_stats = None
    
    if mode == "Classification":
        with st.sidebar:
            model_files_list = list(model_files)
            if model_files_list:
                selected_model = st.selectbox(
                    "Chon model .pkl",
                    options=model_files_list,
                    format_func=lambda p: p.name,
                )

    if uploaded_edf is None:
        st.info("Hay upload file EDF de tien xu ly va du doan.")
        return

    if st.button("Tien xu ly + Du doan", type="primary"):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            edf_path = temp_dir_path / uploaded_edf.name
            edf_path.write_bytes(uploaded_edf.getbuffer())

            tsv_path = None
            if uploaded_tsv is not None:
                tsv_path = temp_dir_path / uploaded_tsv.name
                tsv_path.write_bytes(uploaded_tsv.getbuffer())

            with st.spinner("Dang preprocess EDF..."):
                if tsv_path is not None:
                    X_test, y_true = preprocess_pair(str(edf_path), str(tsv_path))
                else:
                    X_test = process_edf_file(str(edf_path))
                    y_true = None

            if X_test.size == 0:
                if tsv_path is not None:
                    st.warning(
                        "TSV khong tao duoc sample co nhan phu hop, app se chuyen sang che do EDF-only de van du doan duoc."
                    )
                    X_test = process_edf_file(str(edf_path))
                    y_true = None
                if X_test.size == 0:
                    st.error("Khong tao duoc sample tu file EDF nay.")
                    return

            if mode == "Classification":
                if selected_model is None:
                    st.error("Vui lòng chọn một model để phân loại.")
                    return
                    
                with st.spinner("Dang chay model..."):
                    try:
                        y_pred, scaler_path = predict_with_model(selected_model, X_test)
                    except ValueError as exc:
                        st.error(str(exc))
                        st.info("Hay chon model khop preprocess hien tai hoac train lai model theo bo feature tu preprocess.py")
                        return
            else:
                # Clustering mode
                if not KMEANS_AVAILABLE:
                    st.error("K-means model khong co, bo qua chuc nang clustering.")
                    return
                    
                with st.spinner("Dang chay gom cum (K-means)..."):
                    try:
                        cluster_labels, distances, cluster_stats = predict_cluster(X_test, str(models_dir))
                        y_pred = None
                        scaler_path = Path(models_dir) / 'kmeans_scaler.pkl'
                    except Exception as exc:
                        st.error(f"Loi khi chay K-means: {exc}")
                        return

            m1, m2, m3 = st.columns(3)
            if mode == "Classification":
                m1.metric("So epoch test", len(y_pred))
                if y_true is not None:
                    accuracy = accuracy_score(y_true, y_pred)
                    macro_f1 = f1_score(y_true, y_pred, average="macro")
                    m2.metric("Accuracy", f"{accuracy:.4f}")
                    m3.metric("Macro F1", f"{macro_f1:.4f}")
                else:
                    m2.metric("Accuracy", "-")
                    m3.metric("Macro F1", "-")
            else:
                # clustering summary
                m1.metric("So epoch test", X_test.shape[0])
                if cluster_stats:
                    m2.metric("So cum phat hien", f"{len(cluster_stats['clusters'])}")
                    max_cluster_idx = np.argmax(cluster_stats['counts'])
                    m3.metric("Mau cum lon nhat", f"C{cluster_stats['clusters'][max_cluster_idx]}")
                else:
                    m2.metric("So cum phat hien", "N/A")
                    m3.metric("Mau cum lon nhat", "N/A")

            if mode == "Classification":
                st.write(f"Model: `{selected_model.name}`")
                st.write(f"Scaler: `{scaler_path.name}`" if scaler_path else "Scaler: khong dung")
            else:
                st.write("Model: KMeans (models/kmeans_model.pkl)")
                st.write(f"Scaler: `{scaler_path.name}`" if scaler_path else "Scaler: khong tim thay")

            if y_true is None:
                if mode == "Classification":
                    st.subheader("📋 Kết luận hỗ trợ quyết định")
                    if y_pred is not None:
                        decision_text = generate_decision_support_report(y_pred)
                        st.markdown(decision_text)

                        unique, counts = np.unique(y_pred, return_counts=True)
                        stage_names = [LABEL_TO_NAME.get(int(s), str(s)) for s in unique]

                        col_chart1, col_chart2 = st.columns(2)
                        with col_chart1:
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(8, 5))
                            ax.bar(stage_names, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(unique)])
                            ax.set_xlabel("Giai đoạn ngủ")
                            ax.set_ylabel("Số epoch")
                            ax.set_title("Phân bố giai đoạn ngủ dự đoán")
                            st.pyplot(fig)

                        with col_chart2:
                            fig, ax = plt.subplots(figsize=(8, 5))
                            pcts = [100 * c / len(y_pred) for c in counts]
                            ax.pie(pcts, labels=stage_names, autopct='%1.1f%%', startangle=90)
                            ax.set_title("Phần trăm giai đoạn ngủ")
                            st.pyplot(fig)
                else:
                    st.info("Chế độ Clustering: không có dự đoán stage sleep. Hiển thị thông tin cụm.")
                    if KMEANS_AVAILABLE and cluster_labels is not None and cluster_stats is not None:
                        st.subheader("🎯 Phân loại bệnh nhân (Clustering)")
                        st.write(f"**Cụm được phát hiện:** {', '.join(map(str, cluster_stats['clusters']))}")
                        col_c1, col_c2 = st.columns(2)
                        with col_c1:
                            st.write("**Phân bố theo cụm:**")
                            for cluster_id, count, pct in zip(cluster_stats['clusters'], cluster_stats['counts'], cluster_stats['percentages']):
                                st.write(f"- Cụm {cluster_id}: {count} epoch ({pct:.1f}%)")
                        with col_c2:
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(8, 5))
                            cluster_names = [f"Cụm {c}" for c in cluster_stats['clusters']]
                            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                            ax.pie(cluster_stats['percentages'], labels=cluster_names, autopct='%1.1f%%', colors=colors[:len(cluster_names)], startangle=90)
                            ax.set_title("Phân bố cụm (K-means)")
                            st.pyplot(fig)

                        # PCA visualization: compute PCA on scaled X_test
                        try:
                            scaler = joblib.load(Path(models_dir) / 'kmeans_scaler.pkl')
                            expected_features = int(getattr(scaler, 'n_features_in_', X_test.shape[1]))
                            Xp = X_test[:, :expected_features] if X_test.shape[1] > expected_features else X_test
                            Xs = scaler.transform(Xp)
                            pca = PCA(n_components=2)
                            coords = pca.fit_transform(Xs)
                            st.write("**Biểu diễn PCA 2D của các cụm:**")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            unique_clusters = np.unique(cluster_labels)
                            for i, cluster_id in enumerate(unique_clusters):
                                mask = cluster_labels == cluster_id
                                ax.scatter(coords[mask, 0], coords[mask, 1], label=f"Cụm {cluster_id}", s=40, alpha=0.7)
                            ax.set_xlabel("PCA 1")
                            ax.set_ylabel("PCA 2")
                            ax.set_title("PCA 2D - Clusters")
                            ax.legend()
                            st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"Khong the ve PCA: {e}")
            else:
                st.subheader("📊 So sánh dự đoán vs Nhãn thực")

                # Classification: compare y_true vs y_pred
                if mode == "Classification":
                    label_order = sorted(LABEL_TO_NAME.keys())
                    cm = confusion_matrix(y_true, y_pred, labels=label_order)
                    cm_df = pd.DataFrame(
                        cm,
                        index=[f"True_{LABEL_TO_NAME[i]}" for i in label_order],
                        columns=[f"Pred_{LABEL_TO_NAME[i]}" for i in label_order],
                    )
                    st.subheader("Confusion Matrix")
                    st.dataframe(cm_df, use_container_width=True)

                    report_dict = classification_report(
                        y_true,
                        y_pred,
                        labels=label_order,
                        target_names=[LABEL_TO_NAME[i] for i in label_order],
                        output_dict=True,
                        zero_division=0,
                    )
                    report_df = pd.DataFrame(report_dict).transpose()
                    st.subheader("Classification Report")
                    st.dataframe(report_df, use_container_width=True)
                else:
                    # Clustering: show contingency table between true stages and cluster ids
                    if cluster_labels is None:
                        st.error("Không có dự đoán cụm để so sánh với nhãn thực.")
                    else:
                        ct = pd.crosstab(pd.Series(y_true, name='TrueStage'), pd.Series(cluster_labels, name='Cluster'))
                        st.subheader("Contingency: True Stage vs Cluster")
                        st.dataframe(ct, use_container_width=True)

                        # also show percentages per true stage
                        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
                        st.subheader("Percentage per True Stage (by cluster)")
                        st.dataframe(ct_pct.round(2), use_container_width=True)

            if mode == "Classification":
                pred_df = pd.DataFrame(
                    {
                        "epoch_index": np.arange(len(y_pred)) if y_pred is not None else [],
                        "y_true": y_true if y_true is not None else [np.nan] * (len(y_pred) if y_pred is not None else 0),
                        "y_pred": y_pred if y_pred is not None else [],
                        "true_stage": [LABEL_TO_NAME.get(int(v), str(v)) for v in y_true] if y_true is not None else ["-"] * (len(y_pred) if y_pred is not None else 0),
                        "pred_stage": [LABEL_TO_NAME.get(int(v), str(v)) for v in y_pred] if y_pred is not None else [],
                    }
                )
                download_name = f"predictions_{edf_path.stem}_{selected_model.stem}.csv" if selected_model else "predictions.csv"
            else:
                pred_df = pd.DataFrame(
                    {
                        "epoch_index": np.arange(len(cluster_labels)) if cluster_labels is not None else [],
                        "cluster": cluster_labels if cluster_labels is not None else [],
                    }
                )
                download_name = f"clusters_{edf_path.stem}_kmeans.csv"

            st.subheader("Chi tiet du doan")
            st.dataframe(pred_df, use_container_width=True)

            csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Tai ve ket qua du doan (CSV)",
                data=csv_bytes,
                file_name=download_name,
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
