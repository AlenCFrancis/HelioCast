"""
Streamlit Web Application for Solar Flare Prediction.
Provides interactive timestamp-based inference using pre-trained ML/DL models.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.data.loader import load_config, load_solar_data
from src.data.preprocessing import SolarDataPreprocessor
from src.data.sequence_builder import create_sequences
from src.models.logistic_regression import SolarLogisticRegression
from src.models.random_forest import SolarRandomForest
from src.models.xgboost_model import SolarXGBoost
from src.models.lstm import SolarLSTMWrapper

# Page Configuration
st.set_page_config(
    page_title="☀️ Solar Flare Prediction System",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #FF6B35;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #A0AAB0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E222A;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #2E3440;
    }
    .badge-flare {
        background-color: #FF3366;
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.3rem;
        text-align: center;
        display: inline-block;
    }
    .badge-no-flare {
        background-color: #00CC88;
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.3rem;
        text-align: center;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_application_artifacts():
    """Load configuration, dataset, preprocessor, and saved models."""
    config = load_config()
    data_path = config["data"]["processed_path"]
    
    if not os.path.exists(data_path):
        st.error(f"Processed dataset file not found at: {data_path}. Please run training first.")
        st.stop()

    df = load_solar_data(data_path=data_path, config=config)
    
    # Load preprocessor
    scaler_path = "models/scaler.joblib"
    if not os.path.exists(scaler_path):
        st.error(f"Scaler artifact not found at {scaler_path}. Please train models first.")
        st.stop()
        
    preprocessor = SolarDataPreprocessor.load(scaler_path=scaler_path, config=config)

    # Load models dictionary
    models = {}
    
    if os.path.exists("models/logistic_regression.joblib"):
        models["Logistic Regression"] = SolarLogisticRegression.load("models/logistic_regression.joblib")

    if os.path.exists("models/random_forest.joblib"):
        models["Random Forest"] = SolarRandomForest.load("models/random_forest.joblib")

    if os.path.exists("models/xgboost.joblib"):
        models["XGBoost"] = SolarXGBoost.load("models/xgboost.joblib")

    if os.path.exists("models/lstm.pt"):
        models["LSTM"] = SolarLSTMWrapper.load("models/lstm.pt")

    # Load metrics reports if available
    metrics_summary = {}
    metrics_dir = "results/metrics"
    if os.path.exists(metrics_dir):
        for f in os.listdir(metrics_dir):
            if f.endswith(".json"):
                model_key = f.replace(".json", "")
                with open(os.path.join(metrics_dir, f), "r") as json_file:
                    metrics_summary[model_key] = json.load(json_file)

    return config, df, preprocessor, models, metrics_summary

def main():
    st.markdown('<div class="main-title">☀️ Solar Flare Prediction System (M+ 24h Horizon)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Predicting whether an M-class or higher solar flare will occur within the next 24 hours using historical SDO/HMI magnetogram observations.</div>', unsafe_allow_html=True)

    config, df, preprocessor, models, metrics_summary = load_application_artifacts()

    if len(models) == 0:
        st.warning("⚠️ No trained models found in `models/` directory. Please run model training first.")
        st.stop()

    timestamp_col = config["data"].get("timestamp_col", "timestamp")
    target_col = config["data"].get("target_col", "target_m_plus_24h")
    feature_names = preprocessor.feature_names
    seq_length = config["forecast"].get("history_hours", 24)

    # Sidebar Controls
    st.sidebar.header("⚙️ Model & Timestamp Selection")
    
    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        options=list(models.keys()),
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Observation Timestamp")
    
    use_latest = st.sidebar.checkbox("Use Latest Available Observation", value=True)
    
    available_timestamps = df[timestamp_col].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    
    # We need at least `seq_length` rows before selected timestamp
    valid_start_idx = seq_length - 1
    valid_timestamps = available_timestamps[valid_start_idx:]

    if use_latest:
        selected_timestamp_str = valid_timestamps[-1]
    else:
        selected_timestamp_str = st.sidebar.selectbox(
            "Select Historical Observation Time",
            options=valid_timestamps,
            index=len(valid_timestamps) - 1
        )

    predict_btn = st.sidebar.button("🚀 Predict Solar Flare", use_container_width=True)

    # Dataset Summary Metrics Header
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Observations", f"{len(df):,}")
    with col2:
        st.metric("Feature Parameters", f"{len(feature_names)}")
    with col3:
        min_date = df[timestamp_col].min().strftime('%Y-%m-%d')
        max_date = df[timestamp_col].max().strftime('%Y-%m-%d')
        st.metric("Date Range", f"{min_date} to {max_date}")
    with col4:
        st.metric("Observation Cadence", "Hourly (1.0h)")

    st.markdown("---")

    # Target Index Calculation
    selected_idx = df[df[timestamp_col].dt.strftime("%Y-%m-%d %H:%M:%S") == selected_timestamp_str].index[0]
    
    # Get 24-hour window slice up to selected_idx
    window_df = df.iloc[selected_idx - seq_length + 1 : selected_idx + 1].copy()
    
    # Perform Inference
    selected_model = models[selected_model_name]
    
    # Transform window using fitted scaler
    X_window_scaled = preprocessor.transform_df(window_df)  # shape: (24, num_features)

    # Get model threshold from metrics report if saved, else default 0.5
    model_key_map = {
        "Logistic Regression": "logistic_regression",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
        "LSTM": "lstm"
    }
    model_key = model_key_map.get(selected_model_name, "logistic_regression")
    saved_threshold = metrics_summary.get(model_key, {}).get("threshold", 0.5)

    if selected_model_name == "LSTM":
        X_seq_input = np.expand_dims(X_window_scaled, axis=0)  # shape: (1, 24, num_features)
        probability = float(selected_model.predict_proba(X_seq_input)[0])
    else:
        # Classical models take latest row or flattened sequence feature vector
        latest_row_scaled = X_window_scaled[-1:, :]  # shape: (1, num_features)
        probability = float(selected_model.predict_proba(latest_row_scaled)[0])

    is_flare = probability >= saved_threshold

    # Prediction Output Header & Gauge
    left_col, right_col = st.columns([1, 1.2])

    with left_col:
        st.subheader("🎯 Flare Prediction Results")
        st.write(f"**Target Observation Time**: `{selected_timestamp_str}`")
        st.write(f"**Selected Model**: `{selected_model_name}`")
        st.write(f"**Decision Threshold**: `{saved_threshold:.2f}`")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"### M+ Probability: **{probability * 100:.1f}%**")

        if is_flare:
            st.markdown('<div class="badge-flare">🔥 M+ FLARE EXPECTED WITHIN 24H</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge-no-flare">✅ NO M+ FLARE EXPECTED</div>', unsafe_allow_html=True)

        if target_col in window_df.columns:
            actual_label = window_df[target_col].iloc[-1]
            actual_str = "🔥 Actual M+ Flare" if actual_label == 1 else "✅ Actual No Flare"
            st.write(f"**Ground Truth Label**: `{actual_str}`")

    with right_col:
        st.subheader("📊 Probability Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={'suffix': "%"},
            title={'text': "M+ Flare Probability (Next 24 Hours)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#FF3366" if is_flare else "#00CC88"},
                'steps': [
                    {'range': [0, saved_threshold * 100], 'color': "#1E222A"},
                    {'range': [saved_threshold * 100, 100], 'color': "#2A1820"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': saved_threshold * 100
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")

    # Interactive Solar Feature History Plot
    st.subheader("📈 Previous 24-Hour Solar Active Region Magnetogram History")
    
    selected_feature = st.selectbox(
        "Select Solar Feature Parameter to Visualize History",
        options=feature_names,
        index=0
    )

    fig_time = px.line(
        window_df,
        x=timestamp_col,
        y=selected_feature,
        markers=True,
        title=f"24-Hour Lookback Trend: `{selected_feature}` up to `{selected_timestamp_str}`"
    )
    fig_time.update_traces(line_color="#FF6B35", marker=dict(size=6))
    fig_time.update_layout(
        xaxis_title="Observation Time",
        yaxis_title=selected_feature,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350
    )
    st.plotly_chart(fig_time, use_container_width=True)

    # Model Performance Comparison Section
    if len(metrics_summary) > 0:
        st.markdown("---")
        st.subheader("🏆 Offline Benchmark Performance Comparison")
        
        metrics_rows = []
        for name, metrics in metrics_summary.items():
            metrics_rows.append({
                "Model": name.upper(),
                "PR-AUC": metrics.get("pr_auc", 0.0),
                "ROC-AUC": metrics.get("roc_auc", 0.0),
                "F1 Score": metrics.get("f1_score", 0.0),
                "Precision": metrics.get("precision", 0.0),
                "Recall": metrics.get("recall", 0.0),
                "TSS": metrics.get("tss", 0.0),
                "FAR": metrics.get("far", 0.0),
                "Threshold": metrics.get("threshold", 0.5)
            })
        
        metrics_df = pd.DataFrame(metrics_rows)
        st.dataframe(metrics_df.style.highlight_max(axis=0, subset=["PR-AUC", "ROC-AUC", "F1 Score", "TSS"], color="#2E4A3E"), use_container_width=True)

if __name__ == "__main__":
    main()
