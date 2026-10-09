"""
Streamlit Web Application for Solar Flare Prediction.
Provides interactive timestamp-based inference using pre-trained ML/DL models.
"""

import os
import sys
from pathlib import Path

# Add project root directory to sys.path so 'src' can always be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from src.models.cnn_1d import SolarCNN1DWrapper

# Page Configuration
st.set_page_config(
    page_title="Solar Flare Prediction | Intelligence Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Sci-Fi Aesthetic)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@400;700&display=swap');

    /* Global Typography & Colors */
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif !important;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .stApp {
        background-color: #050505 !important;
        color: #F8FAFC !important;
    }
    .stSidebar {
        background-color: #0A0A0A !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Header Styles */
    .system-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.1;
    }
    .system-title span {
        color: #FF6B35;
    }
    .system-subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.5rem;
        margin-bottom: 2rem;
    }

    /* Nested "Double-Bezel" Card Architecture for Streamlit */
    .sci-card-outer {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        padding: 0.35rem;
        border-radius: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .sci-card-inner {
        background: #0B0E14;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.08);
        border-radius: calc(1.5rem - 0.35rem);
        padding: 1.5rem;
        height: 100%;
    }

    /* Typography inside Metric Cards */
    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
        display: block;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #E2E8F0;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    .metric-sub {
        font-size: 0.875rem;
        color: #94A3B8;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Badges */
    .badge-alert, .badge-safe {
        display: inline-flex;
        align-items: center;
        padding: 0.75rem 1.5rem;
        border-radius: 2rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.02em;
        margin-top: 1rem;
    }
    .badge-alert {
        background: rgba(255, 51, 102, 0.1);
        color: #FF3366;
        border: 1px solid rgba(255, 51, 102, 0.2);
    }
    .badge-safe {
        background: rgba(0, 204, 136, 0.1);
        color: #00CC88;
        border: 1px solid rgba(0, 204, 136, 0.2);
    }

    /* Overrides for Streamlit Elements */
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.5rem !important;
    }
    div[data-testid="stMetricLabel"] {
        font-family: 'Space Grotesk', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8 !important;
    }
    hr {
        border-color: rgba(255, 255, 255, 0.05) !important;
        margin: 2rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_application_artifacts():
    """Load configuration, dataset, preprocessor, and saved models."""
    config_path = str(PROJECT_ROOT / "configs" / "config.yaml")
    config = load_config(config_path)
    
    raw_data_path = config["data"]["processed_path"]
    data_path = raw_data_path if os.path.isabs(raw_data_path) else str(PROJECT_ROOT / raw_data_path)
    
    if not os.path.exists(data_path):
        st.error(f"Processed dataset file not found at: {data_path}. Please run training first.")
        st.stop()

    df = load_solar_data(data_path=data_path, config=config)
    
    # Load preprocessor
    scaler_path = str(PROJECT_ROOT / "models" / "scaler.joblib")
    if not os.path.exists(scaler_path):
        st.error(f"Scaler artifact not found at {scaler_path}. Please train models first (e.g. python -m src.training.classical --model all).")
        st.stop()
        
    preprocessor = SolarDataPreprocessor.load(scaler_path=scaler_path, config=config)

    # Load models dictionary
    models = {}
    
    lr_path = str(PROJECT_ROOT / "models" / "logistic_regression.joblib")
    if os.path.exists(lr_path):
        models["Logistic Regression"] = SolarLogisticRegression.load(lr_path)

    rf_path = str(PROJECT_ROOT / "models" / "random_forest.joblib")
    if os.path.exists(rf_path):
        models["Random Forest"] = SolarRandomForest.load(rf_path)

    xgb_path = str(PROJECT_ROOT / "models" / "xgboost.joblib")
    if os.path.exists(xgb_path):
        models["XGBoost"] = SolarXGBoost.load(xgb_path)

    lstm_path = str(PROJECT_ROOT / "models" / "lstm.pt")
    if os.path.exists(lstm_path):
        models["LSTM"] = SolarLSTMWrapper.load(lstm_path)

    cnn_path = str(PROJECT_ROOT / "models" / "cnn_1d.pt")
    if os.path.exists(cnn_path):
        models["1D CNN"] = SolarCNN1DWrapper.load(cnn_path)

    # Load metrics reports if available
    metrics_summary = {}
    metrics_dir = str(PROJECT_ROOT / "results" / "metrics")
    if os.path.exists(metrics_dir):
        for f in os.listdir(metrics_dir):
            if f.endswith(".json"):
                model_key = f.replace(".json", "")
                with open(os.path.join(metrics_dir, f), "r") as json_file:
                    metrics_summary[model_key] = json.load(json_file)

    return config, df, preprocessor, models, metrics_summary

def render_html_card(label, value, sub=""):
    return f"""
    <div class="sci-card-outer">
        <div class="sci-card-inner">
            <span class="metric-label">{label}</span>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
    </div>
    """

def main():
    # Header Section
    st.markdown('<div class="system-title">Solar Flare <span>Prediction System</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="system-subtitle">// Space-Weather Intelligence Dashboard v2.0</div>', unsafe_allow_html=True)

    config, df, preprocessor, models, metrics_summary = load_application_artifacts()

    if len(models) == 0:
        st.warning("⚠️ No trained models found in `models/` directory. Please run model training first.")
        st.stop()

    timestamp_col = config["data"].get("timestamp_col", "timestamp")
    target_col = config["data"].get("target_col", "target_m_plus_24h")
    feature_names = preprocessor.feature_names
    seq_length = config["forecast"].get("history_hours", 24)

    # Sidebar Controls
    with st.sidebar:
        st.markdown('<div class="system-subtitle" style="margin-bottom:1rem;color:#FF6B35;">// CONTROL PANEL</div>', unsafe_allow_html=True)
        
        st.subheader("⚙️ Active Mission Model")
        selected_model_name = st.selectbox("Select Model Architecture", options=list(models.keys()), index=0)

        st.markdown("---")
        st.subheader("⏱️ Temporal Navigation")
        use_latest = st.checkbox("Engage Live Feed (Latest Target)", value=True)
        
        available_timestamps = df[timestamp_col].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        valid_start_idx = seq_length - 1
        valid_timestamps = available_timestamps[valid_start_idx:]

        if use_latest:
            selected_timestamp_str = valid_timestamps[-1]
        else:
            selected_timestamp_str = st.selectbox("Select Historical Snapshot", options=valid_timestamps, index=len(valid_timestamps) - 1)

        st.button("🚀 EXECUTE INFERENCE", use_container_width=True, type="primary")

        st.markdown("---")
        st.markdown('<span style="font-size:0.8rem;color:#475569;">SYSTEM STATUS: ONLINE<br>MODULE: FORECAST-24H</span>', unsafe_allow_html=True)

    # Dataset Summary Metrics row using Streamlit columns but styled by CSS
    m1, m2, m3, m4 = st.columns(4)
    min_date = df[timestamp_col].min().strftime('%Y-%m-%d')
    max_date = df[timestamp_col].max().strftime('%Y-%m-%d')
    m1.markdown(render_html_card("Total Observations", f"{len(df):,}", "Active dataset size"), unsafe_allow_html=True)
    m2.markdown(render_html_card("Feature Space", f"{len(feature_names)}", "Magnetogram parameters"), unsafe_allow_html=True)
    m3.markdown(render_html_card("Temporal Range", f"{min_date}", f"to {max_date}"), unsafe_allow_html=True)
    m4.markdown(render_html_card("Observation Cadence", "1.0h", "Fixed hourly frequency"), unsafe_allow_html=True)

    st.markdown("---")

    # Target Index Calculation
    selected_idx = df[df[timestamp_col].dt.strftime("%Y-%m-%d %H:%M:%S") == selected_timestamp_str].index[0]
    window_df = df.iloc[selected_idx - seq_length + 1 : selected_idx + 1].copy()
    
    # Perform Inference
    selected_model = models[selected_model_name]
    X_window_scaled = preprocessor.transform_df(window_df) 

    model_key_map = {
        "Logistic Regression": "logistic_regression",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
        "LSTM": "lstm",
        "1D CNN": "cnn_1d"
    }
    model_key = model_key_map.get(selected_model_name, "logistic_regression")
    saved_threshold = metrics_summary.get(model_key, {}).get("threshold", 0.5)

    if selected_model_name in ("LSTM", "1D CNN"):
        X_seq_input = np.expand_dims(X_window_scaled, axis=0) 
        probability_raw = float(selected_model.predict_proba(X_seq_input)[0])
    else:
        latest_row_scaled = X_window_scaled[-1:, :] 
        probability_raw = float(selected_model.predict_proba(latest_row_scaled)[0])

    is_flare = probability_raw >= saved_threshold
    
    # ---------------------------------------------------------
    # SINGLE SOURCE OF TRUTH CONVERSION
    # ---------------------------------------------------------
    probability_percent = probability_raw * 100

    import math
    if probability_percent == 0.0:
        val_fmt = ".2f"
        formatted_prob = "0.00%"
    elif probability_percent >= 0.01:
        val_fmt = ".2f"
        formatted_prob = f"{probability_percent:.2f}%"
    else:
        digits = int(-math.floor(math.log10(probability_percent))) + 1
        val_fmt = f".{digits}f"
        formatted_prob = f"{probability_percent:.{digits}f}%"

    # Prediction Hero Section
    st.markdown('<div class="system-subtitle" style="color:#00CC88;margin-bottom:1rem;">// PRIMARY FORECAST OUTCOME</div>', unsafe_allow_html=True)
    
    col_pred, col_gauge = st.columns([1.2, 1])

    with col_pred:
        
        actual_str = "Verification Pending"
        if target_col in window_df.columns:
            actual_label = window_df[target_col].iloc[-1]
            actual_str = "DETECTED" if actual_label == 1 else "NO FLARE"

        badge_html = '<div class="badge-alert">⚠ CRITICAL M+ FLARE EXPECTED</div>' if is_flare else '<div class="badge-safe">✓ NOMINAL: NO M+ EXPECTED</div>'
        
        st.html(f"""
<div class="sci-card-outer">
    <div class="sci-card-inner" style="position: relative;">
        <div style="font-family:'JetBrains Mono'; color:#475569; font-size:0.85rem; margin-bottom: 2rem;">
            [ TARGET T0 ]: {selected_timestamp_str} <br>
            [ ALGORITHM ]: {selected_model_name} <br>
            [ THRESHOLD ]: {saved_threshold:.3f}
        </div>
        
        <span class="metric-label">Likelihood (Internal)</span>
        <div class="metric-value" style="font-size:3.5rem; margin-bottom:0.2rem; color:{'#FF3366' if is_flare else '#00CC88'}">
            {formatted_prob}
        </div>
        
        {badge_html}
        
        <div style="position:absolute; bottom: 1.5rem; right: 1.5rem; text-align:right;">
            <span class="metric-label" style="margin:0;">Ground Truth</span>
            <span style="font-family:'JetBrains Mono'; font-weight:700; color:#64748B;">{actual_str}</span>
        </div>
    </div>
</div>
""")

    with col_gauge:
        st.markdown(f'''
<div class="sci-card-outer" style="height:calc(100% - 1.5rem);">
    <div class="sci-card-inner" style="display: flex; flex-direction: column;">
        <div style="font-family:'Space Grotesk', sans-serif; font-size:0.9rem; color:#64748B; text-align:center; padding-bottom:0.75rem; border-bottom:1px solid rgba(255,255,255,0.05); margin-bottom:1rem;">
            PREDICTION vs THRESHOLD
        </div>
        <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono', monospace; font-size:1rem; padding: 0 1.5rem; margin-bottom: 0.5rem;">
            <span style="color:#94A3B8;">Prediction</span>
            <span style="color:#E2E8F0; font-weight:700;">{formatted_prob}</span>
        </div>
        <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono', monospace; font-size:1rem; padding: 0 1.5rem; margin-bottom: 0.5rem;">
            <span style="color:#FF6B35;">Threshold</span>
            <span style="color:#FF6B35; font-weight:700;">{saved_threshold * 100:.2f}%</span>
        </div>
        ''', unsafe_allow_html=True)
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability_percent,
            number={'valueformat': val_fmt, 'suffix': "%", 'font': {'family': 'JetBrains Mono, monospace', 'color': '#E2E8F0', 'size': 32}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569", 'tickfont': {'color':'#475569', 'family':'JetBrains Mono'}},
                'bar': {'color': "#FF3366" if is_flare else "#00CC88", 'thickness': 0.15},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 1,
                'bordercolor': "rgba(255,255,255,0.05)",
                'steps': [
                    {'range': [0, saved_threshold * 100], 'color': "rgba(0,0,0,0)"},
                    {'range': [saved_threshold * 100, 100], 'color': "rgba(255,51,102,0.05)"}
                ],
                'threshold': {
                    'line': {'color': "#FF6B35", 'width': 2},
                    'thickness': 0.8,
                    'value': saved_threshold * 100
                }
            }
        ))
        fig_gauge.update_layout(height=230, margin=dict(l=20, r=20, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#94A3B8"})
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Interactive Solar Feature History Plot
    st.markdown('<div class="system-subtitle" style="color:#64748B;">// 24-HOUR TELEMETRY FEED (T0-24 -> T0)</div>', unsafe_allow_html=True)
    
    selected_feature = st.selectbox(
        "Select Magnetogram Parameter",
        options=feature_names,
        index=0
    )

    fig_time = px.line(
        window_df,
        x=timestamp_col,
        y=selected_feature,
        markers=True,
    )
    fig_time.update_traces(line_color="#00CC88", marker=dict(size=6, color="#00CC88", symbol="cross"))
    fig_time.update_layout(
        xaxis_title="T (UTC)",
        yaxis_title=f"VAL ({selected_feature})",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        font={'family': 'JetBrains Mono, monospace', 'color': '#64748B'},
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        margin=dict(t=20, b=20, l=10, r=10)
    )
    st.plotly_chart(fig_time, use_container_width=True)

    # Model Performance Comparison Section
    if len(metrics_summary) > 0:
        st.markdown("---")
        st.markdown('<div class="system-subtitle" style="color:#64748B;">// MODEL VALIDATION BENCHMARKS (OFFLINE)</div>', unsafe_allow_html=True)
        
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
        
        st.dataframe(metrics_df.style.format(precision=4).highlight_max(axis=0, subset=["PR-AUC", "ROC-AUC", "F1 Score", "TSS"], color="#FF6B35"), use_container_width=True)

if __name__ == "__main__":
    main()
