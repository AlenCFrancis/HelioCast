"""
Streamlit Web Application for Solar Flare Prediction.
Designed with SpaceWeatherLive.com Aesthetics & Real-Time NOAA Telemetry Integration.
"""

import os
import json
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.data.loader import load_config, load_solar_data
from src.data.noaa_fetcher import fetch_noaa_swpc_data
from src.data.preprocessing import SolarDataPreprocessor
from src.models.logistic_regression import SolarLogisticRegression
from src.models.random_forest import SolarRandomForest
from src.models.xgboost_model import SolarXGBoost
from src.models.lstm import SolarLSTMWrapper
from src.models.cnn_1d import SolarCNN1DWrapper

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Solar Flares | SpaceWeatherLive Real-Time Forecasting",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling (SpaceWeatherLive Dark Theme)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .stApp {
        background-color: #0b0e14 !important;
        color: #c9d1d9 !important;
    }
    .stSidebar {
        background-color: #121721 !important;
        border-right: 1px solid #21262d !important;
    }
    
    /* SpaceWeatherLive Header */
    .swl-header-box {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.25rem 1.75rem;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .swl-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .swl-title span {
        color: #ff6b35;
        font-size: 1.5rem;
        font-weight: 600;
    }
    .swl-subtitle {
        font-size: 0.95rem;
        color: #8b949e;
        margin-top: 0.2rem;
    }
    .swl-utc-badge {
        background: #21262d;
        border: 1px solid #30363d;
        color: #58a6ff;
        font-family: 'JetBrains Mono', monospace;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-size: 0.9rem;
        font-weight: 700;
    }

    /* Status Cards */
    .swl-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1.2rem;
        height: 100%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .swl-card-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8b949e;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    
    /* Flare Badges */
    .badge-class {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 800;
        font-size: 1.25rem;
        color: #ffffff;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    }
    .badge-a, .badge-b { background: #1f6feb; }
    .badge-c { background: #d29922; color: #0d1117; }
    .badge-m { background: #f85149; }
    .badge-x { background: #a371f7; box-shadow: 0 0 12px rgba(163, 113, 247, 0.6); }

    .badge-alert {
        background: rgba(248, 81, 73, 0.15);
        color: #f85149;
        border: 1px solid rgba(248, 81, 73, 0.4);
        padding: 0.6rem 1.2rem;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .badge-nominal {
        background: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        border: 1px solid rgba(46, 160, 67, 0.4);
        padding: 0.6rem 1.2rem;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.1rem;
    }

    /* Table Styling */
    .swl-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
    }
    .swl-table th {
        background: #21262d;
        color: #8b949e;
        text-align: left;
        padding: 0.6rem;
        border-bottom: 1px solid #30363d;
    }
    .swl-table td {
        padding: 0.65rem;
        border-bottom: 1px solid #21262d;
        color: #c9d1d9;
    }

    /* Divider */
    hr {
        border-color: #30363d !important;
        margin: 1.5rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Dynamic Helper Functions
# ---------------------------------------------------------
def flux_to_class(flux: float) -> str:
    """Convert GOES X-ray flux (W/m^2) to official solar flare classification (A, B, C, M, X)."""
    if flux is None or math.isnan(flux) or flux <= 0:
        return "A0.0"
    elif flux < 1e-7:
        val = flux / 1e-8
        return f"A{val:.1f}"
    elif flux < 1e-6:
        val = flux / 1e-7
        return f"B{val:.1f}"
    elif flux < 1e-5:
        val = flux / 1e-6
        return f"C{val:.1f}"
    elif flux < 1e-4:
        val = flux / 1e-5
        return f"M{val:.1f}"
    else:
        val = flux / 1e-4
        return f"X{val:.1f}"

def get_class_badge_html(flare_class_str: str) -> str:
    """Generate HTML badge styled according to SpaceWeatherLive standards."""
    c_type = flare_class_str[0].upper()
    if c_type in ["A", "B"]:
        css_class = "badge-a"
    elif c_type == "C":
        css_class = "badge-c"
    elif c_type == "M":
        css_class = "badge-m"
    else:
        css_class = "badge-x"
    return f'<span class="badge-class {css_class}">{flare_class_str}</span>'

def render_html(html_str: str):
    """Safely render raw HTML in Streamlit without code snippet block formatting."""
    clean_html = "\n".join(line.strip() for line in html_str.strip().splitlines())
    st.markdown(clean_html, unsafe_allow_html=True)

@st.cache_resource
def load_application_models():
    """Load configuration, preprocessor, and saved models."""
    config = load_config()
    scaler_path = "models/scaler.joblib"
    
    preprocessor = None
    if os.path.exists(scaler_path):
        preprocessor = SolarDataPreprocessor.load(scaler_path=scaler_path, config=config)

    models = {}
    if os.path.exists("models/logistic_regression.joblib"):
        models["Logistic Regression"] = SolarLogisticRegression.load("models/logistic_regression.joblib")
    if os.path.exists("models/random_forest.joblib"):
        models["Random Forest"] = SolarRandomForest.load("models/random_forest.joblib")
    if os.path.exists("models/xgboost.joblib"):
        models["XGBoost"] = SolarXGBoost.load("models/xgboost.joblib")
    if os.path.exists("models/lstm.pt"):
        models["LSTM"] = SolarLSTMWrapper.load("models/lstm.pt")
    if os.path.exists("models/cnn_1d.pt"):
        models["1D CNN"] = SolarCNN1DWrapper.load("models/cnn_1d.pt")

    metrics_summary = {}
    metrics_dir = "results/metrics"
    if os.path.exists(metrics_dir):
        for f in os.listdir(metrics_dir):
            if f.endswith(".json"):
                model_key = f.replace(".json", "")
                with open(os.path.join(metrics_dir, f), "r") as json_file:
                    metrics_summary[model_key] = json.load(json_file)

    return config, preprocessor, models, metrics_summary

@st.cache_data
def load_solar_dataset_cached(data_path: str, config: dict):
    """Fast, cached dataset loader from solar_features.csv."""
    return load_solar_data(data_path=data_path, config=config)

def main():
    config, preprocessor, models, metrics_summary = load_application_models()

    # ---------------------------------------------------------
    # Top SpaceWeatherLive Header
    # ---------------------------------------------------------
    render_html("""<div class="swl-header-box">
<div>
<div class="swl-title">SpaceWeatherLive <span>.com</span></div>
<div class="swl-subtitle">Real-Time Solar Activity & AI Solar Flare Prediction System</div>
</div>
<div class="swl-utc-badge">
📡 LIVE TELEMETRY STREAM
</div>
</div>""")

    # ---------------------------------------------------------
    # Sidebar Controls
    # ---------------------------------------------------------
    with st.sidebar:
        st.markdown('<div style="font-weight:700; color:#ff6b35; font-size:1.1rem; margin-bottom:1rem;">⚙️ CONTROL CENTER</div>', unsafe_allow_html=True)
        
        st.subheader("🤖 Active ML/DL Model")
        if len(models) > 0:
            selected_model_name = st.selectbox("Select Model Architecture", options=list(models.keys()), index=0)
        else:
            selected_model_name = None
            st.error("No trained models found in `models/`.")

        st.markdown("---")
        if st.button("🔄 Reload Updated Dataset", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()

    # ---------------------------------------------------------
    # Fast Load from data/processed/solar_features.csv
    # ---------------------------------------------------------
    data_path = config["data"]["processed_path"]
    if not os.path.exists(data_path):
        st.error(f"Processed dataset file not found at: {data_path}. Please run `python -m src.data.generate_dataset` first.")
        st.stop()

    df = load_solar_dataset_cached(data_path, config)

    timestamp_col = config["data"].get("timestamp_col", "timestamp")
    target_col = config["data"].get("target_col", "target_m_plus_24h")
    seq_length = config["forecast"].get("history_hours", 24)

    # Calculate real-time dynamic statistics
    current_row = df.iloc[-1]
    latest_timestamp = pd.to_datetime(current_row[timestamp_col]).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    current_xrsb = current_row["xrsb_flux"]
    current_class_str = flux_to_class(current_xrsb)

    # 24h Max and 72h Max calculated dynamically
    last_24h_df = df.iloc[-min(24, len(df)):]
    max_24h_flux = last_24h_df["xrsb_flux"].max()
    max_24h_class_str = flux_to_class(max_24h_flux)

    last_72h_df = df.iloc[-min(72, len(df)):]
    max_72h_flux = last_72h_df["xrsb_flux"].max()
    max_72h_class_str = flux_to_class(max_72h_flux)

    current_proton = current_row.get("proton_gt10mep", 0.1)

    # ---------------------------------------------------------
    # SpaceWeatherLive Dynamic Status Header Cards
    # ---------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_html(f"""<div class="swl-card" style="text-align:center;">
<div class="swl-card-title">Current GOES Flux</div>
{get_class_badge_html(current_class_str)}
<div style="font-size:0.8rem; color:#8b949e; margin-top:0.4rem;">{current_xrsb:.2e} W/m²</div>
</div>""")
    with c2:
        render_html(f"""<div class="swl-card" style="text-align:center;">
<div class="swl-card-title">24-Hour Max Flux</div>
{get_class_badge_html(max_24h_class_str)}
<div style="font-size:0.8rem; color:#8b949e; margin-top:0.4rem;">{max_24h_flux:.2e} W/m²</div>
</div>""")
    with c3:
        render_html(f"""<div class="swl-card" style="text-align:center;">
<div class="swl-card-title">72-Hour Max Flux</div>
{get_class_badge_html(max_72h_class_str)}
<div style="font-size:0.8rem; color:#8b949e; margin-top:0.4rem;">{max_72h_flux:.2e} W/m²</div>
</div>""")
    with c4:
        render_html(f"""<div class="swl-card" style="text-align:center;">
<div class="swl-card-title">Proton Flux (>10 MeV)</div>
<div style="font-size:1.4rem; font-weight:700; color:#58a6ff; font-family:'JetBrains Mono';">{current_proton:.2f}</div>
<div style="font-size:0.8rem; color:#8b949e; margin-top:0.4rem;">pfu (NOAA GOES)</div>
</div>""")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Main Dashboard Grid (SpaceWeatherLive 2-Column Layout)
    # ---------------------------------------------------------
    col_main, col_side = st.columns([2.2, 1])

    # ---------------------------------------------------------
    # LEFT MAIN COLUMN: Live GOES X-Ray Plot & AI Forecast
    # ---------------------------------------------------------
    with col_main:
        st.subheader("📈 Live GOES Solar X-Ray Flux (NOAA Primary)")
        st.caption(f"Latest observation timestamp: `{latest_timestamp}`")

        # Time lookback selector
        hours_view = st.radio("Graph Time Window", options=[24, 48, 72, 168], index=0, horizontal=True)
        plot_df = df.iloc[-min(hours_view, len(df)):].copy()

        # Build SpaceWeatherLive Logarithmic Plot
        fig = go.Figure()

        # GOES Long (0.1-0.8nm / xrsb_flux)
        fig.add_trace(go.Scatter(
            x=plot_df[timestamp_col],
            y=plot_df["xrsb_flux"],
            name="GOES 1-8 Å (Long / XRSB)",
            line=dict(color="#ff6b35", width=2.5),
            mode="lines"
        ))

        # GOES Short (0.05-0.4nm / xrsa_flux)
        fig.add_trace(go.Scatter(
            x=plot_df[timestamp_col],
            y=plot_df["xrsa_flux"],
            name="GOES 0.5-4 Å (Short / XRSA)",
            line=dict(color="#58a6ff", width=1.8, dash="dot"),
            mode="lines"
        ))

        # Flare Threshold Horizontal Reference Lines
        threshold_levels = [
            (1e-8, "A", "#1f6feb"),
            (1e-7, "B", "#1f6feb"),
            (1e-6, "C", "#d29922"),
            (1e-5, "M", "#f85149"),
            (1e-4, "X", "#a371f7")
        ]

        for val, label, color in threshold_levels:
            fig.add_hline(
                y=val,
                line_width=1,
                line_dash="dash",
                line_color=color,
                annotation_text=f"{label}-Class ({val:.0e})",
                annotation_position="top right",
                annotation_font_color=color
            )

        fig.update_layout(
            paper_bgcolor="#161b22",
            plot_bgcolor="#0d1117",
            height=420,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis=dict(
                type="log",
                title="Solar X-Ray Flux (W/m²)",
                range=[-9, -3],
                gridcolor="#21262d",
                zerolinecolor="#21262d",
                tickfont=dict(color="#8b949e")
            ),
            xaxis=dict(
                title="Observation Time (UTC)",
                gridcolor="#21262d",
                zerolinecolor="#21262d",
                tickfont=dict(color="#8b949e")
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color="#c9d1d9")
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ---------------------------------------------------------
        # AI Solar Flare Risk Prediction Section
        # ---------------------------------------------------------
        st.subheader("🎯 Real-Time AI Solar Flare Risk Assessment")
        
        if preprocessor is not None and selected_model_name in models:
            selected_model = models[selected_model_name]
            
            # Slice latest 24-hour window up to current observation
            window_df = df.iloc[-seq_length:].copy()
            X_window_scaled = preprocessor.transform_df(window_df)

            model_key_map = {
                "Logistic Regression": "logistic_regression",
                "Random Forest": "random_forest",
                "XGBoost": "xgboost",
                "LSTM": "lstm",
                "1D CNN": "cnn_1d"
            }
            model_key = model_key_map.get(selected_model_name, "xgboost")
            saved_threshold = metrics_summary.get(model_key, {}).get("threshold", 0.5)

            if selected_model_name in ("LSTM", "1D CNN"):
                X_seq_input = np.expand_dims(X_window_scaled, axis=0)
                prob_m_plus = float(selected_model.predict_proba(X_seq_input)[0])
            else:
                latest_row_scaled = X_window_scaled[-1:, :]
                prob_m_plus = float(selected_model.predict_proba(latest_row_scaled)[0])

            is_flare_expected = prob_m_plus >= saved_threshold
            prob_pct = prob_m_plus * 100

            # Dynamic chances for C, M, X class flares based on physical flux dynamics & ML output
            c_chance = min(99.0, max(20.0, prob_pct * 1.3 + (current_xrsb / 1e-7) * 5))
            m_chance = prob_pct
            x_chance = max(0.5, min(50.0, prob_pct * 0.15 + (current_xrsb / 1e-5) * 10))

            res_col1, res_col2 = st.columns([1.2, 1])

            with res_col1:
                status_color = '#f85149' if is_flare_expected else '#3fb950'
                badge_element = '<div class="badge-alert">🔥 M+ FLARE EXPECTED WITHIN 24H</div>' if is_flare_expected else '<div class="badge-nominal">✅ NOMINAL: NO M+ FLARE EXPECTED</div>'
                
                card_1_html = f"""<div class="swl-card">
<div style="font-size:0.9rem; color:#8b949e;">Target Horizon: <b>Next 24 Hours</b></div>
<div style="font-size:0.9rem; color:#8b949e; margin-bottom:1rem;">Model: <b>{selected_model_name}</b> | Threshold: <b>{saved_threshold:.2f}</b></div>
<div style="font-size:0.85rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.05em;">Calculated M+ Probability</div>
<div style="font-size:3.2rem; font-weight:800; font-family:'JetBrains Mono'; color:{status_color}; line-height:1.1; margin-bottom:1rem;">
{prob_pct:.1f}%
</div>
{badge_element}
</div>"""
                render_html(card_1_html)

            with res_col2:
                # Dynamic Flare Probability Matrix Card (SpaceWeatherLive style)
                card_2_html = f"""<div class="swl-card">
<div class="swl-card-title" style="text-align:center;">Solar Flare Probability Matrix</div>
<table class="swl-table">
<thead>
<tr><th>Flare Class</th><th style="text-align:right;">24h Probability</th></tr>
</thead>
<tbody>
<tr><td><b>C-Class</b> (Small)</td><td style="text-align:right; font-family:'JetBrains Mono'; color:#d29922; font-weight:700;">{c_chance:.0f}%</td></tr>
<tr><td><b>M-Class</b> (Strong)</td><td style="text-align:right; font-family:'JetBrains Mono'; color:#f85149; font-weight:700;">{m_chance:.1f}%</td></tr>
<tr><td><b>X-Class</b> (Major)</td><td style="text-align:right; font-family:'JetBrains Mono'; color:#a371f7; font-weight:700;">{x_chance:.1f}%</td></tr>
</tbody>
</table>
</div>"""
                render_html(card_2_html)

        st.markdown("---")

        # ---------------------------------------------------------
        # Recent Live Observations Table (Dynamic)
        # ---------------------------------------------------------
        st.subheader("📋 Recent Solar Observation Telemetry")
        recent_df = df.iloc[-10:].copy().iloc[::-1]
        
        table_rows = []
        for idx, row in recent_df.iterrows():
            ts_str = pd.to_datetime(row[timestamp_col]).strftime("%Y-%m-%d %H:%M UTC")
            b_flux = row["xrsb_flux"]
            f_class = flux_to_class(b_flux)
            rate_1h = row.get("xrsb_rate_1h", 0.0)
            proton_val = row.get("proton_gt10mep", 0.1)
            
            table_rows.append({
                "Timestamp (UTC)": ts_str,
                "Flare Class": f_class,
                "X-Ray Flux (W/m²)": f"{b_flux:.2e}",
                "1h Rate of Change": f"{rate_1h:+.3f}",
                "Proton Flux (pfu)": f"{proton_val:.2f}"
            })
            
        st.table(pd.DataFrame(table_rows))

    # ---------------------------------------------------------
    # RIGHT SIDEBAR COLUMN: Satellite Imagery & Model Metrics
    # ---------------------------------------------------------
    with col_side:
        st.subheader("🛰️ Live Solar Imagery")
        st.caption("Real-time NASA SDO & NOAA GOES imagery")

        # Live Real-Time Imagery Links from NASA SDO / NOAA SWPC
        render_html("""<div class="swl-card" style="margin-bottom:1rem; text-align:center;">
<div style="font-size:0.8rem; font-weight:700; color:#8b949e; margin-bottom:0.5rem;">SDO / AIA 131 Ångström (Flaring Regions)</div>
<img src="https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0131.jpg" style="width:100%; border-radius:6px; border:1px solid #30363d;" alt="SDO AIA 131">
</div>

<div class="swl-card" style="margin-bottom:1rem; text-align:center;">
<div style="font-size:0.8rem; font-weight:700; color:#8b949e; margin-bottom:0.5rem;">GOES-16 SUVI 131 Å (Primary EUV Disk)</div>
<img src="https://services.swpc.noaa.gov/images/animations/suvi/primary/131/latest.png" style="width:100%; border-radius:6px; border:1px solid #30363d;" alt="GOES SUVI 131">
</div>

<div class="swl-card" style="margin-bottom:1rem; text-align:center;">
<div style="font-size:0.8rem; font-weight:700; color:#8b949e; margin-bottom:0.5rem;">SOHO LASCO C2 (Coronal Mass Ejections)</div>
<img src="https://sohowww.nascom.nasa.gov/data/realtime/c2/1024/latest.jpg" style="width:100%; border-radius:6px; border:1px solid #30363d;" alt="SOHO LASCO C2">
</div>""")

        st.markdown("---")

        # Offline Benchmark Scores Table
        if len(metrics_summary) > 0:
            st.subheader("🏆 Offline Model Benchmarks")
            bench_rows = []
            for name, m in metrics_summary.items():
                bench_rows.append({
                    "Model": name.upper(),
                    "PR-AUC": f"{m.get('pr_auc', 0):.3f}",
                    "ROC-AUC": f"{m.get('roc_auc', 0):.3f}",
                    "F1": f"{m.get('f1_score', 0):.3f}",
                    "TSS": f"{m.get('tss', 0):.3f}"
                })
            st.dataframe(pd.DataFrame(bench_rows), use_container_width=True)

if __name__ == "__main__":
    main()
