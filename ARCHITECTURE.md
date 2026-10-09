# HelioCast System Architecture & Technical Specification

An in-depth technical reference for the HelioCast operational solar flare forecasting system, detailing the decoupled FastAPI backend, React.js single-screen landscape operations console, machine learning inference pipeline, and anti-leakage safeguards.

---

## 1. System Overview & Core Objectives

HelioCast is an operational space-weather forecasting platform engineered to predict whether an **M-class or higher solar flare ($\ge \text{M1.0}$, soft X-ray flux $\ge 10^{-5} \text{ W/m}^2$) will erupt within 24 hours** following an observation timestamp $T_0$.

### Core Design Requirements
1. **Zero Lookahead Leakage**: Enforce strict chronological separation across data splitting, feature scaling, sequence windowing, and threshold optimization.
2. **Decoupled Production Architecture**: Decouple the ML inference engine from the user interface using a high-throughput FastAPI REST backend and a zero-latency React.js frontend.
3. **Ergonomic Landscape Console**: Present a single-screen landscape workstation fitting 1080p displays at 100% zoom with zero downward page scrolling required.
4. **Honest Test Benchmarks**: Display only verifiable test-set evaluation metrics derived from frozen validation thresholds.

---

## 2. End-to-End Architecture & Dataflow

```text
+--------------------------------------------------------------------------------------------------+
|                                    SOLAR OBSERVATION ARCHIVE                                     |
|               Hourly SDO/HMI Active Region Patches & GOES X-ray / Proton Telemetry               |
+--------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+--------------------------------------------------------------------------------------------------+
|                               LEAKAGE-FREE PREPROCESSING PIPELINE                                |
|        - Chronological Partitioning: Train (70%), Validation (15%), Test (15%)                   |
|        - StandardScaler & Median Imputations fit EXCLUSIVELY on Training Split                   |
|        - Scaler parameters persisted to models/scaler.joblib                                     |
+--------------------------------------------------------------------------------------------------+
                                                 |
                       +-------------------------+-------------------------+
                       |                                                   |
                       v                                                   v
+---------------------------------------------+   +------------------------------------------------+
|       CLASSICAL ML INFERENCE ENGINE         |   |            RECURRENT LSTM ENGINE               |
|      (Logistic Regression, RF, XGBoost)     |   |                (PyTorch LSTM)                  |
|    - Input: Instantaneous Snapshot [1, 11]  |   |    - Sequence Service: Sliding Window [1, 24, 11]
|    - Evaluates magnetic energy state at T0  |   |    - Captures 24h temporal flux gradients      |
+---------------------------------------------+   +------------------------------------------------+
                       |                                                   |
                       +-------------------------+-------------------------+
                                                 |
                                                 v
+--------------------------------------------------------------------------------------------------+
|                               FROZEN VALIDATION THRESHOLD ENGINE                                 |
|         - Decision thresholds tuned on 15% Validation Split to maximize TSS and F1 Score          |
|         - Thresholds permanently frozen in results/metrics/*.json prior to Test evaluation       |
|         - Applied in real-time inference: P(Flare) >= Threshold => M+ Hazard Alert               |
+--------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+--------------------------------------------------------------------------------------------------+
|                                     FASTAPI SERVICE (backend/)                                   |
|         - Lifespan Preloader: Preloads models, scaler, and dataset into memory                   |
|         - CORS Middleware: Configurable origins for custom production domain binding             |
|         - Endpoints: /api/v1/predict, /api/v1/telemetry, /api/v1/metrics, /api/v1/health         |
+--------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+--------------------------------------------------------------------------------------------------+
|                                 REACT.JS OPERATIONAL CONSOLE (frontend/)                         |
|         - Single-Screen Landscape Viewport Fit (100% Zoom, Zero Outer Page Scrolling)            |
|         - Hero Forecast Result Card (Prominent Probability Readout & Decision Gauge Bar)         |
|         - Segmented Model Cards, Time Scrubbing Slider, & Auto-Run Debounced Inference           |
|         - High-Fidelity SVG Telemetry Progression Chart (Peak Value Callout & Shaded Bands)      |
|         - Multi-Model Consensus Comparison Dialog (Simultaneous /predict evaluation)             |
|         - Sortable Benchmarks Table & Interactive Visual Comparison Bar Chart                    |
+--------------------------------------------------------------------------------------------------+
```

---

## 3. Data Pipeline & Feature Engineering

### 3.1 Input Feature Space
The feature vector comprises 11 physical magnetogram and solar irradiance telemetry parameters:

| Parameter | Unit | Physical Description |
| :--- | :---: | :--- |
| `xrsb_flux` | $\text{W/m}^2$ | GOES primary longwave soft X-ray flux ($0.1 - 0.8\text{ nm}$). International flare class standard. |
| `xrsa_flux` | $\text{W/m}^2$ | GOES primary shortwave soft X-ray flux ($0.05 - 0.4\text{ nm}$). High-energy thermal emission. |
| `proton_gt10mep` | $\text{pfu}$ | Energetic solar proton flux ($E > 10\text{ MeV}$). Spacecraft radiation hazard indicator. |
| `ratio_ab` | ratio | Shortwave to longwave flux ratio ($\text{XRSA} / \text{XRSB}$). Coronal temperature proxy. |
| `xrsa_log` | $\log_{10}$ | Base-10 logarithmic transformation of shortwave X-ray flux. |
| `xrsb_log` | $\log_{10}$ | Base-10 logarithmic transformation of longwave X-ray flux. |
| `xrsb_rate_1h` | $\text{W/m}^2\cdot\text{h}$ | 1-hour temporal rate of change derivative of longwave X-ray flux. |
| `xrsb_rate_3h` | $\text{W/m}^2\cdot\text{h}$ | 3-hour temporal rate of change derivative of longwave X-ray flux. |
| `xrsb_max_6h` | $\text{W/m}^2$ | Maximum longwave X-ray flux recorded over the preceding 6 hours. |
| `xrsb_std_6h` | $\text{W/m}^2$ | Rolling 6-hour standard deviation capturing coronal magnetic field turbulence. |
| `xrsa_max_6h` | $\text{W/m}^2$ | Maximum shortwave X-ray flux recorded over the preceding 6 hours. |

### 3.2 Target Formulation
- **Variable**: `target_m_plus_24h`
- **Definition**: Binary indicator where $y = 1$ denotes that at least one M-class or X-class solar flare occurred within $[T_0, T_0 + 24\text{h}]$, and $y = 0$ denotes nominal background activity ($<\text{M1.0}$).

---

## 4. Strict Anti-Leakage Safeguards

Operational space weather systems require strict protocols to prevent artificial performance inflation:

1. **Strict Chronological Data Splitting**:
   - Random shuffling (such as k-fold cross validation) is prohibited due to temporal autocorrelation across solar active region lifetimes.
   - Partitions:
     - **Training Split (70%)**: Historical baseline period ($N = 2,958$ hours).
     - **Validation Split (15%)**: Intermediate chronological period used for hyperparameter tuning and decision threshold optimization ($N = 634$ hours).
     - **Test Split (15%)**: Most recent chronological period held out completely until final benchmarking ($N = 634$ hours).

2. **Leakage-Free Feature Scaling**:
   - Feature means $\mu_{\text{train}}$ and standard deviations $\sigma_{\text{train}}$, along with median imputation values, are fitted strictly on the Training Split.
   - Validation and Test splits are transformed using these frozen statistics:
     $$X_{\text{scaled}} = \frac{X - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
   - Preprocessing parameters are persisted in `models/scaler.joblib`.

3. **Isolated Sequence Windowing**:
   - For recurrent models, 24-step sequence windows `[samples, 24, 11]` are generated strictly within split boundaries.
   - No sequence window bridges across split boundaries, ensuring zero temporal contamination.

---

## 5. Model Architectures & Inference Mechanics

### 5.1 Classical Tabular Classifiers
Classical models operate on the instantaneous feature vector at observation time $T_0$ (shape: `[1, 11]`):
- **Logistic Regression (`models/logistic_regression.joblib`)**: Linear model utilizing L2 regularization and `class_weight='balanced'`.
- **Random Forest (`models/random_forest.joblib`)**: Ensemble of 300 de-correlated decision trees with balanced subsampling.
- **XGBoost (`models/xgboost.joblib`)**: Gradient boosted decision tree ensemble with `scale_pos_weight` tuned to the positive class inverse prevalence ratio.

### 5.2 Recurrent Deep Learning Architecture
- **LSTM (`models/lstm.pt`)**:
  - Input: 24-hour historical sequence matrix (shape: `[1, 24, 11]`), capturing temporal gradients, magnetic energy accumulation, and flux surges.
  - Architecture: 2-layer recurrent LSTM network (hidden dimension: 128, dropout: 0.20) followed by a dense linear classification layer.
  - Loss function: Binary Cross-Entropy with Logits, applying positive class weighting to counteract flare rarity.

---

## 6. Decision Boundaries & Frozen Validation Thresholds

### 6.1 The 0.50 Threshold Pathology
Solar flare eruptions occur in less than 5% of observation hours. A standard default classification threshold of $\theta = 0.50$ results in catastrophic false negative rates, failing to warn satellite operators of dangerous geomagnetic radiation storms.

### 6.2 Optimization Protocol
1. Each trained model produces continuous probabilities $P(y=1)$ on the independent Validation split.
2. A threshold sweep $\theta \in [0.01, 0.99]$ evaluates the **True Skill Statistic (TSS)** and **F1 Score**:
   $$\text{TSS} = \text{Recall} - \text{FPR} = \frac{\text{TP}}{\text{TP} + \text{FN}} - \frac{\text{FP}}{\text{FP} + \text{TN}}$$
3. The optimal threshold $\theta^*$ is selected to maximize TSS and F1 on the Validation split.
4. **Threshold Freezing**: The selected threshold is permanently frozen in `results/metrics/{model}.json`.
5. The Test split and live operational inferences use this frozen threshold with no post-hoc adjustment.

### 6.3 Active Frozen Operational Thresholds
- **LSTM**: $\theta^* = 0.3300$
- **Logistic Regression**: $\theta^* = 0.8000$
- **Random Forest**: $\theta^* = 0.5800$
- **XGBoost**: $\theta^* = 0.2100$

---

## 7. Offline Test Set Benchmark Matrix

Evaluated on the held-out 15% chronological test split ($N = 634$ observation windows, 69 positive M+ events, 565 negative events):

| Architecture | PR-AUC | TSS | ROC-AUC | F1 Score | Recall | Precision | FAR | Frozen Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM** | **0.3664** | **0.8018** | **0.9096** | **0.5520** | **1.0000** | 0.3812 | 0.6188 | 0.3300 |
| **Random Forest** | 0.3383 | 0.3771 | 0.8923 | 0.3977 | 0.4928 | 0.3333 | 0.6667 | 0.5800 |
| **Logistic Regression** | 0.3488 | 0.2782 | 0.8896 | 0.3399 | 0.3768 | 0.3095 | 0.6905 | 0.8000 |
| **XGBoost** | 0.3069 | 0.4692 | 0.8843 | 0.4352 | 0.6087 | 0.3387 | 0.6613 | 0.2100 |

*Key Finding: The PyTorch LSTM architecture achieves a True Skill Statistic of 0.8018 and a Recall of 1.0000, successfully identifying 100% of M-class flares on the unseen test set.*

---

## 8. Frontend Architecture & Workstation Ergonomics

### 8.1 Single-Screen Landscape Viewport Fit
The UI is strictly engineered to fit within a single 1080p landscape viewport at 100% browser zoom with zero outer vertical scrollbars:
- `html, body, #root`: Locked to `100%` viewport width and height (`overflow: hidden`).
- **Compact Single-Tier Header (`h-13`)**: Unifies brand title, live telemetry status dot, and navigation tabs into one bar.
- **Left Column (5 cols)**: Compact Hero Result card and Forecast Controls card sized to fit within vertical limits.
- **Right Column (7 cols)**: Telemetry header, compact 4-stat KPI row, and flexible SVG line chart expanding to fill remaining vertical space.
- **Slim Footer (`h-7`)**: Fixed operational status strip.

### 8.2 Component Hierarchy

```text
App.jsx
├── Header (Brand, Status Dot, 3 Navigation Tabs, Guide Link, Refresh)
├── Main Workspace (12-Column Grid, Full Viewport Height)
│   ├── [Tab: Forecast]
│   │   ├── Left Column: PredictionPanel.jsx
│   │   │   ├── Hero Outcome Card (Probability Readout & Linear Decision Bar)
│   │   │   ├── Model Selector (2x2 Segmented Non-Truncated Cards)
│   │   │   ├── Time Selector (Live/Archive Toggle, Step Arrows, Timeline Slider)
│   │   │   ├── Action Button ("Run forecast" + Auto-Run Debounce Toggle)
│   │   │   └── ModelComparisonModal.jsx (Consensus Comparison Dialog)
│   │   └── Right Column: TemporalChart.jsx
│   │       ├── Telemetry Header & Parameter Chips
│   │       ├── Compact 4-Stat KPI Row (Current, Min, Max, Average with Delta)
│   │       └── High-Fidelity SVG Chart (Shaded Band, Peak Callout, Crosshair)
│   ├── [Tab: Model Benchmarks]
│   │   └── MetricsMatrix.jsx
│   │       ├── Interactive Metric Comparison Bar Chart
│   │       └── Sortable Benchmarks Table with Confusion Matrix Breakdowns
│   └── [Tab: How It Works]
│       └── ThresholdGuide.jsx
│           ├── 3 Methodology Explanatory Cards
│           ├── Active Operational Thresholds Reference Table
│           └── Space Weather Plain-English Glossary
└── Footer (Cadence, Horizon, Zero-Leakage Guarantee)
```

---

## 9. API Specifications & Endpoints

All backend routes are served under prefix `/api/v1`:

### 9.1 Endpoint Summary

| Route | Method | Payload / Params | Return Type | Description |
| :--- | :---: | :--- | :--- | :--- |
| `/health` | `GET` | None | `HealthResponse` | System health and list of active models. |
| `/summary` | `GET` | None | `DatasetSummaryResponse` | Dataset size, parameter list, and coverage. |
| `/models` | `GET` | None | `{"models": [...]}` | Model architectures, descriptions, thresholds. |
| `/metrics` | `GET` | None | `MetricsSummaryResponse` | Offline test-set evaluation benchmarks. |
| `/timestamps` | `GET` | None | `AvailableTimestampsResponse` | Sorted timestamps with 24h history. |
| `/telemetry` | `GET` | `timestamp` (optional) | `TelemetryWindowResponse` | 24-hour sequence of SHARP parameters. |
| `/predict` | `POST` | `PredictionRequest` | `PredictionResponse` | Real-time zero-leakage model inference. |

### 9.2 Request & Response Payloads

#### Prediction Request (`POST /api/v1/predict`)
```json
{
  "model_name": "lstm",
  "timestamp": "2026-09-30 00:00:00",
  "custom_window": null
}
```

#### Prediction Response (`200 OK`)
```json
{
  "model_name": "lstm",
  "model_display_name": "LSTM (Recurrent Neural Network)",
  "target_timestamp": "2026-09-30 00:00:00",
  "probability": 0.362145,
  "probability_percent": 36.2145,
  "probability_formatted": "36.21%",
  "threshold": 0.33,
  "is_flare": true,
  "risk_level": "CRITICAL",
  "ground_truth": 1,
  "ground_truth_label": "DETECTED",
  "sequence_length": 24,
  "window_timestamps": [
    "2026-09-29 01:00:00",
    "...",
    "2026-09-30 00:00:00"
  ]
}
```

---

## 10. Production Deployment & Readiness

### 10.1 Environment Configuration
- **Backend**:
  - `PROJECT_ROOT`: Base repository directory path.
  - `CORS_ORIGINS`: Comma-separated or list of allowed web origins (e.g., `https://heliocast.yourdomain.com`).
- **Frontend**:
  - `VITE_API_BASE_URL`: URL to the FastAPI backend (e.g., `https://api.heliocast.yourdomain.com/api/v1`). Defaults to `/api/v1` with local Vite proxy.

### 10.2 Production Build Commands
```bash
# Backend verification
pytest backend/tests/test_api.py -v

# Frontend production bundle
cd frontend
npm run build
```
The compiled frontend bundle is emitted to `frontend/dist` and can be served using Nginx, Caddy, or cloud storage CDNs with API reverse proxying.
