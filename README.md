# HelioCast: Solar Flare Forecasting System

An operational space weather forecasting application predicting whether an **M-class or higher solar flare ($\ge \text{M1.0}$) will erupt within the next 24 hours** using historical SDO/HMI and GOES solar magnetogram parameters.

The application features a decoupled, production-grade architecture combining a **FastAPI backend** and a high-density, **single-screen landscape React.js operational workstation**.

---

## 1. System Architecture

```text
+--------------------------------------------------------------------------------+
|                           SOLAR OBSERVATION ARCHIVE                            |
|             (Hourly GOES X-ray Flux & SDO/HMI Magnetogram Parameters)           |
+--------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
|                     LEAKAGE-FREE PREPROCESSING PIPELINE                        |
|       - Chronological 70/15/15 Train/Val/Test Split (No Random Shuffling)      |
|       - StandardScaler & Median Imputation Fit Exclusively on Train Split      |
|       - 24-Hour Sliding Window Sequences Formed Within Split Boundaries        |
+--------------------------------------------------------------------------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
+--------------------------------------+   +--------------------------------------+
|       CLASSICAL ML ARCHITECTURES     |   |       RECURRENT DEEP LEARNING        |
|  (Logistic Regression, RF, XGBoost)  |   |           (PyTorch LSTM)             |
|   Input: Instantaneous T0 Snapshot   |   |    Input: 24h Sequence [1, 24, 11]   |
+--------------------------------------+   +--------------------------------------+
                   |                                       |
                   +-------------------+-------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
|                     FROZEN VALIDATION THRESHOLD TUNING                         |
|   Thresholds optimized on 15% Validation Split to maximize TSS (Recall - FPR)   |
|     Permanently frozen prior to test benchmarking and operational inference    |
+--------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
|                          FASTAPI SERVICE (backend/)                            |
|        - Preloads Scaler, Dataset, and PyTorch / Scikit-Learn Model Weights     |
|        - Endpoints: /api/v1/predict, /api/v1/telemetry, /api/v1/metrics        |
|        - Configurable CORS Origins & Environment Bindings                      |
+--------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
|                    REACT.JS LANDSCAPE CONSOLE (frontend/)                      |
|        - 100% Zoom Landscape Workstation Fit (Zero Outer Page Scrolling)       |
|        - Primary Hero Outcome Card (Large Probability & Comparative Gauge)     |
|        - Segmented Non-Truncated Model Cards & Timeline Scrubber Slider        |
|        - 24-Hour Telemetry SVG Chart with Shaded Band & Peak Value Callout     |
|        - Multi-Model Consensus Evaluation Dialog (/predict across 4 models)    |
|        - 3 Dedicated Workspace Tabs: Forecast, Benchmarks, How It Works        |
+--------------------------------------------------------------------------------+
```

---

## 2. Directory Structure

```text
HelioCast/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI application, CORS, lifespan handler
│   │   ├── api/
│   │   │   └── endpoints.py            # Prediction, telemetry, and benchmark routes
│   │   ├── core/
│   │   │   └── config.py               # Settings, paths, allowed CORS origins
│   │   ├── schemas/
│   │   │   └── prediction.py           # Pydantic schemas for request/response payloads
│   │   └── services/
│   │       ├── inference.py            # Model loading, scaler execution, thresholding
│   │       └── sequence_service.py     # 24-step sliding window constructor for LSTM
│   ├── requirements.txt                # FastAPI, Uvicorn, PyTorch, Scikit-learn, etc.
│   └── tests/
│       └── test_api.py                 # API route and inference integration tests
├── frontend/
│   ├── public/
│   │   └── favicon.svg                 # Solar telemetry active region vector icon
│   ├── src/
│   │   ├── components/
│   │   │   ├── PredictionPanel.jsx     # Hero result, model selector, time slider
│   │   │   ├── TemporalChart.jsx       # 24h progression chart with peak marker & band
│   │   │   ├── MetricsMatrix.jsx       # Sortable table & metric comparison bar chart
│   │   │   ├── ThresholdGuide.jsx      # Decision thresholds and methodology guide
│   │   │   └── ModelComparisonModal.jsx# Multi-model consensus comparison dialog
│   │   ├── services/
│   │   │   └── api.js                  # Fetch client wrapper reading VITE_API_BASE_URL
│   │   ├── App.jsx                     # Single-screen landscape cockpit layout
│   │   ├── main.jsx                    # React DOM entry point
│   │   └── index.css                   # Tailwind CSS, glassmorphism, 100% viewport lock
│   ├── .env.example
│   ├── package.json
│   ├── postcss.config.js
│   ├── tailwind.config.js
│   └── vite.config.js
├── models/                             # Pre-trained models (.joblib, .pt) and scaler
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── lstm.pt
│   └── scaler.joblib
├── configs/
│   └── config.yaml                     # System hyperparameters and configuration
├── data/
│   └── processed/
│       └── solar_features.csv          # Chronological solar observation dataset
├── results/
│   └── metrics/                        # Offline test-set metric reports (JSON)
│       ├── logistic_regression.json
│       ├── random_forest.json
│       ├── xgboost.json
│       └── lstm.json
├── docs/
│   └── ML_ARCHITECTURE.md              # Technical ML specification
├── ARCHITECTURE.md                     # Comprehensive architecture and dataflow guide
└── README.md
```

---

## 3. Quick Start Guide

### 3.1 Backend Setup (FastAPI)

1. Open a terminal and navigate to the backend directory:
   ```powershell
   cd backend
   ```

2. Activate the virtual environment (if using the local `.venv`):
   ```powershell
   ..\.venv\Scripts\Activate.ps1
   ```

3. Install requirements:
   ```powershell
   pip install -r requirements.txt
   ```

4. Launch the FastAPI development server:
   ```powershell
   uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```

5. Verify backend status:
   - Interactive Swagger API Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - Health Verification: [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)

### 3.2 Frontend Setup (React.js + Vite)

1. Open a second terminal window and navigate to the frontend directory:
   ```powershell
   cd frontend
   ```

2. Install dependencies (Node.js/npm required):
   ```powershell
   npm install
   ```

3. Launch the Vite development server:
   ```powershell
   npm run dev
   ```

4. Access the operational dashboard at [http://localhost:5173](http://localhost:5173). The console loads immediately upon startup with zero authentication barriers.

---

## 4. Operational Dashboard Features

- **Single-Screen Landscape Cockpit**: Specifically engineered to fit 1080p and widescreen displays at 100% zoom with zero downward page scrolling required.
- **Hero Forecast Result**: Instant visual hierarchy featuring a large 48px+ probability readout, decision threshold gauge bar, and plain-English risk headline (`Low Risk`, `Elevated Risk`, or `High Risk`).
- **Multi-Model Consensus Comparison**: 1-click comparison button evaluating all 4 model architectures simultaneously to display agreement and consensus confidence.
- **24-Hour Telemetry Progression**: High-fidelity SVG line chart featuring:
  - Shaded fluctuation bands.
  - Callout pin for the 24-hour **Peak Value**.
  - Highlighted halo dot for the **Current Value (Now)**.
  - Hover crosshairs with floating UTC timestamp, offset, value, and unit tooltips.
  - Quick-select parameter chips for primary X-ray and proton channels.
- **Sortable Benchmarks Table**: Offline test-set metrics sortable by any column with lead benchmark highlights and expandable confusion matrices.
- **Visual Metric Bar Chart**: Interactive horizontal bar comparison charting model performance across TSS, PR-AUC, F1 Score, Recall, Precision, and ROC-AUC.
- **Auto-Run Debouncing**: Automatic background inference (~350ms debounce) whenever models or timeline steps change, with a toggle switch.
- **Persistence**: Remembers active tab, selected model, and auto-run state in `localStorage`.

---

## 5. API Endpoints Reference

All routes are served under prefix `/api/v1`:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | System health and list of verified active models |
| `GET` | `/summary` | Dataset observation volume, parameter names, and coverage |
| `GET` | `/models` | Available model architectures and metadata |
| `GET` | `/metrics` | Tabular offline test-set evaluation benchmarks |
| `GET` | `/timestamps` | Sorted chronological timestamps with full 24h history |
| `GET` | `/telemetry?timestamp={T0}` | 24-hour sequence of SHARP parameters for charting |
| `POST` | `/predict` | Executes real-time inference for selected model and timestamp |

### Example Prediction Request
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "lstm",
    "timestamp": "2026-09-30 00:00:00"
  }'
```

### Example Prediction Response
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

## 6. Offline Test-Set Benchmark Summary

Evaluated on the held-out 15% chronological test split ($N = 634$ observation windows):

| Model Architecture | PR-AUC | TSS | ROC-AUC | F1 Score | Recall | Precision | FAR | Frozen Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM** | **0.3664** | **0.8018** | **0.9096** | **0.5520** | **1.0000** | 0.3812 | 0.6188 | 0.3300 |
| **Random Forest** | 0.3383 | 0.3771 | 0.8923 | 0.3977 | 0.4928 | 0.3333 | 0.6667 | 0.5800 |
| **Logistic Regression** | 0.3488 | 0.2782 | 0.8896 | 0.3399 | 0.3768 | 0.3095 | 0.6905 | 0.8000 |
| **XGBoost** | 0.3069 | 0.4692 | 0.8843 | 0.4352 | 0.6087 | 0.3387 | 0.6613 | 0.2100 |

For full mathematical equations, dataflow diagrams, and sequence specifications, refer to [ARCHITECTURE.md](ARCHITECTURE.md) and [docs/ML_ARCHITECTURE.md](docs/ML_ARCHITECTURE.md).

---

## 7. Testing

Run backend tests using PyTest:
```powershell
pytest backend/tests/test_api.py -v
```
