# Machine Learning Architecture & Operational Specification

## 1. System Overview

The Solar Flare Forecasting System is an operational time-series classification platform engineered to predict whether an M-class or higher solar flare ($\ge \text{M1.0}$, soft X-ray flux $\ge 10^{-5} \text{ W/m}^2$) will erupt in the 24-hour window following observation time $T_0$.

The application is structured into two decoupled, production-grade subsystems:
1. **FastAPI Backend (`backend/`)**: High-throughput REST API serving pre-trained classical ML models and PyTorch deep learning models, enforcing strict zero-leakage preprocessing, sliding window sequence generation, and frozen validation thresholds.
2. **React.js Frontend (`frontend/`)**: Utility-first scientific telemetry console providing real-time model execution, 24-hour parameter progression charts, offline validation benchmark tables, and decision threshold guides.

---

## 2. Dataset & Feature Engineering

### 2.1 Raw Telemetry Sources
The input feature space is derived from Space Weather Prediction Center (SWPC) / NOAA GOES satellite observations and Solar Dynamics Observatory (SDO) / HMI Space-weather HMI Active Region Patches (SHARP):
- **`xrsa_flux`**: Shortwave X-ray flux (0.05 to 0.4 nm wavelength band, $\text{W/m}^2$).
- **`xrsb_flux`**: Longwave X-ray flux (0.1 to 0.8 nm wavelength band, $\text{W/m}^2$).
- **`proton_gt10mep`**: Energetic proton flux ($E > 10 \text{ MeV}$, pfu).
- **`ratio_ab`**: Shortwave to longwave flux ratio ($\text{XRSA} / \text{XRSB}$).
- **`xrsa_log`**, **`xrsb_log`**: Base-10 logarithmic transforms of raw flux.
- **`xrsb_rate_1h`**, **`xrsb_rate_3h`**: 1-hour and 3-hour temporal rate of change derivatives in flux intensity.
- **`xrsb_max_6h`**, **`xrsa_max_6h`**: Peak flux values over preceding 6-hour rolling windows.
- **`xrsb_std_6h`**: Rolling 6-hour standard deviation capturing magnetic flare turbulence.

### 2.2 Target Formulation
- **Target Variable**: `target_m_plus_24h`
- **Definition**: Binary indicator where `y = 1` denotes that at least one M-class or X-class solar flare occurred within $[T_0, T_0 + 24\text{h}]$, and `y = 0` denotes nominal or background activity ($<\text{M1.0}$).

---

## 3. Strict Anti-Leakage Safeguards

Operational space weather systems require rigorous data boundaries to prevent artificial performance inflation:

1. **Strict Chronological Data Splitting**:
   - Random shuffling (such as standard k-fold cross-validation) is prohibited because time-series observations are auto-correlated across hours.
   - The dataset is partitioned chronologically into:
     - **Training Split (70%)**: Historical baseline period.
     - **Validation Split (15%)**: Intermediate chronological period used for hyperparameter tuning and threshold selection.
     - **Test Split (15%)**: Most recent chronological period held out completely until final model benchmarking.

2. **Leakage-Free Feature Preprocessing**:
   - `StandardScaler` and median imputation parameters are fitted exclusively on the Training Split.
   - Validation and Test splits are transformed using the pre-computed training statistics:
     $$\mu_{\text{train}}, \quad \sigma_{\text{train}}$$
   - Imputation values and scaler parameters are persisted in `models/scaler.joblib`.

3. **Isolated Sliding Window Sequences**:
   - For recurrent models (LSTM), 24-hour sequence windows `[samples, 24, num_features]` are constructed within each split boundary.
   - Sequence generation does not bridge across split boundaries, ensuring zero target or feature contamination.

---

## 4. Model Architectures & Checkpoints

The repository contains pre-trained weights for four distinct architectures:

### 4.1 Classical Tabular Classifiers
Classical models operate on the instantaneous feature vector at observation time $T_0$ (shape: `[1, 11]`):
- **Logistic Regression (`models/logistic_regression.joblib`)**: Linear baseline utilizing L2 regularization and `class_weight='balanced'`.
- **Random Forest (`models/random_forest.joblib`)**: Ensemble of 300 de-correlated decision trees with balanced subsample weighting.
- **XGBoost (`models/xgboost.joblib`)**: Gradient boosted decision tree ensemble with `scale_pos_weight` tuned to the positive class inverse prevalence ratio.

### 4.2 Recurrent Deep Learning Architecture
- **LSTM (`models/lstm.pt`)**:
  - Input: 24-hour historical sequence matrix (shape: `[1, 24, 11]`), capturing temporal gradients, magnetic energy accumulation, and flux surges.
  - Architecture: 2-layer recurrent LSTM network (hidden dimension: 128, dropout: 0.20) followed by a dense linear classification layer.
  - Loss function: Binary Cross-Entropy with Logits, applying positive class weighting to counteract flare rarity.

---

## 5. Frozen Validation Threshold Optimization

### 5.1 Why the 0.50 Threshold Fails
Solar flare events exhibit extreme positive class rarity, with M+ flares occurring in less than 5% of hourly observations. Using a standard default probability threshold of $0.50$ results in near-zero recall, failing to alert space operators to impending geomagnetically effective solar storms.

### 5.2 Optimization Protocol
To establish operational decision boundaries:
1. Each trained model produces continuous probabilities $P(y=1)$ on the independent Validation split.
2. A threshold sweep $\theta \in [0.01, 0.99]$ evaluates the **True Skill Statistic (TSS)** and **F1 Score**:
   $$\text{TSS} = \text{Recall} - \text{FPR} = \frac{\text{TP}}{\text{TP} + \text{FN}} - \frac{\text{FP}}{\text{FP} + \text{TN}}$$
3. The optimal threshold $\theta^*$ is selected to maximize TSS and F1 on the Validation split.
4. **Threshold Freezing**: The selected threshold is permanently frozen and saved in `results/metrics/{model}.json`.
5. The Test split and live operational inferences use this frozen threshold with no post-hoc adjustment.

### 5.3 Active Frozen Thresholds
- **LSTM**: $\theta^* = 0.3300$
- **Logistic Regression**: $\theta^* = 0.8000$
- **Random Forest**: $\theta^* = 0.5800$
- **XGBoost**: $\theta^* = 0.2100$

---

## 6. Offline Test Set Benchmark Matrix

All metrics below represent honest out-of-sample performance on the held-out 15% chronological test split ($N = 634$ observation windows, 69 positive M+ events, 565 negative events):

| Architecture | PR-AUC | TSS | ROC-AUC | F1 Score | Recall | Precision | FAR | Frozen Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM** | **0.3664** | **0.8018** | **0.9096** | **0.5520** | **1.0000** | 0.3812 | 0.6188 | 0.3300 |
| **Random Forest** | 0.3383 | 0.3771 | 0.8923 | 0.3977 | 0.4928 | 0.3333 | 0.6667 | 0.5800 |
| **Logistic Regression** | 0.3488 | 0.2782 | 0.8896 | 0.3399 | 0.3768 | 0.3095 | 0.6905 | 0.8000 |
| **XGBoost** | 0.3069 | 0.4692 | 0.8843 | 0.4352 | 0.6087 | 0.3387 | 0.6613 | 0.2100 |

*Note: The LSTM model attains a True Skill Statistic of 0.8018 and a Recall of 1.0000 on the test set, capturing all positive flare events while maintaining acceptable false alarm ratios.*

---

## 7. Decoupled Service Architecture

### 7.1 Backend API Specification (`backend/`)
- Built on FastAPI with Uvicorn ASGI server.
- Key Endpoints:
  - `GET /api/v1/health`: System health and model availability checks.
  - `GET /api/v1/summary`: Dataset size, parameter list, and time coverage.
  - `GET /api/v1/models`: Available architectures and metadata.
  - `GET /api/v1/metrics`: Offline test-set validation benchmark tables.
  - `GET /api/v1/timestamps`: Valid observation timestamps with complete 24h history.
  - `GET /api/v1/telemetry`: 24-hour sequence of SHARP parameters for time-series plotting.
  - `POST /api/v1/predict`: Real-time model inference using specified model and timestamp.

### 7.2 Frontend Application (`frontend/`)
- Built with React.js (Vite bundler) and Tailwind CSS.
- Slate and zinc neutral color scheme with high-contrast functional status indicators:
  - Emerald green: Nominal status ($<\text{M1.0}$).
  - Amber: Moderate risk or lead validation benchmark.
  - Crimson/Rose: Critical M+ flare hazard alert.
- Zero-latency native SVG line chart rendering for 24-hour telemetry sequences.
- Dynamic environment variable configuration (`VITE_API_BASE_URL`) supporting custom production domains.
