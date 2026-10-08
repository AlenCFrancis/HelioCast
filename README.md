# ☀️ Solar Flare Prediction Using Machine Learning & Deep Learning

An end-to-end time-series machine learning system for predicting whether an **M-class or higher solar flare (M+) will occur within the next 24 hours** using historical solar observations (SDO/HMI SHARP magnetic parameters).

The project compares classical machine learning models (**Logistic Regression**, **Random Forest**, **XGBoost**) against a recurrent deep learning model (**LSTM**) and provides an interactive **Streamlit web application** for real-time predictions.

---

## 📌 Project Architecture

```text
                 SOLAR OBSERVATION DATASET
                       │
                       ▼
              Data Validation & Inspection
                       │
                       ▼
                Preprocessing & Feature Scaling
                       │
                       ▼
             Chronological Train/Val/Test Split
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
   Classical Models           Sequence Builder
   (LR, RF, XGBoost)                │
          │                         ▼
          │                       LSTM
          │                         │
          └────────────┬────────────┘
                       │
                       ▼
        Validation Threshold Selection
                       │
                       ▼
           Test Evaluation & Export
          (Metrics JSON & Plots PNG)
                       │
                       ▼
             Streamlit Application
```

---

## 🚀 Key Features & Anti-Leakage Safeguards

- **Chronological Data Splitting**: Strict zero-leakage split into Train (70%), Validation (15%), and Test (15%). No random shuffling.
- **Leakage-Free Preprocessing**: `StandardScaler` and missing value imputations fitted **exclusively on Training data**.
- **Independent Sequence Windowing**: 3D temporal sequence windows `[samples, 24h, features]` created independently per split to prevent sequence boundary contamination.
- **Validation Threshold Optimization**: Classification threshold tuned on Validation set to maximize F1 / TSS and frozen prior to Test set evaluation.
- **Class Imbalance Handling**: `class_weight='balanced'` in LR & RF, `scale_pos_weight` in XGBoost, and positive class weighting in PyTorch `BCEWithLogitsLoss` for LSTM.

---

## 📁 Directory Structure

```text
solar-flare-prediction/
│
├── app/
│   └── app.py                  # Streamlit Web Application
│
├── configs/
│   └── config.yaml             # Hyperparameters & Project Configuration
│
├── data/
│   ├── raw/
│   └── processed/
│       └── solar_features.csv  # Solar observation dataset
│
├── models/                     # Saved pre-trained models & scaler
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── lstm.pt
│   └── scaler.joblib
│
├── results/
│   ├── metrics/                # Metric reports (JSON)
│   └── plots/                  # ROC, PR curves, Confusion matrices (PNG)
│
├── src/
│   ├── data/
│   │   ├── loader.py           # Data loader & config parser
│   │   ├── validator.py        # Schema inspection & validation
│   │   ├── preprocessing.py    # Leakage-free preprocessor & scaler
│   │   └── sequence_builder.py # Sliding window sequence generator
│   ├── models/
│   │   ├── logistic_regression.py
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   └── lstm.py
│   ├── training/
│   │   ├── classical.py        # Classical models training pipeline
│   │   └── lstm_trainer.py     # LSTM PyTorch training loop
│   └── evaluation/
│       └── metrics.py          # 9 Evaluation metrics & plot routines
│
├── tests/                      # PyTest test suite
│   ├── test_data.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_metrics.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Installation & Setup

1. **Clone repository**:
   ```bash
   git clone <repository-url>
   cd solar-flare-prediction
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   ```
   - Windows: `.venv\Scripts\activate`
   - Linux/macOS: `source .venv/bin/activate`

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🔍 Step 1: Inspect & Validate Dataset

Run the automated dataset inspection module:

```bash
python -m src.data.validator
```

---

## 🤖 Step 2: Train Classical Machine Learning Models

Train Logistic Regression, Random Forest, and XGBoost:

```bash
python -m src.training.classical --model all
```

Or train a specific model:
```bash
python -m src.training.classical --model xgboost
```

---

## 🧠 Step 3: Train Deep Learning Model (LSTM)

Train the PyTorch 2-layer LSTM network with early stopping:

```bash
python -m src.training.lstm_trainer
```

---

## 🌐 Step 4: Run Interactive Streamlit Web Application

Launch the Streamlit web dashboard:

```bash
streamlit run app/app.py
```

The application will load the processed dataset and pre-trained model artifacts from `models/` for instant inference.

---

## 🧪 Step 5: Run Unit Tests

Execute the complete test suite:

```bash
pytest
```

---

## 📊 Evaluation Metrics Computed

- **PR-AUC**: Primary metric for class imbalance
- **ROC-AUC**: Ranking capability across thresholds
- **Precision & Recall**
- **F1 Score**
- **True Skill Statistic (TSS)**: `TPR - FPR`
- **False Alarm Ratio (FAR)**: `FP / (TP + FP)`
- **Brier Score**: Calibrated probability score
- **Confusion Matrix**: `TN, FP, FN, TP`
