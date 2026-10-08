"""
Classical ML Model Training Pipeline for Solar Flare Forecasting.
Trains Logistic Regression, Random Forest, and XGBoost models while ensuring zero data leakage.
"""

import os
import argparse
import numpy as np
from src.data.loader import load_config, load_solar_data
from src.data.generate_dataset import generate_solar_dataset
from src.data.preprocessing import SolarDataPreprocessor
from src.models.logistic_regression import SolarLogisticRegression
from src.models.random_forest import SolarRandomForest
from src.models.xgboost_model import SolarXGBoost
from src.evaluation.metrics import (
    compute_all_metrics,
    find_optimal_threshold,
    save_metrics_report,
    plot_evaluation_curves
)

def train_classical_models(model_type: str = "all", config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    data_path = config["data"]["processed_path"]

    if not os.path.exists(data_path):
        print(f"[Dataset] Processed solar dataset missing at {data_path}. Generating dataset...")
        generate_solar_dataset(output_path=data_path, random_seed=config["project"].get("seed", 42))

    # Load dataset & run zero-leakage preprocessor
    df = load_solar_data(data_path=data_path, config=config)
    preprocessor = SolarDataPreprocessor(config=config)
    X_dict, y_dict, _ = preprocessor.fit_transform(df)

    # Save fitted preprocessor/scaler
    preprocessor.save(model_dir="models")

    X_train, y_train = X_dict["train"], y_dict["train"]
    X_val, y_val = X_dict["val"], y_dict["val"]
    X_test, y_test = X_dict["test"], y_dict["test"]

    print(f"\n[Split Summary]")
    print(f"Train samples : {len(X_train)} (Positive: {np.sum(y_train)})")
    print(f"Val samples   : {len(X_val)} (Positive: {np.sum(y_val)})")
    print(f"Test samples  : {len(X_test)} (Positive: {np.sum(y_test)})\n")

    models_to_train = []
    if model_type == "all":
        models_to_train = ["logistic_regression", "random_forest", "xgboost"]
    else:
        models_to_train = [model_type]

    eval_curves_dict = {}

    for name in models_to_train:
        print("=" * 60)
        print(f" Training Model: {name.upper()}")
        print("=" * 60)

        if name == "logistic_regression":
            model = SolarLogisticRegression(
                max_iter=config["training"].get("max_iter", 1000),
                random_state=config["project"].get("seed", 42)
            )
            model.fit(X_train, y_train)
            save_path = "models/logistic_regression.joblib"

        elif name == "random_forest":
            model = SolarRandomForest(
                n_estimators=300,
                max_depth=15,
                class_weight="balanced",
                random_state=config["project"].get("seed", 42)
            )
            model.fit(X_train, y_train)
            save_path = "models/random_forest.joblib"

        elif name == "xgboost":
            model = SolarXGBoost(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                random_state=config["project"].get("seed", 42)
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            save_path = "models/xgboost.joblib"

        else:
            raise ValueError(f"Unknown model name: {name}")

        # Optimize classification threshold on Validation set ONLY
        val_probs = model.predict_proba(X_val)
        optimal_thresh = find_optimal_threshold(y_val, val_probs, metric_to_maximize="f1")
        print(f"[Threshold Optimization] Selected optimal threshold on Val: {optimal_thresh:.4f}")

        # Freeze threshold and evaluate on Test set ONLY
        test_probs = model.predict_proba(X_test)
        test_metrics = compute_all_metrics(y_test, test_probs, threshold=optimal_thresh)

        print("-" * 45)
        print(f" TEST METRICS for {name.upper()} (Threshold={optimal_thresh:.2f}):")
        print(f"  PR-AUC       : {test_metrics['pr_auc']:.4f}")
        print(f"  ROC-AUC      : {test_metrics['roc_auc']:.4f}")
        print(f"  Precision    : {test_metrics['precision']:.4f}")
        print(f"  Recall       : {test_metrics['recall']:.4f}")
        print(f"  F1 Score     : {test_metrics['f1_score']:.4f}")
        print(f"  TSS          : {test_metrics['tss']:.4f}")
        print(f"  FAR          : {test_metrics['far']:.4f}")
        print(f"  Brier Score  : {test_metrics['brier_score']:.4f}")
        print(f"  Confusion Matrix : {test_metrics['confusion_matrix']}")
        print("-" * 45)

        # Save model & metrics
        model.save(save_path)
        save_metrics_report(test_metrics, model_name=name)
        eval_curves_dict[name.title()] = (y_test, test_probs)

    if len(eval_curves_dict) > 0:
        plot_evaluation_curves(eval_curves_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classical ML models for solar flare forecasting.")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["logistic_regression", "random_forest", "xgboost", "all"],
        help="Model to train (logistic_regression, random_forest, xgboost, or all)"
    )
    args = parser.parse_args()
    train_classical_models(model_type=args.model)
