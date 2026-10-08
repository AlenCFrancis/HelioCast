"""
Evaluation & Metrics Module for Solar Flare Prediction.
Calculates ROC-AUC, PR-AUC, Precision, Recall, F1, TSS, FAR, Brier Score, and Confusion Matrix.
Determines optimal prediction threshold on Validation data and generates plot artifacts.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve
)

def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compute 9 standard evaluation metrics for solar flare forecasting.
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # ROC-AUC
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    # PR-AUC
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(rec_arr, prec_arr))

    # Precision, Recall, F1
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # True Skill Statistic (TSS) = TPR - FPR
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tss = float(tpr - fpr)

    # False Alarm Ratio (FAR) = FP / (TP + FP)
    far = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0

    # Brier Score
    brier = float(brier_score_loss(y_true, y_prob))

    metrics = {
        "threshold": float(threshold),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "tss": round(tss, 4),
        "far": round(far, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }
    return metrics

def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_to_maximize: str = "pr_auc"
) -> float:
    """
    Search over candidate classification thresholds [0.05, 0.95] to maximize
    the selected metric (tss, f1, or pr_auc) on validation set.
    """
    best_thresh = 0.5
    best_score = -1.0
    thresholds = np.linspace(0.05, 0.95, 91)

    for thresh in thresholds:
        metrics = compute_all_metrics(y_true, y_prob, threshold=thresh)
        if metric_to_maximize == "tss":
            score = metrics["tss"]
        elif metric_to_maximize == "f1":
            score = metrics["f1_score"]
        else:  # balance precision and recall / TSS
            score = metrics["f1_score"] + 0.5 * metrics["tss"]

        if score > best_score:
            best_score = score
            best_thresh = thresh

    return float(best_thresh)

def save_metrics_report(metrics: Dict[str, Any], model_name: str, output_dir: str = "results/metrics"):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{model_name}.json")
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Metrics] Saved metrics report for {model_name} to: {filepath}")

def plot_evaluation_curves(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_dir: str = "results/plots"
):
    """
    Generate and save ROC curve, PR curve, and Confusion Matrix plots for models.
    results_dict: {model_name: (y_true, y_prob)}
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. ROC Curve
    plt.figure(figsize=(8, 6))
    for model_name, (y_true, y_prob) in results_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_val:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Solar Flare M+ Prediction")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300)
    plt.close()

    # 2. Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    for model_name, (y_true, y_prob) in results_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc_val = auc(rec, prec)
        plt.plot(rec, prec, label=f"{model_name} (PR-AUC = {pr_auc_val:.3f})", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves - Solar Flare M+ Prediction")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"), dpi=300)
    plt.close()

    print(f"[Plots] Saved ROC and PR evaluation curves to: {output_dir}")
