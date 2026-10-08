"""
1D CNN Deep Learning Training Pipeline for Solar Flare Forecasting.
Constructs 3D temporal sequence windows, trains PyTorch 1D CNN model with early stopping,
optimizes prediction threshold on validation set, and evaluates on test set.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from src.data.loader import load_config, load_solar_data
from src.data.generate_dataset import generate_solar_dataset
from src.data.preprocessing import SolarDataPreprocessor
from src.data.sequence_builder import create_sequences
from src.models.cnn_1d import SolarCNN1D, SolarCNN1DWrapper
from src.evaluation.metrics import (
    compute_all_metrics,
    find_optimal_threshold,
    save_metrics_report,
    plot_evaluation_curves
)


def train_cnn_pipeline(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    data_path = config["data"]["processed_path"]

    if not os.path.exists(data_path):
        print(f"[Dataset] Processed dataset missing at {data_path}. Generating dataset...")
        generate_solar_dataset(output_path=data_path, random_seed=config["project"].get("seed", 42))

    # Seed reproducibility
    seed = config["project"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Load data & fit preprocessor
    df = load_solar_data(data_path=data_path, config=config)
    preprocessor = SolarDataPreprocessor(config=config)
    X_dict, y_dict, _ = preprocessor.fit_transform(df)
    preprocessor.save(model_dir="models")

    # Determine sequence length from history_hours
    seq_length = config["forecast"].get("history_hours", 24)

    # 2. Build 3D sequences INDEPENDENTLY per split (No cross-split leakage)
    X_train_seq, y_train_seq = create_sequences(X_dict["train"], y_dict["train"], seq_length=seq_length)
    X_val_seq, y_val_seq = create_sequences(X_dict["val"], y_dict["val"], seq_length=seq_length)
    X_test_seq, y_test_seq = create_sequences(X_dict["test"], y_dict["test"], seq_length=seq_length)

    print(f"\n[CNN 1D Sequence Shapes]")
    print(f"Train sequence tensor : {X_train_seq.shape} (Positive target: {int(np.sum(y_train_seq))})")
    print(f"Val sequence tensor   : {X_val_seq.shape} (Positive target: {int(np.sum(y_val_seq))})")
    print(f"Test sequence tensor  : {X_test_seq.shape} (Positive target: {int(np.sum(y_test_seq))})\n")

    # Hardware acceleration check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hardware] Training CNN 1D model on device: {device}")

    # PyTorch DataLoaders
    batch_size = config["training"].get("batch_size", 32)
    train_dataset = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32), torch.tensor(y_train_seq, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val_seq, dtype=torch.float32), torch.tensor(y_val_seq, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_features = X_train_seq.shape[2]

    # CNN-specific hyperparameters from config (with sensible defaults)
    cnn_config = config.get("cnn", {})
    channels = cnn_config.get("channels", [32, 64])
    kernel_size = cnn_config.get("kernel_size", 3)
    dropout = cnn_config.get("dropout", 0.2)

    # Training hyperparameters — CNN uses higher LR than LSTM (converges faster)
    lr = 1e-3
    weight_decay = config["training"].get("weight_decay", 0.0001)
    epochs = config["training"].get("epochs", 50)
    patience = config["training"].get("patience", 7)

    model = SolarCNN1D(
        input_size=num_features,
        channels=channels,
        kernel_size=kernel_size,
        dropout=dropout
    ).to(device)

    # Class weighting for positive loss term
    num_neg = np.sum(y_train_seq == 0)
    num_pos = np.sum(y_train_seq == 1)
    pos_weight = torch.tensor([num_neg / max(num_pos, 1.0)], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Early stopping loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_weights = None

    print("=" * 60)
    print(" TRAINING 1D CNN MODEL")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_b.size(0)

        train_loss /= len(train_dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model(X_b)
                loss = criterion(logits, y_b)
                val_loss += loss.item() * X_b.size(0)
        val_loss /= len(val_dataset)

        if epoch % 5 == 0 or epoch == 1 or val_loss < best_val_loss:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_weights = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Early Stopping] Triggered at epoch {epoch} (Best Val Loss: {best_val_loss:.4f})")
                break

    # Load best weights
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # Package in wrapper
    wrapper = SolarCNN1DWrapper(
        input_size=num_features,
        channels=channels,
        kernel_size=kernel_size,
        dropout=dropout,
        device=device
    )
    wrapper.model = model

    # Optimize threshold on Validation set
    val_probs = wrapper.predict_proba(X_val_seq)
    optimal_thresh = find_optimal_threshold(y_val_seq, val_probs, metric_to_maximize="f1")
    print(f"[Threshold Optimization] Selected optimal threshold on Val: {optimal_thresh:.4f}")

    # Freeze threshold & evaluate on Test set
    test_probs = wrapper.predict_proba(X_test_seq)
    test_metrics = compute_all_metrics(y_test_seq, test_probs, threshold=optimal_thresh)

    print("-" * 45)
    print(f" TEST METRICS for CNN 1D (Threshold={optimal_thresh:.2f}):")
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

    # Save CNN model state
    save_path = "models/cnn_1d.pt"
    wrapper.save(save_path)
    save_metrics_report(test_metrics, model_name="cnn_1d")

    # Update plot curves
    plot_evaluation_curves({"CNN_1D": (y_test_seq, test_probs)})


if __name__ == "__main__":
    train_cnn_pipeline()
