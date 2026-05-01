#!/usr/bin/env python3
import argparse
import copy
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.algorithms.mlp_pvc import (
    FEATURE_KEYS,
    DS1_RECORDS,
    DS2_RECORDS,
    PVCCandidateMLP,
    impute_nan_with_medians,
    normalize_features,
)


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError("PyTorch is required to train the MLP. Install torch first.") from exc
    return torch, DataLoader, TensorDataset


def _compute_medians(X):
    medians = np.nanmedian(X, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    return medians.astype(np.float32)


def _compute_mean_std(X):
    means = np.mean(X, axis=0).astype(np.float32)
    stds = np.std(X, axis=0).astype(np.float32)
    stds = np.where(stds <= 1e-9, 1.0, stds)
    return means, stds


def _binary_metrics(y_true, probs, threshold=0.5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = (np.asarray(probs, dtype=np.float32) >= threshold).astype(np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    f1 = (2.0 * precision * sensitivity) / max(1e-9, (precision + sensitivity))
    return {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _evaluate(model, X, y, torch, device):
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X).to(device)
        logits = model(xt)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    return _binary_metrics(y, probs), probs


def train_mlp(
    dataset_file,
    model_output,
    scaler_output,
    epochs=100,
    batch_size=256,
    patience=10,
    learning_rate=1e-3,
    seed=42,
):
    torch, DataLoader, TensorDataset = _require_torch()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    data = np.load(dataset_file, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    record_ids = np.asarray(data["record_ids"], dtype=np.int32)

    # Filter for DS1 and DS2 records using de Chazal split
    ds1_mask = np.isin(record_ids, list(DS1_RECORDS))
    ds2_mask = np.isin(record_ids, list(DS2_RECORDS))
    
    if not np.any(ds1_mask):
        raise ValueError("No DS1 records found in dataset for training.")
    if not np.any(ds2_mask):
        raise ValueError("No DS2 records found in dataset for testing.")

    X_ds1, y_ds1 = X[ds1_mask], y[ds1_mask]
    X_test, y_test = X[ds2_mask], y[ds2_mask]

    idx = np.arange(X_ds1.shape[0])
    rng.shuffle(idx)
    split = max(1, int(0.8 * len(idx)))
    split = min(split, len(idx) - 1)
    train_idx = idx[:split]
    val_idx = idx[split:]

    X_train, y_train = X_ds1[train_idx], y_ds1[train_idx]
    X_val, y_val = X_ds1[val_idx], y_ds1[val_idx]

    medians = _compute_medians(X_train)
    X_train_imp = impute_nan_with_medians(X_train, medians)
    X_val_imp = impute_nan_with_medians(X_val, medians)
    X_test_imp = impute_nan_with_medians(X_test, medians)

    means, stds = _compute_mean_std(X_train_imp)
    X_train_n = normalize_features(X_train_imp, means, stds)
    X_val_n = normalize_features(X_val_imp, means, stds)
    X_test_n = normalize_features(X_test_imp, means, stds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PVCCandidateMLP(input_dim=X_train_n.shape[1]).to(device)
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()  
    pos_weight = torch.tensor([pos_weight]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_ds = TensorDataset(
        torch.from_numpy(X_train_n.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)).view(-1, 1),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * xb.shape[0]

        train_loss = running_loss / max(1, len(train_ds))

        model.eval()
        with torch.no_grad():
            x_val_t = torch.from_numpy(X_val_n.astype(np.float32)).to(device)
            y_val_t = torch.from_numpy(y_val.astype(np.float32)).view(-1, 1).to(device)
            val_probs = model(x_val_t)
            val_probs = torch.sigmoid(val_probs)
            val_loss = float(criterion(val_probs, y_val_t).item())

        val_metrics = _binary_metrics(y_val, val_probs.detach().cpu().numpy().reshape(-1))
        print(
            f"[EPOCH {epoch:03d}] train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_sens={val_metrics['sensitivity']:.4f} "
            f"val_spec={val_metrics['specificity']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    val_metrics, _ = _evaluate(model, X_val_n, y_val, torch, device)
    test_metrics, _ = _evaluate(model, X_test_n, y_test, torch, device)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": X_train_n.shape[1],
            "feature_keys": list(FEATURE_KEYS),
        },
        model_output,
    )
    np.savez_compressed(
        scaler_output,
        medians=medians.astype(np.float32),
        means=means.astype(np.float32),
        stds=stds.astype(np.float32),
        feature_keys=np.asarray(FEATURE_KEYS, dtype=object),
    )

    print(f"[INFO] Saved model to {model_output}")
    print(f"[INFO] Saved scaler params to {scaler_output}")
    print(
        f"[VAL] acc={val_metrics['accuracy']:.4f} sens={val_metrics['sensitivity']:.4f} "
        f"spec={val_metrics['specificity']:.4f} f1={val_metrics['f1']:.4f}"
    )
    print(
        f"[TEST] acc={test_metrics['accuracy']:.4f} sens={test_metrics['sensitivity']:.4f} "
        f"spec={test_metrics['specificity']:.4f} f1={test_metrics['f1']:.4f} "
        f"tp={test_metrics['tp']} fp={test_metrics['fp']} fn={test_metrics['fn']} tn={test_metrics['tn']}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP baseline for PVC detection")
    parser.add_argument("--dataset", default="mlp_pvc_dataset.npz", help="Input dataset NPZ")
    parser.add_argument("--model-output", default="mlp_pvc_model.pt", help="Output model file")
    parser.add_argument("--scaler-output", default="mlp_scaler_params.npz", help="Output scaler NPZ")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_mlp(
        dataset_file=args.dataset,
        model_output=args.model_output,
        scaler_output=args.scaler_output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        learning_rate=args.lr,
        seed=args.seed,
    )
