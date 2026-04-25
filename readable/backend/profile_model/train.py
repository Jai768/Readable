"""
train.py
========
Full training pipeline for DyslexiaProfiler.

Usage
-----
# Basic
python train.py --data data/dataset.csv --out checkpoints/

# Full options
python train.py \
    --data data/dataset.csv \
    --out checkpoints/ \
    --epochs 300 \
    --batch 64 \
    --lr 1e-3 \
    --dropout 0.2 \
    --patience 30 \
    --val_split 0.15 \
    --test_split 0.10 \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from model import DyslexiaProfiler, WeightedProfileLoss, FEATURE_NAMES, OUTPUT_NAMES
from data_generator import csv_to_samples, Sample


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("[GPU] Apple MPS")
    else:
        dev = torch.device("cpu")
        print("[CPU] No GPU found — training on CPU")
    return dev


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def samples_to_tensors(samples: List[Sample]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Sample list → (X, Y) numpy arrays."""
    from dataclasses import asdict

    X = np.array([
        [getattr(s.features, f) for f in FEATURE_NAMES]
        for s in samples
    ], dtype=np.float32)

    Y = np.array([
        [getattr(s.labels, o) for o in OUTPUT_NAMES]
        for s in samples
    ], dtype=np.float32)

    return X, Y


def build_dataloaders(
    samples: List[Sample],
    val_split: float = 0.15,
    test_split: float = 0.10,
    batch_size: int = 64,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Split samples → train/val/test DataLoaders.
    Fits StandardScaler on train X only.
    """
    set_seed(seed)

    X, Y = samples_to_tensors(samples)
    n = len(X)
    n_test = max(1, int(n * test_split))
    n_val  = max(1, int(n * val_split))
    n_train = n - n_val - n_test

    assert n_train > 0, "Not enough samples for train/val/test split"

    # Shuffle indices
    idx = np.random.permutation(n)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val   = scaler.transform(X[val_idx])
    X_test  = scaler.transform(X[test_idx])

    def make_loader(x, y, shuffle=False):
        ds = TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                         num_workers=0, pin_memory=False)

    return (
        make_loader(X_train, Y[train_idx], shuffle=True),
        make_loader(X_val,   Y[val_idx]),
        make_loader(X_test,  Y[test_idx]),
        scaler,
    )


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    preds: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Compute MAE, RMSE, R² per output head."""
    metrics = {}
    for name in OUTPUT_NAMES:
        p = preds[name]
        t = targets[name]
        mae = mean_absolute_error(t, p)
        rmse = np.sqrt(np.mean((p - t) ** 2))
        r2 = r2_score(t, p)
        metrics[name] = {"mae": mae, "rmse": rmse, "r2": r2}
    return metrics


def collect_predictions(
    model: DyslexiaProfiler,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Run model over entire loader, collect preds and targets."""
    model.eval()
    all_preds  = {n: [] for n in OUTPUT_NAMES}
    all_targets = {n: [] for n in OUTPUT_NAMES}

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            out = model(x_batch)
            for i, name in enumerate(OUTPUT_NAMES):
                all_preds[name].append(out[name].cpu().numpy())
                all_targets[name].append(y_batch[:, i].numpy())

    return (
        {n: np.concatenate(v) for n, v in all_preds.items()},
        {n: np.concatenate(v) for n, v in all_targets.items()},
    )


def print_metrics_table(metrics: Dict[str, Dict[str, float]], title: str = ""):
    pad = max(len(n) for n in OUTPUT_NAMES) + 2
    header = f"{'Output':<{pad}}  {'MAE':>7}  {'RMSE':>7}  {'R²':>7}"
    sep = "─" * len(header)
    print(f"\n{'── ' + title + ' ──' if title else sep}")
    print(header)
    print(sep)
    for name, m in metrics.items():
        sign = "✓" if m["r2"] > 0.7 else ("~" if m["r2"] > 0.4 else "✗")
        print(f"{name:<{pad}}  {m['mae']:>7.4f}  {m['rmse']:>7.4f}  "
              f"{m['r2']:>7.4f}  {sign}")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# LEARNING RATE SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, epochs: int, warmup: int = 10):
    """Cosine annealing with linear warmup."""

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 30, delta: float = 1e-5):
        self.patience = patience
        self.delta = delta
        self.best = np.inf
        self.counter = 0
        self.best_state: Optional[dict] = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best - self.delta:
            self.best = val_loss
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module):
        if self.best_state:
            model.load_state_dict(self.best_state)


def train_epoch(
    model: DyslexiaProfiler,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: WeightedProfileLoss,
    device: torch.device,
    clip_grad: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    per_loss_accum = {n: 0.0 for n in OUTPUT_NAMES}
    n_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        targets = {n: y_batch[:, i] for i, n in enumerate(OUTPUT_NAMES)}

        optimizer.zero_grad()
        preds = model(x_batch)
        loss, per_loss = criterion(preds, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item()
        for n in OUTPUT_NAMES:
            per_loss_accum[n] += per_loss[n]
        n_batches += 1

    avg_per = {n: v / n_batches for n, v in per_loss_accum.items()}
    return total_loss / n_batches, avg_per


@torch.no_grad()
def val_epoch(
    model: DyslexiaProfiler,
    loader: DataLoader,
    criterion: WeightedProfileLoss,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    per_loss_accum = {n: 0.0 for n in OUTPUT_NAMES}
    n_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        targets = {n: y_batch[:, i] for i, n in enumerate(OUTPUT_NAMES)}
        preds = model(x_batch)
        loss, per_loss = criterion(preds, targets)

        total_loss += loss.item()
        for n in OUTPUT_NAMES:
            per_loss_accum[n] += per_loss[n]
        n_batches += 1

    avg_per = {n: v / n_batches for n, v in per_loss_accum.items()}
    return total_loss / n_batches, avg_per


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINER
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    set_seed(args.seed)
    device = get_device()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading data from {args.data} ...")
    samples = csv_to_samples(args.data)
    print(f"  Total samples : {len(samples)}")
    n_dys = sum(s.is_dyslexic for s in samples)
    print(f"  Dyslexic      : {n_dys} ({100*n_dys/len(samples):.1f}%)")
    print(f"  Typical       : {len(samples) - n_dys}")

    # ── Build data loaders ─────────────────────────────────────────────────
    print("\n[2/5] Building train/val/test splits ...")
    train_loader, val_loader, test_loader, scaler = build_dataloaders(
        samples,
        val_split=args.val_split,
        test_split=args.test_split,
        batch_size=args.batch,
        seed=args.seed,
    )
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}")

    # Save scaler parameters for inference
    scaler_params = {
        "mean_": scaler.mean_.tolist(),
        "scale_": scaler.scale_.tolist(),
        "feature_names": FEATURE_NAMES,
    }
    with open(out_dir / "scaler.json", "w") as f:
        json.dump(scaler_params, f, indent=2)

    # ── Build model ────────────────────────────────────────────────────────
    print("\n[3/5] Building model ...")
    model = DyslexiaProfiler(
        eye_dim=args.eye_dim,
        speech_dim=args.speech_dim,
        acoustic_dim=args.acoustic_dim,
        dropout=args.dropout,
    ).to(device)
    print(f"  Parameters : {model.count_params():,}")

    criterion = WeightedProfileLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.epochs, warmup=10)
    early_stop = EarlyStopping(patience=args.patience)

    # ── Train ──────────────────────────────────────────────────────────────
    print(f"\n[4/5] Training for up to {args.epochs} epochs ...")
    print(f"  LR={args.lr}, Batch={args.batch}, Dropout={args.dropout}")
    print(f"  Early stopping patience: {args.patience} epochs\n")

    history = {"train_loss": [], "val_loss": [], "lr": []}
    t0 = time.time()
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_per = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_per = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(scheduler.get_last_lr()[0])

        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch:4d}/{args.epochs}  "
                f"train={train_loss:.5f}  val={val_loss:.5f}  "
                f"lr={lr_now:.2e}  [{elapsed:.0f}s]"
            )

        # Early stopping check
        if early_stop(val_loss, model):
            print(f"\n  ⏸ Early stopping at epoch {epoch} "
                  f"(best val={early_stop.best:.5f} at epoch {epoch - args.patience})")
            best_epoch = epoch - args.patience
            break
        else:
            best_epoch = epoch

    early_stop.restore(model)
    print(f"\n  Best epoch: {best_epoch}  Best val loss: {early_stop.best:.5f}")

    # Save training history
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Evaluate on test set ───────────────────────────────────────────────
    print("\n[5/5] Evaluating on held-out test set ...")
    preds, targets = collect_predictions(model, test_loader, device)
    metrics = compute_metrics(preds, targets)
    print_metrics_table(metrics, title="Test Set Results")

    # Save metrics
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save model
    model_path = str(out_dir / "dyslexia_profiler.pt")
    model.save(model_path, scaler_params=scaler_params)

    total_time = time.time() - t0
    print(f"\n[✓] Training complete in {total_time:.0f}s")
    print(f"    Model     → {model_path}")
    print(f"    Scaler    → {out_dir / 'scaler.json'}")
    print(f"    History   → {out_dir / 'history.json'}")
    print(f"    Metrics   → {out_dir / 'test_metrics.json'}")

    return model, scaler_params, history, metrics


# ─────────────────────────────────────────────────────────────────────────────
# LEARNING CURVE PLOT (optional — requires matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def plot_history(history: dict, out_path: str):
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(history["train_loss"], label="Train Loss", color="royalblue")
        axes[0].plot(history["val_loss"],   label="Val Loss",   color="tomato")
        axes[0].set_title("Loss Curves")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Weighted MSE Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(history["lr"], color="forestgreen")
        axes[1].set_title("Learning Rate Schedule")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("LR")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[✓] Training curves → {out_path}")
    except ImportError:
        print("[!] matplotlib not installed — skipping plot")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train DyslexiaProfiler")

    # Data
    p.add_argument("--data",       type=str, default="data/dataset.csv")
    p.add_argument("--out",        type=str, default="checkpoints/")
    p.add_argument("--val_split",  type=float, default=0.15)
    p.add_argument("--test_split", type=float, default=0.10)

    # Training
    p.add_argument("--epochs",     type=int,   default=300)
    p.add_argument("--batch",      type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience",   type=int,   default=40)
    p.add_argument("--seed",       type=int,   default=42)

    # Model architecture
    p.add_argument("--eye_dim",      type=int,   default=32)
    p.add_argument("--speech_dim",   type=int,   default=32)
    p.add_argument("--acoustic_dim", type=int,   default=16)
    p.add_argument("--dropout",      type=float, default=0.2)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, scaler_params, history, metrics = train(args)
    plot_history(history, str(Path(args.out) / "training_curves.png"))