"""Train the tiny GRU classifier on windowed episodes.

Episode-disjoint train/val split (windows from one episode never straddle
the split), standardization fitted on the train split only, and a compact
report dict (accuracy / per-class / macro-F1 / ECE / confusion) so callers
can log it straight into the data pipeline.

CPU-first: the model is ~100k parameters; a few thousand windows converge
in well under a minute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cloud_robotics_sim.trajclass.dataset import (
    LABEL_NAMES,
    EpisodeBatch,
    batch_to_json,
    windows,
)
from cloud_robotics_sim.trajclass.model import TrajGRU, count_parameters


@dataclass
class TrainConfig:
    """Knobs for the probe trainer (defaults sized for CPU smoke runs)."""

    window: int = 25
    stride: int = 5
    hidden: int = 64
    layers: int = 2
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    val_frac: float = 0.3
    seed: int = 0
    min_train_windows: int = 4


def _standardize(
    x_tr: np.ndarray, x_va: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_tr.reshape(-1, x_tr.shape[-1]).mean(axis=0)
    std = x_tr.reshape(-1, x_tr.shape[-1]).std(axis=0) + 1e-6
    return (
        ((x_tr - mean) / std).astype(np.float32),
        ((x_va - mean) / std).astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def _expected_calibration_error(
    probs: np.ndarray, targets: np.ndarray, n_bins: int = 10
) -> float:
    """Unreliability of the reported probabilities (lower is better)."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    ece = 0.0
    n = len(conf)
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (conf >= lo) & (conf < hi if b < n_bins - 1 else conf <= hi)
        if mask.any():
            acc = float((pred[mask] == targets[mask]).mean())
            ece += float((conf[mask].mean() - acc) * mask.sum() / n)
    return ece


def _classification_report(probs: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    pred = probs.argmax(axis=1)
    n_classes = probs.shape[1]
    per_class: dict[str, float] = {}
    f1s: list[float] = []
    for c in range(n_classes):
        tp = int(((pred == c) & (targets == c)).sum())
        fp = int(((pred == c) & (targets != c)).sum())
        fn = int(((pred != c) & (targets == c)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_class[LABEL_NAMES[c]] = round(recall, 4)
        f1s.append(f1)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(targets, pred):
        confusion[int(t), int(p)] += 1
    return {
        "accuracy": round(float((pred == targets).mean()), 4),
        "macro_f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
        "ece": round(_expected_calibration_error(probs, targets), 4),
        "per_class_recall": per_class,
        "confusion": confusion.tolist(),
    }


def train_classifier(
    batch: EpisodeBatch, cfg: TrainConfig | None = None
) -> tuple[dict[str, Any], TrajGRU]:
    """Fit :class:`TrajGRU` on the batch and return ``(report, model)``.

    The report is JSON-serializable and carries everything needed to compare
    runs (data provenance, split sizes, metrics, standardization stats for
    checkpoint reuse at inference).
    """
    cfg = cfg or TrainConfig()
    rng = np.random.default_rng(cfg.seed)
    # Stratified episode-disjoint split: an unstratified shuffle can put a
    # minority class entirely on one side, and windows must never straddle
    # the split (they would leak the val answer into training).
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for lab in np.unique(batch.labels):
        group = np.where(batch.labels == lab)[0]
        group = rng.permutation(group)
        n_val = max(1, int(round(len(group) * cfg.val_frac)))
        if len(group) == 1:
            n_val = 1
        val_parts.append(group[:n_val])
        train_parts.append(group[n_val:])
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    if len(train_idx) < 1:
        raise ValueError(f"need >= 2 episodes for a split, got {batch.num_episodes}")

    x_tr, y_tr = windows(batch, cfg.window, cfg.stride, train_idx)
    x_va, y_va = windows(batch, cfg.window, cfg.stride, val_idx)
    if len(x_tr) < cfg.min_train_windows or len(x_va) == 0:
        raise ValueError(
            f"too few windows (train={len(x_tr)}, val={len(x_va)}); "
            f"lower window/stride or generate more episodes"
        )

    x_tr, x_va, mean, std = _standardize(x_tr, x_va)

    torch.manual_seed(cfg.seed)
    model = TrajGRU(
        in_channels=batch.channels,
        num_classes=len(LABEL_NAMES),
        hidden=cfg.hidden,
        layers=cfg.layers,
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr).long()),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(x_va)), dim=-1).cpu().numpy()

    report: dict[str, Any] = {
        "data": batch_to_json(batch),
        "config": {
            "window": cfg.window,
            "stride": cfg.stride,
            "hidden": cfg.hidden,
            "layers": cfg.layers,
            "epochs": cfg.epochs,
            "val_frac": cfg.val_frac,
            "seed": cfg.seed,
        },
        "split": {
            "train_episodes": len(train_idx),
            "val_episodes": len(val_idx),
            "train_windows": int(len(x_tr)),
            "val_windows": int(len(x_va)),
        },
        "parameters": count_parameters(model),
        "val": _classification_report(probs, y_va),
        "standardization": {"mean": mean.tolist(), "std": std.tolist()},
    }
    return report, model
