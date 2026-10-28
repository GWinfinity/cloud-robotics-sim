"""Train the trajclass GRU on an exported episode .npz and write a report.

Companion to ``tools/gen_traj_classifier_data.py`` — the same three commands
run unchanged on a cloud worker, which is the point of the probe: the
expensive half is episode generation (GPU worker), this half is minutes on
CPU.

Example:
    python tools/train_traj_classifier.py \
        --data outputs/trajclass/pickcube_probe.npz \
        --out outputs/trajclass/run_001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cloud_robotics_sim.trajclass import (
    TrainConfig,
    load_npz,
    train_classifier,
)


def main() -> None:
    """CLI entry: train on an episode .npz, write report.json + checkpoint.pt."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="episode .npz")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    batch = load_npz(args.data)
    cfg = TrainConfig(
        window=args.window,
        stride=args.stride,
        epochs=args.epochs,
        hidden=args.hidden,
        layers=args.layers,
        seed=args.seed,
    )
    report, model = train_classifier(batch, cfg)

    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_channels": model.gru.input_size,
            "num_classes": model.head.out_features,
            "hidden": model.gru.hidden_size,
            "num_layers": model.gru.num_layers,
            "standardization": report["standardization"],
        },
        args.out / "checkpoint.pt",
    )
    print(
        f"val acc={report['val']['accuracy']} "
        f"macro_f1={report['val']['macro_f1']} ece={report['val']['ece']} "
        f"params={report['parameters']}"
    )
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
