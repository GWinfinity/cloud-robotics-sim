"""Minimal world-model training smoke test on pre-generated SDG data.

Closes the vr_bridge/dreamdojo data loop: takes a dreamdojo-layout HDF5
(written by teleop recording or ``replay_sdg.py``), serves it through
``GenesisDataset``, and trains a tiny action-conditioned one-frame
predictor (frame_t + action_t -> frame_{t+1}) for a few optimizer steps.

This is a data-usability smoke test, not a real trainer: it proves the
record -> synthesize -> train pipeline end-to-end (shapes, dtypes,
normalization, gradient flow) on CPU with a two-episode dataset.

Usage:
    python train_smoke.py --data outputs/replay_sdg_e2e/sdg_dataset.h5 \
        --steps 20 --batch 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

_PLUGIN = Path(__file__).resolve().parents[1]  # plugins/datasets/dreamdojo
for _p in (str(_PLUGIN.parents[0]), str(_PLUGIN)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dreamdojo.core.dataset import GenesisDataset  # noqa: E402


class OneStepPredictor(nn.Module):
    """Tiny action-conditioned frame predictor.

    Encoder: 2x stride-2 convs. Action embedding added channel-wise at the
    bottleneck. Decoder: 2x transposed convs back to full resolution.
    """

    def __init__(self, action_dim: int, base_channels: int = 16) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = nn.Conv2d(3, c, 3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(c, c * 2, 3, stride=2, padding=1)
        self.act_proj = nn.Linear(action_dim, c * 2)
        self.dec2 = nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose2d(c, 3, 4, stride=2, padding=1)

    def forward(self, frame: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict the next frame from (B,3,H,W) and (B,D)."""
        h = F.relu(self.enc1(frame))
        h = F.relu(self.enc2(h))
        h = h + self.act_proj(action)[:, :, None, None]
        h = F.relu(self.dec2(h))
        return self.dec1(h)


def make_sample_pairs(
    sample: dict, downscale: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Turn one GenesisDataset sample into (frames_t, actions_t, frames_t1).

    Frames are float in [0, 1], shape (T-1, C, H, W); optionally avg-pooled
    by ``downscale`` to keep CPU smoke runs cheap.
    """
    video = sample["video"].float()  # (T, C, H, W) already in [0, 1]
    action = sample["action"].float()  # (T, D)
    if downscale > 1:
        video = F.avg_pool2d(video, kernel_size=downscale)
    frames_t = video[:-1]
    frames_t1 = video[1:]
    actions_t = action[:-1]
    return frames_t, actions_t, frames_t1


def train_steps(
    dataset: GenesisDataset,
    model: nn.Module,
    *,
    steps: int = 20,
    batch: int = 4,
    downscale: int = 1,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
) -> list[float]:
    """Run ``steps`` SGD steps over random dataset samples; return losses."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    for _ in range(steps):
        idx = int(
            torch.randint(0, len(dataset.pre_generated_data), (1,), generator=gen)
        )
        frames_t, actions_t, frames_t1 = make_sample_pairs(
            dataset[idx], downscale=downscale
        )
        # Random time windows as the batch.
        perm = torch.randperm(len(frames_t), generator=gen)[:batch]
        if len(perm) == 0:
            raise ValueError("sample has too few frames for one training step")
        x = frames_t[perm].to(device)
        a = actions_t[perm].to(device)
        y = frames_t1[perm].to(device)

        opt.zero_grad()
        pred = model(x, a)
        loss = F.mse_loss(pred, y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="dreamdojo HDF5")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--downscale", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ds = GenesisDataset(
        pre_generated_path=str(args.data),
        num_frames=16,
        robot_type="franka",
        device="cpu",
    )
    n_dofs = int(ds[0]["action"].shape[-1])
    model = OneStepPredictor(action_dim=n_dofs)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[train_smoke] {len(ds.pre_generated_data)} episodes, "
        f"action_dim={n_dofs}, params={n_params}"
    )

    losses = train_steps(
        ds,
        model,
        steps=args.steps,
        batch=args.batch,
        downscale=args.downscale,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )
    first, last = losses[0], losses[-1]
    print(
        f"[train_smoke] loss: first={first:.5f} last={last:.5f} "
        f"min={min(losses):.5f}"
    )
    ok = bool(np.all(np.isfinite(losses)))
    print(f"[train_smoke] {'OK' if ok else 'FAILED: non-finite loss'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
