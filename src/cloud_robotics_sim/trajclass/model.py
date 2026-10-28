"""Tiny GRU over per-step ``obs‖act`` channels.

Sized for the trajectory-judgment probe: ~100k parameters, microsecond CPU
inference, minutes-to-train. A temporal model is the right model class here
because the input is a dense numeric sequence — the thing a text-triage
encoder (laya) structurally cannot see.
"""

from __future__ import annotations

import torch
from torch import nn


class TrajGRU(nn.Module):
    """GRU encoder over window steps + linear class head on the last hidden."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden: int = 64,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, W, C)`` windows -> ``(N, num_classes)`` logits."""
        out, _ = self.gru(x)
        last: torch.Tensor = out[:, -1, :]
        logits: torch.Tensor = self.head(last)
        return logits


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count (for the report)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
