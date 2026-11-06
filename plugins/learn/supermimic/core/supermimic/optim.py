"""Hybrid Muon/Adam optimizer for hidden matrices and remaining parameters."""

from __future__ import annotations

import torch
import torch.nn as nn


def _linear_modules(module: nn.Module) -> list[nn.Linear]:
    return [child for child in module.modules() if isinstance(child, nn.Linear)]


class MuonAdam:
    """Use Muon on hidden weight matrices of the given MLPs and Adam elsewhere."""

    def __init__(
        self,
        model: nn.Module,
        mlps: list[nn.Module],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        fused: bool = True,
    ):
        hidden_weights = {
            id(linear.weight) for mlp in mlps for linear in _linear_modules(mlp)[1:-1]
        }
        muon_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) in hidden_weights
        ]
        adam_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in hidden_weights and parameter.requires_grad
        ]
        self.muon = torch.optim.Muon(
            muon_parameters,
            lr=lr,
            weight_decay=0.0,
            adjust_lr_fn="match_rms_adamw",
        )
        self.adam = torch.optim.Adam(
            adam_parameters,
            lr=lr,
            betas=betas,
            fused=fused,
        )

    @property
    def param_groups(self) -> list[dict]:
        return self.muon.param_groups + self.adam.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        self.muon.step(closure=closure)
        self.adam.step(closure=closure)

    def state_dict(self) -> dict:
        return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.muon.load_state_dict(state["muon"])
        self.adam.load_state_dict(state["adam"])


__all__ = ["MuonAdam"]
