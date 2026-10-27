"""ONNX Runtime operator wrapper and closed-loop drift metric."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class OrtOperator(torch.nn.Module):
    """Torch-callable ONNX Runtime session (dynamic batch, named I/O).

    Mimics the torch module interface so it can replace the operator of a
    learned-dynamics hook unchanged. On a Qualcomm board the only difference is
    ``providers=["QNNExecutionProvider", "CPUExecutionProvider"]``.
    """

    def __init__(self, model_path: str | Path, providers: list[str] | None = None) -> None:
        super().__init__()
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - exercised without extra
            raise ImportError(
                "OrtOperator requires the optional 'onnxruntime' package "
                "(pip install cloud-robotics-sim[deploy]); "
                f"original error: {exc}"
            ) from exc
        self._providers = providers or ["CPUExecutionProvider"]
        self._sess = ort.InferenceSession(str(model_path), providers=self._providers)
        self._inp = self._sess.get_inputs()[0].name

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        arr = state.detach().cpu().numpy().astype(np.float32, copy=False)
        out = self._sess.run(None, {self._inp: arr})[0]
        return torch.from_numpy(np.asarray(out)).to(device=state.device, dtype=state.dtype)


def drift(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    """Mean / max absolute elementwise error between two trajectories.

    Shapes must match; both tensors are detached first.
    """
    ref = reference.detach()
    cand = candidate.detach()
    if ref.shape != cand.shape:
        raise ValueError(f"trajectory shape mismatch: {ref.shape} vs {cand.shape}")
    err = (cand - ref).abs()
    return {"mean_abs": float(err.mean().item()), "max_abs": float(err.max().item())}
