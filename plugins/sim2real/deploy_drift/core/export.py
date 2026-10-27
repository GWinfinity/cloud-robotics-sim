"""ONNX export and precision transforms for learned-dynamics operators.

All heavy optional dependencies (``onnx``, ``onnxruntime``,
``onnxconverter-common``) are imported lazily inside each function and raise a
descriptive :class:`ImportError` naming the ``deploy`` extra when missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_OPSET = 17

_DEPLOY_EXTRA_HINT = "pip install cloud-robotics-sim[deploy]"


def _require(module: str) -> Any:
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise ImportError(
            f"deploy drift harness requires the optional '{module}' package "
            f"({_DEPLOY_EXTRA_HINT}); original error: {exc}"
        ) from exc


def export_onnx(
    op: Any,
    state_dim: int,
    path: str | Path,
    opset: int = DEFAULT_OPSET,
    batch: int = 1,
) -> str:
    """Export ``op`` (a torch module) to ONNX with a dynamic batch axis.

    Uses the legacy torch exporter: the dynamo exporter (default on torch 2.9+)
    emits graphs whose shape inference ORT's quantizer rejects, while the
    legacy exporter produces the classic MatMul/Tanh graph downstream tools
    expect.
    """
    import torch

    was_training = op.training
    op.eval()
    dummy = torch.zeros(batch, state_dim, dtype=torch.float32)
    torch.onnx.export(
        op,
        dummy,
        str(path),
        input_names=["state"],
        output_names=["out"],
        dynamic_axes={"state": {0: "batch"}, "out": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )
    if was_training:
        op.train()
    return str(path)


def quantize_dynamic_int8(src: str | Path, dst: str | Path) -> str:
    """Dynamic INT8 weight quantization via onnxruntime.quantization."""
    _require("onnxruntime")
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    return str(dst)


def convert_fp16(src: str | Path, dst: str | Path) -> str:
    """Convert an ONNX model to fp16 weights/activations (I/O stay f32)."""
    onnx = _require("onnx")
    _require("onnxconverter_common")
    from onnxconverter_common.float16 import convert_float_to_float16

    model = onnx.load(str(src))
    convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model, str(dst))
    return str(dst)
