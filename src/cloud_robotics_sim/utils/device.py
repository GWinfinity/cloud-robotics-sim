"""Device selection utilities for CPU/CUDA/MUSA backends.

This module provides a thin abstraction over PyTorch device availability so
that callers can request ``cpu``, ``cuda``, or ``musa`` without hard-coding
vendor-specific APIs. MUSA support depends on ``torch_musa`` being installed.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torch

    HAS_TORCH = True
except Exception:  # pragma: no cover - handled gracefully
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

try:
    import torch.cuda

    HAS_CUDA = True
except Exception:  # pragma: no cover - handled gracefully
    HAS_CUDA = False

try:
    import torch_musa  # noqa: F401

    HAS_MUSA = True
except Exception:  # pragma: no cover - handled gracefully
    HAS_MUSA = False


def is_musa_available() -> bool:
    """Return True if a MUSA device is available via ``torch_musa``."""
    if not HAS_MUSA or not HAS_TORCH or torch is None:
        return False
    try:
        musa_module = getattr(torch, "musa", None)
        if musa_module is None:
            return False
        return bool(musa_module.is_available())
    except Exception:
        return False


def is_cuda_available() -> bool:
    """Return True if a CUDA device is available."""
    if not HAS_CUDA or not HAS_TORCH or torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _env_device() -> Optional[str]:
    """Read ``CRS_DEVICE`` environment variable if set to a known value."""
    env = os.environ.get("CRS_DEVICE", "").lower()
    if env in ("musa", "cuda", "cpu"):
        return env
    return None


def default_device() -> str:
    """Return the default device based on availability and ``CRS_DEVICE``.

    Priority:
        1. ``CRS_DEVICE`` environment variable (if set to a known value).
        2. ``musa`` if ``torch_musa`` is installed and a device is present.
        3. ``cuda`` if an NVIDIA GPU is available.
        4. ``cpu`` otherwise.
    """
    env = _env_device()
    if env == "musa" and is_musa_available():
        return "musa"
    if env == "cuda" and is_cuda_available():
        return "cuda"
    if env == "cpu":
        return "cpu"

    if is_musa_available():
        return "musa"
    if is_cuda_available():
        return "cuda"
    return "cpu"


def get_device(preference: Optional[str] = None) -> str:
    """Resolve a device preference to an available device.

    Args:
        preference: Optional device name (``cpu``, ``cuda``, ``musa``). If the
            requested device is not available, falls back to the auto-selected
            default with a warning.

    Returns:
        One of ``"cpu"``, ``"cuda"``, or ``"musa"``.
    """
    if preference:
        pref = preference.lower()
        if pref == "musa" and is_musa_available():
            return "musa"
        if pref == "cuda" and is_cuda_available():
            return "cuda"
        if pref == "cpu":
            return "cpu"
        logger.warning(
            "Requested device %r is not available; falling back to auto-selected device",
            preference,
        )
    return default_device()


def set_default_device(device: Optional[str] = None) -> str:
    """Set PyTorch's default device if it is not CPU.

    Args:
        device: Optional device preference. If None, uses the default device.

    Returns:
        The resolved device string.
    """
    resolved = get_device(device)
    if HAS_TORCH and torch is not None and resolved != "cpu":
        try:
            torch.set_default_device(resolved)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to set default torch device to %r: %s", resolved, exc
            )
    return resolved


__all__ = [
    "HAS_TORCH",
    "HAS_CUDA",
    "HAS_MUSA",
    "is_musa_available",
    "is_cuda_available",
    "default_device",
    "get_device",
    "set_default_device",
]
