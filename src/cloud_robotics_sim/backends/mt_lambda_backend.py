"""MT Lambda (Genesis + Quadrants/MUSA) backend implementation.

This backend targets Moore Threads' MUSA architecture while keeping the
Genesis physics engine and scene API. It is designed for the scenario where
Genesis/Quadrants gains a MUSA backend: the backend selects the MUSA compute
arch when available, and falls back to CUDA for development/testing on
non-MUSA machines.

For training and inference, this backend pairs with ``torch_musa`` so that
PyTorch models run on MUSA while the simulator runs on the same GPU through
Genesis/Quadrants.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from cloud_robotics_sim.backend.base import SceneBackend
from cloud_robotics_sim.backend.types import BackendName, ViewerOptions
from cloud_robotics_sim.backends.genesis_backend import GenesisBackend

logger = logging.getLogger(__name__)


def _musa_available() -> bool:
    """Detect whether a MUSA device is visible to the current process."""
    try:
        import torch
        import torch_musa  # noqa: F401

        return bool(torch.musa.is_available())  # type: ignore[attr-defined]
    except Exception:
        pass

    # Fallback: check MUSA environment variables.
    if os.environ.get("MTHREADS_VISIBLE_DEVICES", "").lower() in {"", "none"}:
        return False
    return True


class MTLambdaBackend(GenesisBackend):
    """Moore Threads MT Lambda simulator backend.

    This backend is a Genesis-compatible backend that attempts to run the
    Genesis/Quadrants physics engine on a MUSA GPU. As of ``genesis-world``
    1.2.2, the public Genesis package does not yet expose a MUSA backend
    enum; this implementation:

    1. Probes for MUSA device availability via ``torch_musa``.
    2. If a future Genesis/Quadrants build exposes ``gs.musa``, uses it.
    3. Otherwise falls back to the Genesis CUDA backend so that code can be
       developed and tested on NVIDIA/CUDA hardware with the same API.

    The fallback path logs a clear warning; it is intended for development
    and CI, not production deployment on MT Lambda clusters.
    """

    def __init__(self) -> None:
        super().__init__()
        self._device: str = "musa"
        self._effective_genesis_backend: Any = None

    @property
    def name(self) -> BackendName:
        return BackendName.MT_LAMBDA

    def initialize(
        self,
        *,
        headless: bool = True,
        device: str = "musa",
        **kwargs: Any,
    ) -> None:
        """Initialize Genesis/Quadrants, preferring MUSA when available.

        Args:
            headless: Run without a viewer.
            device: Target compute device (``musa`` or ``cuda``).
            **kwargs: Extra arguments forwarded to ``GenesisBackend.initialize``.
        """
        self._device = device.lower()

        # 1. Detect MUSA hardware.
        has_musa = _musa_available() if self._device == "musa" else False

        # 2. Try to find a MUSA backend in Genesis/Quadrants.
        musa_backend = None
        if has_musa:
            try:
                import genesis as gs

                musa_backend = getattr(gs, "musa", None)
                if musa_backend is None and hasattr(gs, "_gs_backend"):
                    musa_backend = getattr(gs._gs_backend, "musa", None)
            except Exception:
                pass

        # 3. Use MUSA backend if it exists; otherwise fall back to CUDA.
        if musa_backend is not None:
            logger.info("MT Lambda backend: using Genesis/Quadrants MUSA backend")
            self._effective_genesis_backend = musa_backend
            use_cuda = False
        elif has_musa:
            logger.warning(
                "MUSA device detected but Genesis/Quadrants does not yet "
                "expose a MUSA backend (genesis-world<1.3.0). "
                "Falling back to Genesis CUDA backend for development."
            )
            self._effective_genesis_backend = None
            use_cuda = True
        else:
            logger.warning(
                "MT Lambda backend: no MUSA device detected. "
                "Falling back to Genesis CUDA backend."
            )
            self._effective_genesis_backend = None
            use_cuda = True

        super().initialize(
            headless=headless,
            device="cuda" if use_cuda else "musa",
            **kwargs,
        )

    def create_scene(
        self,
        *,
        dt: float,
        substeps: int,
        headless: bool = True,
        viewer_options: ViewerOptions | None = None,
        fem_options: Any | None = None,
        pbd_options: Any | None = None,
        sph_options: Any | None = None,
        mpm_options: Any | None = None,
        sf_options: Any | None = None,
    ) -> SceneBackend:
        """Create a Genesis scene.

        This delegates to ``GenesisBackend.create_scene``; once Genesis/Quadrants
        gains a MUSA backend, the effective backend selected in ``initialize``
        will be used automatically.
        """
        return super().create_scene(
            dt=dt,
            substeps=substeps,
            headless=headless,
            viewer_options=viewer_options,
            fem_options=fem_options,
            pbd_options=pbd_options,
            sph_options=sph_options,
            mpm_options=mpm_options,
            sf_options=sf_options,
        )
