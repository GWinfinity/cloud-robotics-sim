"""Three-tier render configuration (migration doc section 6).

Implements the doc's dual-layer rendering strategy:

============  ==========================  ==================================
Tier          Genesis component           Use case (doc section 6.1)
============  ==========================  ==================================
``rasterizer``  ``gs.renderers.Rasterizer``  seed search, RL, mass rollouts
``raytracer``   ``gs.renderers.RayTracer``   final training data (SAPIEN RT
              (LuisaRender)                 fidelity alignment: 32spp+OIDN)
``batch``       ``gs.renderers.BatchRenderer`` parallel policy eval, DR scans
              (Madrona, ``gs-madrona``)
============  ==========================  ==================================

Configs live in ``configs/render/*.yaml`` so the calibration outcome
(tone mapping, spp, denoise) is shared by all tasks (doc section 6.2).

GPU note: ``raytracer``/``batch`` require capable hardware; configuration,
scene wiring and the alignment metrics are testable on CPU, while the
final visual calibration (PSNR/LPIPS against SAPIEN RT) is a GPU-side
milestone (M4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

__all__ = [
    "RenderConfig",
    "RenderComparison",
    "compare_render_pair",
    "create_scene_with_render_config",
    "load_render_config",
    "make_genesis_renderer",
]

RenderMode = Literal["rasterizer", "raytracer", "batch"]


@dataclass
class RenderConfig:
    """Render tier configuration shared across tasks.

    Attributes:
        mode: Render tier (``rasterizer`` / ``raytracer`` / ``batch``).
        resolution: Camera resolution ``(width, height)``.
        fov: Vertical field of view in degrees (D435 three-camera layouts
            copy this verbatim from the task config, doc section 7).
        spp: Samples per pixel for the raytracer tier (SAPIEN RT used
            32spp + OIDN denoising).
        denoise: Enable denoising on the raytracer tier.
        raytracer_options: Extra keyword args forwarded to
            ``gs.renderers.RayTracer`` (e.g. ``tracing_depth``,
            ``rr_depth``, ``env_surface``).
        batch_use_rasterizer: Madrona backend toggle forwarded to
            ``gs.renderers.BatchRenderer(use_rasterizer=...)``.
    """

    mode: RenderMode = "rasterizer"
    resolution: tuple[int, int] = (640, 480)
    fov: float = 60.0
    spp: int = 32
    denoise: bool = True
    raytracer_options: dict[str, Any] = field(default_factory=dict)
    batch_use_rasterizer: bool = True

    def __post_init__(self) -> None:
        """Validate mode, resolution and spp."""
        if self.mode not in ("rasterizer", "raytracer", "batch"):
            raise ValueError(f"Unknown render mode: {self.mode!r}")
        if len(self.resolution) != 2 or any(int(v) <= 0 for v in self.resolution):
            raise ValueError(f"Invalid resolution: {self.resolution!r}")
        self.resolution = (int(self.resolution[0]), int(self.resolution[1]))
        if self.spp <= 0:
            raise ValueError("spp must be positive")


def load_render_config(path: str | Path) -> RenderConfig:
    """Load a render tier config from a YAML file."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Render config must be a mapping: {path}")
    known = {
        "mode",
        "resolution",
        "fov",
        "spp",
        "denoise",
        "raytracer_options",
        "batch_use_rasterizer",
    }
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown render config keys in {path}: {sorted(unknown)}")
    if "resolution" in data:
        data["resolution"] = tuple(data["resolution"])
    return RenderConfig(**data)


def make_genesis_renderer(config: RenderConfig) -> Any:
    """Instantiate the Genesis renderer for the configured tier."""
    import genesis as gs  # lazy import: config stays usable without Genesis

    if config.mode == "rasterizer":
        return gs.renderers.Rasterizer()
    if config.mode == "raytracer":
        return gs.renderers.RayTracer(**config.raytracer_options)
    return gs.renderers.BatchRenderer(use_rasterizer=config.batch_use_rasterizer)


def create_scene_with_render_config(
    backend: Any,
    config: RenderConfig,
    *,
    dt: float = 1.0 / 100.0,
    substeps: int = 1,
    headless: bool = True,
) -> Any:
    """Create a scene whose renderer follows the render tier config.

    Args:
        backend: A ``SimulatorBackend`` (e.g. ``GenesisBackend``).
        config: Render tier configuration.
        dt: Simulation timestep.
        substeps: Physics substeps per step.
        headless: Run without the interactive viewer.

    Returns:
        The created ``SceneBackend``.
    """
    renderer = make_genesis_renderer(config)
    return backend.create_scene(
        dt=dt, substeps=substeps, headless=headless, renderer=renderer
    )


@dataclass
class RenderComparison:
    """Image-pair alignment metrics (doc section 6.2, gate V4).

    The doc's acceptance gate is ``LPIPS < 0.15``; LPIPS requires a learned
    perceptual model and is computed on the GPU calibration host. The
    metrics here (PSNR / MAE) are the simulator-independent reference
    numbers recorded alongside it.
    """

    psnr: float
    mae: float

    @property
    def is_perfect(self) -> bool:
        """True when both images are identical."""
        return self.mae == 0.0


def compare_render_pair(
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    max_value: float = 255.0,
) -> RenderComparison:
    """Compute PSNR / MAE between two rendered images.

    Args:
        image_a: First image ``(H, W, C)``, uint8 or float.
        image_b: Second image, same shape as ``image_a``.
        max_value: Peak signal value (255 for uint8, 1.0 for float images).

    Returns:
        :class:`RenderComparison` with PSNR (dB, ``inf`` for identical
        images) and mean absolute error.
    """
    a = np.asarray(image_a, dtype=np.float64)
    b = np.asarray(image_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Image shapes differ: {a.shape} vs {b.shape}")
    mse = float(np.mean((a - b) ** 2))
    mae = float(np.mean(np.abs(a - b)))
    psnr = float("inf") if mse == 0.0 else 10.0 * np.log10(max_value**2 / mse)
    return RenderComparison(psnr=psnr, mae=mae)
