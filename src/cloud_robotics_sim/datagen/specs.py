"""Episode specifications for Genesis-based synthetic data generation.

These dataclasses are the portable contract between scene/motion producers and
the renderer: they carry everything needed to stage one episode (what to
spawn, where, which cameras watch it, how domain randomization is sampled)
without importing Genesis. Keeping them simulator-agnostic mirrors the
datagen/README.md split in the reference pipeline while staying importable on
machines without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

_CAMERA_MODELS = ("pinhole", "thinlens", "fisheye")


@dataclass(frozen=True)
class ObjectPlacement:
    """One object instance to spawn for an episode."""

    class_name: str
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float] | None = None
    instance: int = 0

    def __post_init__(self) -> None:
        """Validate the placement fields."""
        if not self.class_name:
            raise ValueError("placement needs a non-empty class_name")
        if len(self.pos) != 3:
            raise ValueError(f"pos must have 3 components, got {self.pos}")
        if self.quat is not None and len(self.quat) != 4:
            raise ValueError(f"quat must have 4 components, got {self.quat}")
        if self.instance < 0:
            raise ValueError("instance index must be >= 0")


@dataclass(frozen=True)
class CameraSpec:
    """A scene camera; set ``follow`` to track a spawned object's base link."""

    name: str
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    res: tuple[int, int] = (640, 480)
    fov: float = 60.0
    model: str = "pinhole"
    follow: str | None = None

    def __post_init__(self) -> None:
        """Validate the camera fields."""
        if not self.name:
            raise ValueError("camera needs a non-empty name")
        if self.model not in _CAMERA_MODELS:
            raise ValueError(
                f"camera model must be one of {_CAMERA_MODELS}, got {self.model!r}"
            )
        if len(self.res) != 2 or min(self.res) < 16:
            raise ValueError(f"resolution must be >= 16 px per axis, got {self.res}")
        if not 1.0 <= self.fov <= 179.0:
            raise ValueError(f"fov must be in [1, 179] degrees, got {self.fov}")
        if len(self.pos) != 3 or len(self.lookat) != 3:
            raise ValueError("camera pos/lookat must have 3 components")


@dataclass(frozen=True)
class RandomizationSpec:
    """Domain randomization ranges, sampled once per episode from ``seed``."""

    seed: int = 0
    background_colors: tuple[tuple[float, float, float], ...] = (
        (0.75, 0.78, 0.82),
        (0.55, 0.52, 0.48),
        (0.35, 0.40, 0.48),
    )
    floor_colors: tuple[tuple[float, float, float], ...] = (
        (0.62, 0.58, 0.52),
        (0.45, 0.47, 0.50),
        (0.30, 0.34, 0.30),
    )
    light_intensity_range: tuple[float, float] = (0.8, 1.4)
    light_color_jitter: float = 0.08
    n_lights: int = 2

    def __post_init__(self) -> None:
        """Validate palettes and light ranges."""
        if not self.background_colors:
            raise ValueError("need at least one background color")
        if not self.floor_colors:
            raise ValueError("need at least one floor color")
        lo, hi = self.light_intensity_range
        if not 0.0 < lo <= hi:
            raise ValueError(
                f"light intensity range must satisfy 0 < lo <= hi, got {self.light_intensity_range}"
            )
        if not 0.0 <= self.light_color_jitter <= 1.0:
            raise ValueError("light_color_jitter must be in [0, 1]")
        if self.n_lights < 0:
            raise ValueError("n_lights must be >= 0")
        for palette, label in (
            (self.background_colors, "background_colors"),
            (self.floor_colors, "floor_colors"),
        ):
            for color in palette:
                if len(color) != 3 or not all(0.0 <= c <= 1.0 for c in color):
                    raise ValueError(f"{label} entries must be RGB triples in [0, 1]")

    @staticmethod
    def _rgb(values: np.ndarray) -> tuple[float, float, float]:
        """Narrow a length-3 array to an RGB triple (mypy-friendly)."""
        return (float(values[0]), float(values[1]), float(values[2]))

    def sample(self) -> "DrSample":
        """Draw one concrete randomization outcome (reproducible via ``seed``)."""
        rng = np.random.default_rng(self.seed)
        background = np.asarray(
            self.background_colors[rng.integers(len(self.background_colors))],
            dtype=np.float64,
        )
        floor = np.asarray(
            self.floor_colors[rng.integers(len(self.floor_colors))], dtype=np.float64
        )
        lo, hi = self.light_intensity_range
        lights = []
        for _ in range(self.n_lights):
            intensity = float(rng.uniform(lo, hi))
            color = np.clip(rng.normal(1.0, self.light_color_jitter, size=3), 0.0, 1.0)
            lights.append(LightSample(intensity=intensity, color=self._rgb(color)))
        return DrSample(
            background=self._rgb(background),
            floor=self._rgb(floor),
            lights=tuple(lights),
        )


@dataclass(frozen=True)
class LightSample:
    """One randomized point light."""

    intensity: float
    color: tuple[float, float, float]


@dataclass(frozen=True)
class DrSample:
    """A concrete per-episode domain randomization outcome."""

    background: tuple[float, float, float]
    floor: tuple[float, float, float]
    lights: tuple[LightSample, ...]

    def to_metadata(self) -> dict[str, Any]:
        """Serialize the sample into episode metadata (JSON-safe)."""
        return {
            "background": list(self.background),
            "floor": list(self.floor),
            "lights": [
                {"intensity": light.intensity, "color": list(light.color)}
                for light in self.lights
            ],
        }


@dataclass(frozen=True)
class EpisodeSpec:
    """Everything the renderer needs for one episode."""

    episode_id: str
    objects: tuple[ObjectPlacement, ...]
    cameras: tuple[CameraSpec, ...]
    steps: int
    randomization: RandomizationSpec = field(default_factory=RandomizationSpec)

    def __post_init__(self) -> None:
        """Validate the episode fields."""
        if not self.episode_id:
            raise ValueError("episode needs a non-empty episode_id")
        if self.steps < 0:
            raise ValueError(f"steps must be >= 0, got {self.steps}")
        names = [camera.name for camera in self.cameras]
        if len(names) != len(set(names)):
            raise ValueError(f"camera names must be unique, got {names}")
        follows = {camera.follow for camera in self.cameras if camera.follow}
        placed = {placement.class_name for placement in self.objects}
        unknown = follows - placed
        if unknown:
            raise ValueError(
                f"cameras follow unknown objects {sorted(unknown)}; "
                f"place them first (have {sorted(placed)})"
            )
