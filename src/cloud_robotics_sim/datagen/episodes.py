"""Seeded episode-spec factory for batch synthetic data generation.

Turns a small config plus the list of available object classes into
deterministic :class:`~.specs.EpisodeSpec` instances, so shards on any
machine can regenerate the exact same episode list without sharing state.
Placements are random drops inside an arena with spacing rejection, the
third-person camera is fixed, and one random placed object carries the
follow (egocentric) camera — the same staging shape the reference
pipeline uses before rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .specs import (
    CameraSpec,
    EpisodeSpec,
    ObjectPlacement,
    RandomizationSpec,
)

_TPV_CAMERA = CameraSpec(
    name="tpv",
    pos=(1.2, 0.0, 0.9),
    lookat=(0.0, 0.0, 0.1),
    res=(640, 480),
    fov=60.0,
)
_FPV_CAMERA = CameraSpec(
    name="fpv",
    pos=(0.0, 0.35, 0.35),
    lookat=(0.0, -0.4, -0.2),
    res=(320, 240),
    fov=90.0,
    follow=None,  # bound to a random placed object per episode
)
_MIN_SPACING = 0.12
_PLACEMENT_TRIES = 24


@dataclass(frozen=True)
class GenerationConfig:
    """Controls how episode specs are drawn from ``seed``.

    Args:
        episodes: total number of episodes in the dataset.
        steps: physics steps captured per episode.
        seed: base seed; episode ``i`` uses ``seed + i``.
        objects_per_episode: inclusive ``(min, max)`` object count.
        arena: ``(half_x, half_y)`` placement bounds on the floor.
        available_classes: bare class names the generator may place.
    """

    episodes: int
    steps: int
    available_classes: tuple[str, ...]
    seed: int = 0
    objects_per_episode: tuple[int, int] = (2, 4)
    arena: tuple[float, float] = (0.45, 0.45)

    def __post_init__(self) -> None:
        """Validate the generation ranges."""
        if self.episodes < 1:
            raise ValueError(f"episodes must be >= 1, got {self.episodes}")
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        if not self.available_classes:
            raise ValueError("need at least one available class")
        if len(set(self.available_classes)) != len(self.available_classes):
            raise ValueError(f"duplicate classes: {self.available_classes}")
        lo, hi = self.objects_per_episode
        if not 1 <= lo <= hi:
            raise ValueError(
                f"objects_per_episode needs 1 <= min <= max, got {self.objects_per_episode}"
            )
        if min(self.arena) <= 0.0:
            raise ValueError(f"arena half-extents must be positive, got {self.arena}")

    def shard_indices(self, shard: int, num_shards: int) -> range:
        """Strided episode indices owned by ``shard`` (disjoint, complete)."""
        if num_shards < 1 or not 0 <= shard < num_shards:
            raise ValueError(f"shard must be in [0, {num_shards}), got {shard}")
        return range(shard, self.episodes, num_shards)


class EpisodeGenerator:
    """Deterministic factory: ``spec(index)`` is stable across processes."""

    def __init__(self, config: GenerationConfig) -> None:
        self._config = config

    @property
    def config(self) -> GenerationConfig:
        return self._config

    def spec(self, index: int) -> EpisodeSpec:
        """Build the episode spec for ``index`` (deterministic)."""
        config = self._config
        if not 0 <= index < config.episodes:
            raise ValueError(f"episode index {index} outside [0, {config.episodes})")
        rng = np.random.default_rng(config.seed + index)
        lo, hi = config.objects_per_episode
        count = int(rng.integers(lo, hi + 1))
        placements = self._placements(rng, count)
        follow = placements[int(rng.integers(len(placements)))].class_name
        fpv = CameraSpec(
            name=_FPV_CAMERA.name,
            pos=_FPV_CAMERA.pos,
            lookat=_FPV_CAMERA.lookat,
            res=_FPV_CAMERA.res,
            fov=_FPV_CAMERA.fov,
            model=_FPV_CAMERA.model,
            follow=follow,
        )
        return EpisodeSpec(
            episode_id=f"ep_{index:06d}",
            objects=tuple(placements),
            cameras=(_TPV_CAMERA, fpv),
            steps=config.steps,
            randomization=RandomizationSpec(seed=config.seed + index),
        )

    def _placements(self, rng: Any, count: int) -> list[ObjectPlacement]:
        config = self._config
        chosen = rng.choice(np.asarray(config.available_classes, dtype=str), size=count)
        accepted: list[tuple[str, tuple[float, float, float], tuple]] = []
        for class_name in chosen:
            xy = self._free_spot(rng, [p[1][:2] for p in accepted])
            height = float(rng.uniform(0.08, 0.25))
            pos = (xy[0], xy[1], height)
            yaw = float(rng.uniform(-np.pi, np.pi))
            quat = (
                float(np.cos(yaw / 2.0)),
                0.0,
                0.0,
                float(np.sin(yaw / 2.0)),
            )
            accepted.append((str(class_name), pos, quat))
        return [
            ObjectPlacement(name, pos=pos, quat=quat) for name, pos, quat in accepted
        ]

    def _free_spot(self, rng: Any, taken: list[tuple[float, float]]) -> tuple:
        half_x, half_y = self._config.arena
        for _ in range(_PLACEMENT_TRIES):
            candidate = (
                float(rng.uniform(-half_x, half_x)),
                float(rng.uniform(-half_y, half_y)),
            )
            if all(
                (candidate[0] - x) ** 2 + (candidate[1] - y) ** 2 >= _MIN_SPACING**2
                for x, y in taken
            ):
                return candidate
        # arena too crowded for the spacing rule; fall back to the last draw
        return candidate


__all__ = ["EpisodeGenerator", "GenerationConfig"]
