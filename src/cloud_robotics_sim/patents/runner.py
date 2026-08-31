"""Runner utilities for executing patent simulations.

Provides headless/viewer execution, parameter sweeps, and simple video
recording helpers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from cloud_robotics_sim.patents.base import PatentSimConfig, SimState
from cloud_robotics_sim.patents.registry import create_simulation

logger = logging.getLogger(__name__)


class _Cv2VideoWriter:
    """In-process MP4 writer via OpenCV.

    Preferred over ``imageio-ffmpeg`` on Windows because it avoids spawning a
    separate ffmpeg subprocess, whose handle inheritance can fail with
    ``OSError: [WinError 6]`` in non-console environments.
    """

    def __init__(
        self,
        path: str | Path,
        fps: int,
        resolution: tuple[int, int],
    ) -> None:
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        self._writer = cv2.VideoWriter(str(path), fourcc, float(fps), resolution)
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {path}")

    def append_data(self, frame: np.ndarray) -> None:
        import cv2

        self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self._writer.release()


def _make_writer(
    record_path: str | Path,
    fps: int,
    resolution: tuple[int, int],
) -> Any:
    """Create a video writer, preferring OpenCV and falling back to imageio."""
    try:
        return _Cv2VideoWriter(record_path, fps, resolution)
    except Exception as exc:
        logger.warning("OpenCV video writer unavailable (%s); trying imageio", exc)
    try:
        import imageio

        return imageio.get_writer(str(record_path), fps=fps)
    except Exception as exc:
        logger.warning("Could not initialize any video writer: %s", exc)
        return None


def run_patent_simulation(
    patent_id: str,
    *,
    headless: bool = True,
    steps: int = 500,
    dt: float = 0.01,
    substeps: int = 10,
    resolution: tuple[int, int] = (640, 480),
    device: str = "cuda",
    seed: int = 0,
    parameters: dict[str, Any] | None = None,
    record_path: str | Path | None = None,
    fps: int = 30,
    follow_entity: str | None = None,
) -> SimState:
    """Build, reset, and run a patent simulation.

    Args:
        patent_id: Canonical patent identifier.
        headless: Whether to run without the interactive viewer.
        steps: Number of control steps to execute.
        dt: Physics timestep.
        substeps: Physics substeps per control step.
        resolution: Camera resolution.
        device: Genesis compute device.
        seed: Random seed.
        parameters: Interactive parameter overrides.
        record_path: Optional path to save an MP4 video.
        fps: Video framerate.
        follow_entity: Optional entity name the camera should follow
            (e.g. ``"aircraft"`` for the Wright Flyer).

    Returns:
        Final simulation state.
    """
    config = PatentSimConfig(
        patent_id=patent_id,
        headless=headless,
        dt=dt,
        substeps=substeps,
        resolution=resolution,
        device=device,
        seed=seed,
        parameters=parameters or {},
    )
    sim = create_simulation(patent_id, config=config)
    sim.build()
    sim.reset()

    if follow_entity is not None and sim.camera is not None:
        entity = sim.get_entity(follow_entity)
        if entity is None:
            logger.warning(
                "follow_entity '%s' not found; available: %s",
                follow_entity,
                list(sim._entities.keys()),
            )
        else:
            sim.camera.follow_entity(entity, smoothing=0.2, fix_orientation=False)

    writer = (
        _make_writer(record_path, fps, resolution) if record_path is not None else None
    )

    final_state: SimState = sim.get_state()
    for step in range(steps):
        final_state = sim.step()

        if writer is not None and step % max(1, int(1.0 / (dt * substeps * fps))) == 0:
            frame = sim.render()
            if frame is not None:
                writer.append_data(frame)

    if writer is not None:
        writer.close()

    sim.close()
    return final_state
