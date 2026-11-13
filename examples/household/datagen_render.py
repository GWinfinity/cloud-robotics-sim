"""Render household episodes with the genesis-world datagen pipeline.

Builds the generated household assets when missing, then renders a few
episodes with per-episode domain randomization (background, floor, lights)
and writes a resume-safe dataset (PNG frames + JSON metadata).

Run::

    uv run python examples/household/datagen_render.py --episodes 4 --steps 60

Genesis is initialized once per process; use --backend cpu on GPU-less hosts.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from cloud_robotics_sim.datagen import (  # noqa: E402
    CameraSpec,
    DatasetWriter,
    EpisodeSpec,
    GenesisEpisodeRenderer,
    ObjectPlacement,
    RandomizationSpec,
)
from cloud_robotics_sim.utils.genesis_compat import get_genesis_backend  # noqa: E402


def _build_objects(objects_dir: Path) -> None:
    """Ensure the small demo asset set exists (skips existing classes)."""
    from tools.build_home_physics_assets import main as build_assets

    build_assets(
        [
            "--objects-dir",
            str(objects_dir),
            "--classes",
            "glass_cup",
            "sponge",
        ]
    )


def _episode_spec(index: int, steps: int) -> EpisodeSpec:
    return EpisodeSpec(
        episode_id=f"demo_{index:06d}",
        steps=steps,
        objects=(
            ObjectPlacement("glass_cup", pos=(0.0, 0.0, 0.15)),
            ObjectPlacement("sponge", pos=(0.15, 0.0, 0.05)),
        ),
        cameras=(
            CameraSpec(
                name="tpv",
                pos=(0.9, 0.0, 0.6),
                lookat=(0.0, 0.0, 0.1),
                res=(640, 480),
                fov=60.0,
            ),
            CameraSpec(
                name="fpv",
                pos=(0.0, 0.0, 0.35),
                lookat=(0.0, 0.0, -1.0),
                res=(320, 240),
                fov=90.0,
                follow="glass_cup",
            ),
        ),
        randomization=RandomizationSpec(seed=index),
    )


def main(argv: list[str] | None = None) -> int:
    """Render episodes into a dataset directory and print a summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/home_physics_assets/datagen")
    parser.add_argument("--objects-dir", default=None)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--backend", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.steps < 1:
        parser.error("--episodes and --steps must be >= 1")

    import genesis as gs

    get_genesis_backend(args.backend)
    gs.init(
        backend=gs.gpu if args.backend == "cuda" else gs.cpu,
        logging_level="warning",
    )

    if args.objects_dir is not None:
        objects_dir = Path(args.objects_dir)
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="home_physics_objects_")
        objects_dir = Path(temp_dir.name)
    objects_dir.mkdir(parents=True, exist_ok=True)
    _build_objects(objects_dir)

    renderer = GenesisEpisodeRenderer(objects_dir, backend=args.backend)
    writer = DatasetWriter(args.output_dir)
    summary = []
    try:
        for index in range(args.episodes):
            spec = _episode_spec(index, args.steps)
            if writer.is_done(spec.episode_id):
                print(f"skip existing episode {spec.episode_id}")
                continue
            result = renderer.render_episode(spec)
            writer.write_episode(result)
            summary.append(
                {
                    "episode_id": result.episode_id,
                    "steps": result.steps,
                    "cameras": sorted(result.frames),
                    "background": result.metadata["domain_randomization"]["background"],
                }
            )
    finally:
        if args.objects_dir is None:
            temp_dir.cleanup()

    print(json.dumps({"episodes": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
