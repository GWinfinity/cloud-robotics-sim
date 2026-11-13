#!/usr/bin/env python3
"""Shard-parallel batch runner for genesis-world synthetic data generation.

Genesis-side replacement for the reference pipeline's render stage
orchestration: one process per shard, strided episode ownership,
per-episode resume via the dataset writer. Rerun a shard safely after a
crash; finished episodes are skipped.

    uv run python tools/run_datagen.py --objects-dir outputs/home_physics_assets/objects \
        --output-dir outputs/home_physics_assets/datagen --episodes 100 --steps 30 \
        --shard 0 --num-shards 4 --backend cuda

Genesis is initialized once per process (per-env ``gs.init`` constraint).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cloud_robotics_sim.datagen import (  # noqa: E402
    DatasetWriter,
    EpisodeGenerator,
    GenerationConfig,
    GenesisEpisodeRenderer,
)
from cloud_robotics_sim.robotwin.object_library import (  # noqa: E402
    RoboTwinObjectLibrary,
)
from cloud_robotics_sim.utils.genesis_compat import get_genesis_backend  # noqa: E402

logger = logging.getLogger("run_datagen")

_CLASS_DIR_RE = re.compile(r"^\d{3}_(.+)$")


def _bare_name(class_dir: str) -> str:
    """Strip the ``NNN_`` id prefix from a class dir name."""
    match = _CLASS_DIR_RE.match(class_dir)
    return match.group(1) if match else class_dir


def _available_classes(objects_dir: Path) -> tuple[str, ...]:
    """Bare class names in the object library (sorted for stable CLI output)."""
    library = RoboTwinObjectLibrary(objects_dir)
    return tuple(sorted(_bare_name(c) for c in library.list_classes()))


def main(argv: list[str] | None = None) -> int:
    """Render one shard of the dataset; exit 1 when any episode failed."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--objects-dir",
        default="outputs/home_physics_assets/objects",
        help="RoboTwin-OD object library root (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/home_physics_assets/datagen",
        help="dataset output root (default: %(default)s)",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--backend", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="restrict placement to these bare class names (default: all)",
    )
    parser.add_argument("--objects-min", type=int, default=2)
    parser.add_argument("--objects-max", type=int, default=4)
    parser.add_argument("--arena", type=float, default=0.45)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    available = _available_classes(Path(args.objects_dir))
    if args.classes is not None:
        unknown = sorted(set(args.classes) - set(available))
        if unknown:
            logger.error("unknown classes %s (available: %s)", unknown, list(available))
            return 2
        available = tuple(sorted(args.classes))
    if not available:
        logger.error("no object classes found under %s", args.objects_dir)
        return 2

    config = GenerationConfig(
        episodes=args.episodes,
        steps=args.steps,
        available_classes=available,
        seed=args.seed,
        objects_per_episode=(args.objects_min, args.objects_max),
        arena=(args.arena, args.arena),
    )

    import genesis as gs

    get_genesis_backend(args.backend)
    gs.init(
        backend=gs.gpu if args.backend == "cuda" else gs.cpu,
        logging_level="warning",
    )

    generator = EpisodeGenerator(config)
    renderer = GenesisEpisodeRenderer(args.objects_dir, backend=args.backend)
    writer = DatasetWriter(args.output_dir)

    done = 0
    skipped = 0
    failed: list[dict[str, str]] = []
    for index in config.shard_indices(args.shard, args.num_shards):
        spec = generator.spec(index)
        if writer.is_done(spec.episode_id):
            skipped += 1
            continue
        try:
            writer.write_episode(renderer.render_episode(spec))
            done += 1
        except Exception as exc:  # noqa: BLE001 - shard resilience, retried on rerun
            logger.exception("episode %s failed", spec.episode_id)
            failed.append({"episode_id": spec.episode_id, "error": str(exc)})

    summary = {
        "shard": args.shard,
        "num_shards": args.num_shards,
        "done": done,
        "skipped": skipped,
        "failed": failed,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
