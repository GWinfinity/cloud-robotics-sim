"""Benchmark evaluation runner (W8 of docs/ROBODOJO_P0_PLAN.md).

Runs every task config listed in a benchmark suite YAML over its
``evaluation.seeds x episodes_per_seed`` grid with a seeded policy and writes
a leaderboard report:

    report.json       deterministic leaderboard (byte-reproducible across runs)
    report.md         human-readable leaderboard table
    episodes.jsonl    per-episode records incl. wall-clock durations

Example:
    python tools/run_benchmark.py --suite data/suites/pick_place_smoke.yaml \
        --out outputs/benchmarks/pick_place_smoke --policy zero
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from cloud_robotics_sim.core.benchmark import load_suite, run_benchmark
from cloud_robotics_sim.core.config_loader import ConfigError

logger = logging.getLogger("run_benchmark")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the benchmark runner."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--suite",
        required=True,
        help="Path to a benchmark suite YAML (data/suites/*.yaml).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for report.json / report.md / episodes.jsonl.",
    )
    parser.add_argument(
        "--policy",
        choices=["zero", "random"],
        default="zero",
        help="Rollout policy: 'zero' (hold command) or seeded 'random'.",
    )
    parser.add_argument(
        "--episodes-limit",
        type=int,
        default=None,
        help="Cap episodes per task (truncates the seed grid; smoke testing).",
    )
    parser.add_argument(
        "--save-trajectories",
        action="store_true",
        help="Write per-episode action/reward JSON traces to trajectories/.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args(argv)

    # Register the built-in components (empty_room / franka_panda / pick_place)
    # into the default AssetRegistry. Imported for side effects only.
    import cloud_robotics_sim.runtime.main  # noqa: F401

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        suite = load_suite(args.suite)
        summary = run_benchmark(
            suite,
            args.out,
            policy_kind=args.policy,
            episodes_limit=args.episodes_limit,
            save_trajectories=args.save_trajectories,
        )
    except ConfigError as exc:
        logger.error("configuration error: %s", exc)
        return 2

    overall = summary["overall"]
    logger.info(
        "suite '%s': %d/%d episodes succeeded (%.1f%%); reports in %s",
        summary["suite"],
        overall["success"],
        overall["episodes"],
        100.0 * overall["success_rate"],
        Path(args.out).resolve(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
