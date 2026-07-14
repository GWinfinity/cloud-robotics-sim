"""Smoke test for the improvement loop."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the package root is on sys.path so ``src.cloud_robotics_sim`` resolves.
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from cloud_robotics_sim.runtime import ImprovementLoop, LoopConfig, make_env_from_config

logging.basicConfig(level=logging.INFO)


def make_env_fn(config: dict) -> object:
    """Create environment with CPU-friendly, stable physics settings."""
    config["environment"]["simulation"]["headless"] = True
    config["environment"]["simulation"]["dt"] = 0.005
    config["environment"]["simulation"]["substeps"] = 20
    config["environment"]["simulation"]["resolution"] = [320, 240]
    config["use_cuda"] = False
    return make_env_from_config(config)


loop = ImprovementLoop(
    make_env_fn=make_env_fn,
    loop_config=LoopConfig(
        baseline_config_path="configs/franka_pickplace.yaml",
        output_dir="./outputs/improvement_loop_smoke",
        n_baseline_episodes=2,
        n_validation_episodes=3,
        max_iterations=1,
        thresholds={
            "min_success_rate": 0.8,
            "max_latency_ms": 1000.0,
            "max_physics_violations": 100.0,
            "min_stability_score": 0.0,
            "min_mean_reward": -1e6,
            "min_episodes": 1,
        },
    ),
)

report = loop.run()
print(report)
