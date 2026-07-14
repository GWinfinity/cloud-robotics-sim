"""Run the improvement loop with a stub environment.

This script demonstrates the improvement-loop workflow without depending on
Genesis or the real cloud_robotics_sim components. It uses a minimal Gymnasium-
like stub environment so the Observe → Diagnose → Propose → Validate → Adopt →
Learn cycle can execute end-to-end.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

import numpy as np

from cloud_robotics_sim.runtime import ImprovementLoop, LoopConfig

logging.basicConfig(level=logging.INFO)


class StubRobotEnv:
    """Minimal stub environment with the interface the loop expects."""

    def __init__(self, config: dict, seed: int = 0) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.action_space = {
            "low": -1.0,
            "high": 1.0,
            "shape": (8,),
            "dtype": "float32",
        }
        self._episode = 0

    def reset(self) -> tuple[dict, dict]:
        self.step_count = 0
        self._episode += 1
        self.rng = np.random.default_rng(self._episode)
        return {"obs": np.zeros(10, dtype=np.float32)}, {
            "success": False,
            "physics_violations": 0,
            "stability_score": 1.0,
        }

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        self.step_count += 1

        # Simulate a task that is currently failing: low success rate / low reward.
        reward = -0.1 * self.step_count + self.rng.normal(0.0, 0.05)
        terminated = self.step_count >= 50 and self.rng.random() < 0.1
        truncated = self.step_count >= 200

        info = {
            "success": terminated and self.rng.random() < 0.05,
            "physics_violations": int(self.rng.poisson(0.5)),
            "stability_score": max(0.0, 1.0 - self.step_count * 0.005),
        }

        return {"obs": np.zeros(10, dtype=np.float32)}, float(reward), terminated, truncated, info


def make_stub_env(config: dict) -> StubRobotEnv:
    """Factory matching the make_env_fn signature."""
    return StubRobotEnv(config)


loop = ImprovementLoop(
    make_env_fn=make_stub_env,
    loop_config=LoopConfig(
        baseline_config_path="configs/franka_pickplace.yaml",
        output_dir="./outputs/improvement_loop_stub",
        n_baseline_episodes=10,
        n_validation_episodes=15,
        max_iterations=1,
    ),
)

report = loop.run()
print(json.dumps(report, indent=2, default=str))
