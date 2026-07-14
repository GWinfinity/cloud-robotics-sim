"""Metrics collection and storage for the continuous improvement loop."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EpisodeMetrics:
    """Metrics collected from a single episode."""

    success: bool = False
    episode_reward: float = 0.0
    episode_length: int = 0
    latency_ms: float = 0.0
    physics_violations: int = 0  # collisions, penetrations, joint limits
    stability_score: float = 1.0  # 1.0 = stable, 0.0 = unstable

    # Task-specific
    final_distance_to_goal: float | None = None
    time_to_success: float | None = None

    # Extra info for diagnostics
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "episode_reward": self.episode_reward,
            "episode_length": self.episode_length,
            "latency_ms": self.latency_ms,
            "physics_violations": self.physics_violations,
            "stability_score": self.stability_score,
            "final_distance_to_goal": self.final_distance_to_goal,
            "time_to_success": self.time_to_success,
            "info": self.info,
        }


@dataclass
class MetricSummary:
    """Aggregated metrics over multiple episodes."""

    episodes: int = 0
    success_rate: float = 0.0
    mean_reward: float = 0.0
    mean_latency_ms: float = 0.0
    mean_length: float = 0.0
    mean_physics_violations: float = 0.0
    mean_stability_score: float = 1.0
    mean_final_distance: float | None = None

    raw: list[EpisodeMetrics] = field(default_factory=list)

    @classmethod
    def from_episodes(cls, episodes: list[EpisodeMetrics]) -> "MetricSummary":
        if not episodes:
            return cls()

        successes = sum(1 for e in episodes if e.success)
        success_rate = successes / len(episodes)

        dists = [
            e.final_distance_to_goal
            for e in episodes
            if e.final_distance_to_goal is not None
        ]
        return cls(
            episodes=len(episodes),
            success_rate=success_rate,
            mean_reward=float(np.mean([e.episode_reward for e in episodes])),
            mean_latency_ms=float(np.mean([e.latency_ms for e in episodes])),
            mean_length=float(np.mean([e.episode_length for e in episodes])),
            mean_physics_violations=float(
                np.mean([e.physics_violations for e in episodes])
            ),
            mean_stability_score=float(np.mean([e.stability_score for e in episodes])),
            mean_final_distance=float(np.mean(dists)) if dists else None,
            raw=episodes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "success_rate": self.success_rate,
            "mean_reward": self.mean_reward,
            "mean_latency_ms": self.mean_latency_ms,
            "mean_length": self.mean_length,
            "mean_physics_violations": self.mean_physics_violations,
            "mean_stability_score": self.mean_stability_score,
            "mean_final_distance": self.mean_final_distance,
        }


class MetricCollector:
    """Collect metrics from a ComposedEnvironment during episodes."""

    def __init__(self) -> None:
        self.episodes: list[EpisodeMetrics] = []

    def collect_episode(self, env: Any, max_steps: int = 1000) -> EpisodeMetrics:
        """Run one episode and collect metrics."""
        import numpy as np

        metric = EpisodeMetrics()
        start_time = time.perf_counter()

        obs, info = env.reset()
        episode_reward = 0.0
        step_count = 0

        while step_count < max_steps:
            action_space = env.action_space if hasattr(env, "action_space") else None
            if action_space and isinstance(action_space, dict):
                low = action_space.get("low", -1.0)
                high = action_space.get("high", 1.0)
                shape = action_space.get("shape", (1,))
                action = np.asarray(
                    np.random.uniform(low, high, size=shape), dtype=np.float32
                )
            elif action_space is not None and hasattr(action_space, "sample"):
                action = action_space.sample()
            else:
                action = np.zeros(1, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            step_count += 1

            # Simple stability heuristics based on info
            if info:
                if "physics_violations" in info:
                    metric.physics_violations += info["physics_violations"]
                if "stability_score" in info:
                    metric.stability_score = min(
                        metric.stability_score, info["stability_score"]
                    )

            if terminated or truncated:
                break

        metric.episode_reward = episode_reward
        metric.episode_length = step_count
        metric.latency_ms = (time.perf_counter() - start_time) * 1000
        metric.success = (
            bool(info.get("success", False)) if isinstance(info, dict) else False
        )
        metric.info = info if isinstance(info, dict) else {}

        if "final_distance" in metric.info:
            metric.final_distance_to_goal = metric.info["final_distance"]
        if metric.success and "time_to_success" in metric.info:
            metric.time_to_success = metric.info["time_to_success"]

        self.episodes.append(metric)
        return metric

    def collect_n_episodes(
        self, env: Any, n: int, max_steps: int = 1000
    ) -> MetricSummary:
        """Run N episodes and return a summary."""
        for i in range(n):
            logger.info(f"Collecting episode {i + 1}/{n}")
            self.collect_episode(env, max_steps=max_steps)
        return MetricSummary.from_episodes(self.episodes)

    def reset(self) -> None:
        self.episodes = []

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([e.to_dict() for e in self.episodes], f, indent=2)

    def load(self, path: str | Path) -> None:
        with open(path) as f:
            data = json.load(f)
        self.episodes = [EpisodeMetrics(**d) for d in data]


def default_thresholds() -> dict[str, float]:
    """Default thresholds for diagnostics."""
    return {
        "min_success_rate": 0.8,
        "max_latency_ms": 100.0,
        "max_physics_violations": 5.0,
        "min_stability_score": 0.85,
        "min_mean_reward": -50.0,
    }
