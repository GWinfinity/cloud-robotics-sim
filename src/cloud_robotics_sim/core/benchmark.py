"""Benchmark evaluation protocol and leaderboard aggregation (W8).

This module implements the W8 deliverable of ``docs/ROBODOJO_P0_PLAN.md``:
a seed-controlled evaluation runner that rolls out task configs
(``core/task_loader.py``) over the ``seeds x episodes`` grid and aggregates
results into a leaderboard report.

Reproducibility contract: ``report.json`` contains only deterministic fields
(task / seed / episode / status / steps / reward). Wall-clock durations and
trajectory paths go to ``episodes.jsonl`` and are excluded from the byte-level
reproducibility guarantee, so two runs with the same suite and seeds produce
byte-identical ``report.json``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import yaml

from cloud_robotics_sim.core.config_loader import ConfigError
from cloud_robotics_sim.core.task_loader import LoadedTask, load_task
from cloud_robotics_sim.robotwin.grasp_report import FAIL_STAGES

logger = logging.getLogger(__name__)

STATUS_SUCCESS = "success"


@dataclass
class EpisodeRecord:
    """Outcome of one evaluated episode (GraspRecord generalized to tasks)."""

    task: str
    seed: int
    episode: int
    status: str = "error"  # STATUS_SUCCESS or a FAIL_STAGES label
    failure_stage: str | None = None
    steps: int = 0
    reward: float = 0.0
    duration_s: float = 0.0
    trajectory_path: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize with deterministic key order for JSONL output."""
        return asdict(self)


@dataclass
class EpisodeTrajectory:
    """Per-episode action/reward trace written when trajectory saving is on."""

    actions: list[list[float]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the trajectory for JSON output."""
        return {"actions": self.actions, "rewards": self.rewards}


@dataclass
class SuiteSpec:
    """A benchmark suite: an ordered list of task config paths."""

    name: str
    task_paths: list[Path]
    version: int = 1


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


class BenchmarkPolicy(Protocol):
    """Action source for benchmark rollouts."""

    def act(self, obs: dict) -> np.ndarray:
        """Return the action for the current step."""
        ...


@dataclass
class ZeroPolicy:
    """Hold-command policy (all zeros)."""

    action_dim: int

    def act(self, obs: dict) -> np.ndarray:
        """Return a zero action."""
        return np.zeros(self.action_dim)


@dataclass
class RandomPolicy:
    """Seeded uniform policy in [-1, 1]; deterministic per seed."""

    action_dim: int
    seed: int
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Seed the action RNG."""
        self._rng = np.random.default_rng(self.seed)

    def act(self, obs: dict) -> np.ndarray:
        """Sample a uniform random action."""
        return self._rng.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)


def make_policy(kind: str, action_dim: int, seed: int) -> BenchmarkPolicy:
    """Build a benchmark policy by name."""
    if kind == "zero":
        return ZeroPolicy(action_dim)
    if kind == "random":
        return RandomPolicy(action_dim, seed)
    raise ConfigError(f"unknown policy '{kind}'; available: zero, random")


# ---------------------------------------------------------------------------
# Suite loading
# ---------------------------------------------------------------------------


def load_suite(path: str | Path) -> SuiteSpec:
    """Load a benchmark suite YAML.

    Schema v1::

        version: 1
        name: my-suite
        tasks:
          - ../configs/tasks/pick_place_cube.yaml   # relative to the suite file
    """
    suite_path = Path(path)
    if not suite_path.is_file():
        raise ConfigError(f"suite file not found: {suite_path}")
    try:
        data = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"failed to parse suite {suite_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"suite {suite_path} must be a YAML mapping")

    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise ConfigError(f"suite {suite_path}: 'name' is required")
    raw_tasks = data.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ConfigError(f"suite {suite_path}: 'tasks' must be a non-empty list")

    task_paths: list[Path] = []
    for index, entry in enumerate(raw_tasks):
        if not isinstance(entry, str) or not entry:
            raise ConfigError(
                f"suite {suite_path}: tasks[{index}] must be a path string"
            )
        task_path = Path(entry)
        if not task_path.is_absolute():
            task_path = (suite_path.parent / task_path).resolve()
        if not task_path.is_file():
            raise ConfigError(
                f"suite {suite_path}: tasks[{index}] not found: {task_path}"
            )
        task_paths.append(task_path)
    return SuiteSpec(name=name, task_paths=task_paths)


# ---------------------------------------------------------------------------
# Episode / benchmark execution
# ---------------------------------------------------------------------------


def _episode_status(info: dict) -> tuple[str, str | None]:
    """Map the final step info to (status, failure_stage)."""
    if info.get("success"):
        return STATUS_SUCCESS, None
    stage = info.get("failure_stage")
    if stage in FAIL_STAGES:
        return stage, stage
    return "error", None


def run_episode(
    loaded: LoadedTask,
    seed: int,
    episode: int,
    policy: BenchmarkPolicy,
    *,
    step_cap: int | None = None,
    trajectory: EpisodeTrajectory | None = None,
) -> EpisodeRecord:
    """Roll out one episode: reset with the seed layout, step until done.

    Args:
        loaded: Composed task environment (task_loader.LoadedTask).
        seed: Evaluation seed (drives reset sampling and the policy RNG).
        episode: Episode index within the seed.
        policy: Action source; a fresh policy is assumed per episode.
        step_cap: Safety cap overriding the spec's max_episode_steps.
        trajectory: Optional sink collecting per-step actions and rewards.

    Returns:
        The episode record (deterministic fields only depend on the seed).
    """
    spec = loaded.spec
    max_steps = step_cap or spec.max_episode_steps
    record = EpisodeRecord(task=spec.name, seed=seed, episode=episode)

    start = time.perf_counter()
    obs, info = loaded.reset_episode(seed)
    total_reward = 0.0
    steps = 0
    final_info: dict = info
    terminated = truncated = False

    while not (terminated or truncated) and steps < max_steps:
        action = policy.act(obs)
        obs, reward, terminated, truncated, final_info = loaded.env.step(action)
        if trajectory is not None:
            trajectory.actions.append(
                np.asarray(action, dtype=float).reshape(-1).tolist()
            )
            trajectory.rewards.append(float(reward))
        total_reward += float(reward)
        steps += 1

    record.duration_s = time.perf_counter() - start
    record.steps = steps
    record.reward = total_reward
    if not (terminated or truncated):
        record.status = "error"
        record.failure_stage = None
        logger.warning(
            "episode hit step cap (%d) without termination for task '%s' seed=%d",
            max_steps,
            spec.name,
            seed,
        )
    else:
        record.status, record.failure_stage = _episode_status(final_info)
    return record


def run_benchmark(
    suite: SuiteSpec,
    out_dir: str | Path,
    *,
    policy_kind: str = "zero",
    episodes_limit: int | None = None,
    save_trajectories: bool = False,
    task_factory: Callable[..., LoadedTask] = load_task,
) -> dict[str, Any]:
    """Evaluate every task in the suite and write the leaderboard report.

    Outputs (under ``out_dir``):
        report.json         deterministic leaderboard (byte-reproducible)
        report.md           human-readable leaderboard table
        episodes.jsonl      per-episode records incl. durations/paths
        trajectories/       per-episode action/reward traces (optional)

    Args:
        suite: Loaded benchmark suite.
        out_dir: Report output directory (created if missing).
        policy_kind: ``zero`` (default) or ``random`` (seeded per episode).
        episodes_limit: Optional cap on the number of episodes per task
            (truncates the seeds x episodes grid; smoke testing).
        save_trajectories: Write per-episode action/reward JSON traces and
            record their relative path in ``episodes.jsonl``.
        task_factory: Task loader injection point (tests pass a fake).

    Returns:
        The summary dict as written to report.json.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records: list[EpisodeRecord] = []
    for task_path in suite.task_paths:
        loaded = task_factory(task_path)
        try:
            grid = loaded.evaluation_episodes()
            if episodes_limit is not None:
                grid = grid[:episodes_limit]
            action_dim = int(getattr(loaded.env.robot, "action_dim", 1))
            for seed, episode in grid:
                policy = make_policy(
                    policy_kind, action_dim, seed=seed * 100_000 + episode
                )
                trajectory = EpisodeTrajectory() if save_trajectories else None
                record = run_episode(
                    loaded, seed, episode, policy, trajectory=trajectory
                )
                if trajectory is not None:
                    record.trajectory_path = _write_trajectory(
                        out_path, record, trajectory
                    )
                records.append(record)
                logger.info(
                    "task=%s seed=%d episode=%d -> %s (%d steps)",
                    record.task,
                    seed,
                    episode,
                    record.status,
                    record.steps,
                )
        finally:
            close = getattr(loaded.env, "close", None)
            if callable(close):
                close()

    summary = summarize(records, suite_name=suite.name)
    write_report(summary, records, out_path)
    return summary


def _write_trajectory(
    out_path: Path, record: EpisodeRecord, trajectory: EpisodeTrajectory
) -> str:
    """Write one episode trajectory and return its path relative to out_path."""
    traj_dir = out_path / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    rel_path = Path("trajectories") / (
        f"{record.task}_seed{record.seed}_ep{record.episode}.json"
    )
    payload = {
        "task": record.task,
        "seed": record.seed,
        "episode": record.episode,
        **trajectory.to_json_dict(),
    }
    (out_path / rel_path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # POSIX separators keep reports identical across platforms.
    return rel_path.as_posix()


# ---------------------------------------------------------------------------
# Aggregation / leaderboard
# ---------------------------------------------------------------------------


def summarize(
    records: list[EpisodeRecord], suite_name: str = "benchmark"
) -> dict[str, Any]:
    """Aggregate episode records into a leaderboard structure.

    The output is deterministic: insertion-ordered keys, no wall-clock data.
    """
    per_task: dict[str, dict[str, Any]] = {}
    for record in records:
        entry = per_task.setdefault(
            record.task,
            {
                "episodes": 0,
                "success": 0,
                "failures_by_stage": {stage: 0 for stage in FAIL_STAGES},
                "steps_total": 0,
                "reward_total": 0.0,
            },
        )
        entry["episodes"] += 1
        entry["steps_total"] += record.steps
        entry["reward_total"] += record.reward
        if record.status == STATUS_SUCCESS:
            entry["success"] += 1
        elif record.status in entry["failures_by_stage"]:
            entry["failures_by_stage"][record.status] += 1

    tasks: dict[str, Any] = {}
    for task_name, entry in per_task.items():
        episodes = entry["episodes"]
        failures_by_stage = {
            stage: count
            for stage, count in entry["failures_by_stage"].items()
            if count > 0
        }
        tasks[task_name] = {
            "episodes": episodes,
            "success": entry["success"],
            "success_rate": entry["success"] / episodes if episodes else 0.0,
            "failures_by_stage": failures_by_stage,
            "mean_steps": entry["steps_total"] / episodes if episodes else 0.0,
            "mean_reward": entry["reward_total"] / episodes if episodes else 0.0,
        }

    total = sum(entry["episodes"] for entry in per_task.values())
    successes = sum(entry["success"] for entry in per_task.values())
    return {
        "suite": suite_name,
        "tasks": tasks,
        "overall": {
            "episodes": total,
            "success": successes,
            "success_rate": successes / total if total else 0.0,
        },
    }


def write_report(
    summary: dict[str, Any],
    records: list[EpisodeRecord],
    out_dir: str | Path,
) -> tuple[Path, Path, Path]:
    """Write report.json (deterministic), report.md, and episodes.jsonl."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    report_json = out_path / "report.json"
    report_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    episodes_jsonl = out_path / "episodes.jsonl"
    with episodes_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json_dict(), sort_keys=True) + "\n")

    report_md = out_path / "report.md"
    report_md.write_text(_render_markdown(summary), encoding="utf-8")
    return report_json, report_md, episodes_jsonl


def _render_markdown(summary: dict[str, Any]) -> str:
    """Render the leaderboard as a markdown table."""
    lines = [
        f"# Benchmark report: {summary['suite']}",
        "",
        "| Task | Episodes | Success | Success rate | Mean steps | Mean reward | Top failure |",
        "|---|---|---|---|---|---|---|",
    ]
    for task_name in sorted(summary["tasks"]):
        entry = summary["tasks"][task_name]
        failures = entry["failures_by_stage"]
        top_failure = max(failures, key=failures.get) if failures else "-"
        lines.append(
            f"| {task_name} | {entry['episodes']} | {entry['success']} | "
            f"{entry['success_rate']:.1%} | {entry['mean_steps']:.1f} | "
            f"{entry['mean_reward']:.3f} | {top_failure} |"
        )
    overall = summary["overall"]
    lines += [
        "",
        f"Overall: {overall['success']}/{overall['episodes']} episodes succeeded "
        f"({overall['success_rate']:.1%}).",
        "",
    ]
    return "\n".join(lines)
