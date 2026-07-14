"""Main continuous improvement loop for Genesis simulation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from cloud_robotics_sim.core.embodiment import EmbodimentConfig, FrankaPanda
from cloud_robotics_sim.core.registry import (
    register_robot,
    register_scene,
    register_task,
)
from cloud_robotics_sim.core.scenes import EmptyRoom
from cloud_robotics_sim.core.task import PickPlaceTask, TaskConfig
from cloud_robotics_sim.utils import device as device_utils
from cloud_robotics_sim.utils import genesis_compat

from .diagnostics import DiagnosticEngine
from .experiments import ExperimentValidator, ValidationResult
from .knowledge_base import KnowledgeBase
from .metrics import MetricCollector
from .proposals import ProposalGenerator


@register_scene("empty_room")
def _make_empty_room(**kwargs: Any) -> Any:
    kwargs.pop("type", None)
    return EmptyRoom(**kwargs)


@register_robot("franka_panda")
def _make_franka_panda(**kwargs: Any) -> Any:
    kwargs.pop("type", None)
    base_position = kwargs.pop("base_position", (0.0, 0.0, 0.0))
    model_path = kwargs.pop("urdf_path", None) or kwargs.pop("model_path", None)
    return FrankaPanda(
        EmbodimentConfig(
            name="franka",
            base_position=base_position,
            urdf_path=model_path,
        )
    )


@register_task("pick_place")
def _make_pick_place(**kwargs: Any) -> Any:
    kwargs.pop("type", None)
    object_name = kwargs.pop("object_name", "red_cube")
    target_position = kwargs.pop("target_position", (0.5, 0.0, 0.05))
    return PickPlaceTask(
        TaskConfig(max_episode_steps=200),
        object_name=object_name,
        target_position=target_position,
    )


logger = logging.getLogger(__name__)


@dataclass
class LoopConfig:
    """Configuration for the improvement loop."""

    baseline_config_path: str
    output_dir: str = "./outputs/improvement_loop"
    n_baseline_episodes: int = 30
    n_validation_episodes: int = 50
    max_iterations: int = 10
    min_success_rate_delta: float = 0.05
    min_reward_delta: float = 5.0
    max_latency_regression: float = 10.0
    thresholds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_config_path": self.baseline_config_path,
            "output_dir": self.output_dir,
            "n_baseline_episodes": self.n_baseline_episodes,
            "n_validation_episodes": self.n_validation_episodes,
            "max_iterations": self.max_iterations,
            "min_success_rate_delta": self.min_success_rate_delta,
            "min_reward_delta": self.min_reward_delta,
            "max_latency_regression": self.max_latency_regression,
            "thresholds": self.thresholds,
        }


class ImprovementLoop:
    """Continuous improvement loop for Genesis simulation."""

    def __init__(
        self,
        make_env_fn: Callable[[dict[str, Any]], Any],
        loop_config: LoopConfig,
    ) -> None:
        self.make_env_fn = make_env_fn
        self.loop_config = loop_config

        self.output_dir = Path(loop_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.diagnostic_engine = DiagnosticEngine(loop_config.thresholds)
        self.proposal_generator = ProposalGenerator()
        self.validator = ExperimentValidator(
            make_env_fn=make_env_fn,
            n_episodes=loop_config.n_validation_episodes,
            min_success_rate_delta=loop_config.min_success_rate_delta,
            min_reward_delta=loop_config.min_reward_delta,
            max_latency_regression=loop_config.max_latency_regression,
        )
        self.knowledge_base = KnowledgeBase(
            path=self.output_dir / "knowledge_base.json"
        )

        self.baseline_config = self._load_config(loop_config.baseline_config_path)
        self.history: list[dict[str, Any]] = []

    def _load_config(self, path: str) -> dict[str, Any]:
        with open(path) as f:
            config = yaml.safe_load(f)
            if not isinstance(config, dict):
                raise ValueError(f"Expected YAML mapping in {path}")
            return config

    def _save_config(self, config: dict[str, Any], path: Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    def run(self) -> dict[str, Any]:
        """Run the continuous improvement loop."""
        logger.info("Starting Genesis simulation improvement loop")
        best_config = self.baseline_config

        for iteration in range(1, self.loop_config.max_iterations + 1):
            logger.info(
                f"=== Iteration {iteration}/{self.loop_config.max_iterations} ==="
            )

            # 1. Observe
            env = self.make_env_fn(best_config)
            collector = MetricCollector()
            summary = collector.collect_n_episodes(
                env, n=self.loop_config.n_baseline_episodes
            )
            logger.info(f"Baseline summary: {summary.to_dict()}")

            # 2. Diagnose
            diagnoses = self.diagnostic_engine.diagnose(summary)
            worst = self.diagnostic_engine.worst_problem(summary)
            if worst is None or worst.category == "healthy":
                logger.info("No critical problems found. Loop converged.")
                break

            logger.info(f"Worst problem: {worst.category} - {worst.message}")

            # 3. Propose
            proposals = self.proposal_generator.generate(worst, best_config)
            # Filter out proposals that have repeatedly failed
            proposals = [
                p
                for p in proposals
                if not self.knowledge_base.has_failed(p, min_attempts=2)
            ]
            if not proposals:
                logger.info("No fresh proposals left for this diagnosis. Stopping.")
                break

            # 4. Validate
            results = self.validator.validate_all(best_config, proposals)
            for result in results:
                self.knowledge_base.record(result)
                logger.info(
                    f"Proposal {result.proposal.name}: passed={result.passed}, "
                    f"confidence={result.confidence:.2f}, reason={result.reason}"
                )

            # 5. Adopt
            best_result = self._select_best(results)
            if best_result and best_result.passed:
                best_config = self.validator._apply_delta(
                    best_config, best_result.proposal.config_delta
                )
                logger.info(f"Adopted proposal: {best_result.proposal.name}")
            else:
                logger.info("No passing proposal this iteration.")

            # 6. Learn & record
            iteration_record = {
                "iteration": iteration,
                "summary": summary.to_dict(),
                "diagnoses": [d.to_dict() for d in diagnoses],
                "best_proposal": (
                    best_result.proposal.to_dict() if best_result else None
                ),
                "best_result": best_result.to_dict() if best_result else None,
                "knowledge_base_summary": self.knowledge_base.summary(),
            }
            self.history.append(iteration_record)
            self._save_artifacts(iteration, best_config, iteration_record)

        final_report = self._generate_final_report(best_config)
        self._save_artifacts(iteration="final", config=best_config, record=final_report)
        logger.info("Improvement loop finished")
        return final_report

    def _select_best(self, results: list[ValidationResult]) -> ValidationResult | None:
        """Select the best passing result, or None if none pass."""
        passing = [r for r in results if r.passed]
        if passing:
            return passing[0]
        return results[0] if results else None

    def _save_artifacts(
        self, iteration: int | str, config: dict[str, Any], record: dict[str, Any]
    ) -> None:
        iter_dir = self.output_dir / f"iter_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        self._save_config(config, iter_dir / "config.yaml")

        # Save record
        with open(iter_dir / "record.json", "w") as f:
            json.dump(record, f, indent=2)

        # Save markdown report for regular iterations only
        if isinstance(iteration, int):
            with open(iter_dir / "report.md", "w") as f:
                f.write(self._iteration_markdown(record))

    def _iteration_markdown(self, record: dict[str, Any]) -> str:
        lines = [
            f"# Iteration {record['iteration']}",
            "",
            "## Baseline Summary",
            "",
            f"- Episodes: {record['summary']['episodes']}",
            f"- Success rate: {record['summary']['success_rate']:.1%}",
            f"- Mean reward: {record['summary']['mean_reward']:.2f}",
            f"- Mean latency: {record['summary']['mean_latency_ms']:.2f}ms",
            f"- Physics violations: {record['summary']['mean_physics_violations']:.2f}",
            f"- Stability score: {record['summary']['mean_stability_score']:.2f}",
            "",
            "## Diagnoses",
            "",
        ]
        for d in record["diagnoses"]:
            lines.append(f"- **{d['category']}** ({d['severity']}): {d['message']}")
            lines.append(f"  - Recommendation: {d['recommendation']}")

        lines.extend(["", "## Best Proposal", ""])
        if record["best_proposal"]:
            p = record["best_proposal"]
            lines.append(f"- Name: {p['name']}")
            lines.append(f"- Description: {p['description']}")
            lines.append(f"- Passed: {record['best_result']['passed']}")
            lines.append(f"- Confidence: {record['best_result']['confidence']:.2f}")
            lines.append(f"- Reason: {record['best_result']['reason']}")
        else:
            lines.append("No proposal adopted.")

        return "\n".join(lines)

    def _generate_final_report(self, best_config: dict[str, Any]) -> dict[str, Any]:
        return {
            "final_config": best_config,
            "history": self.history,
            "knowledge_base_summary": self.knowledge_base.summary(),
            "total_iterations": len(self.history),
            "output_dir": str(self.output_dir),
        }


# Example make_env_fn for cloud_robotics_sim
_gs_initialized = False


def make_env_from_config(config: dict[str, Any]) -> Any:
    """Create a ComposedEnvironment from a loop config dictionary."""
    from cloud_robotics_sim.core.composer import ComposerConfig, EnvironmentComposer
    from cloud_robotics_sim.core.registry import default_registry

    global _gs_initialized
    if not _gs_initialized:
        import torch

        device = device_utils.get_device(config.get("device"))
        if config.get("device") is None:
            # Legacy configs use use_cuda instead of device.
            use_cuda = config.get("use_cuda", torch.cuda.is_available())
            device = device_utils.get_device("cuda" if use_cuda else "cpu")

        device_utils.set_default_device(device)
        genesis_compat.genesis_init(headless=True, device=device)
        _gs_initialized = True
        # Subsequent calls to compose() also try to initialize Genesis. Since
        # Genesis can only be initialized once per process, make the helper
        # idempotent for the lifetime of this process.
        genesis_compat.ensure_genesis_initialized = lambda **kwargs: None

    env_cfg = config["environment"]
    composer = EnvironmentComposer(
        ComposerConfig(
            dt=env_cfg["simulation"]["dt"],
            substeps=env_cfg["simulation"]["substeps"],
            headless=env_cfg["simulation"].get("headless", True),
            resolution=tuple(env_cfg["simulation"]["resolution"]),
        )
    )

    scene = default_registry().create_scene(
        env_cfg["scene"]["type"], **env_cfg["scene"]
    )
    robot = default_registry().create_robot(
        env_cfg["robot"]["type"], **env_cfg["robot"]
    )
    task = default_registry().create_task(env_cfg["task"]["type"], **env_cfg["task"])
    return composer.compose(scene, robot, task)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loop = ImprovementLoop(
        make_env_fn=make_env_from_config,
        loop_config=LoopConfig(
            baseline_config_path="configs/franka_pickplace.yaml",
            output_dir="./outputs/improvement_loop",
            n_baseline_episodes=10,
            n_validation_episodes=15,
            max_iterations=3,
        ),
    )
    report = loop.run()
    print(json.dumps(report["knowledge_base_summary"], indent=2))
