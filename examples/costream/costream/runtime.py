"""Stage supervisor / runtime loop."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .behaviors import PredictiveBehavior, ReactiveBehavior, SemanticBehavior
from .composer import ActionComposer
from .controller import CartesianCompliantController, ControllerCompiler
from .math_utils import identity, matrix_to_pos_quat
from .scene_builder import InsertionScene
from .specs import ComposeSpec, SceneSummary, StageSpec


@dataclass
class CoStreamBehaviors:
    semantic: SemanticBehavior
    predictive: PredictiveBehavior
    reactive: ReactiveBehavior


class StageRuntime:
    """Execute a sequence of stages using the three behaviors."""

    def __init__(
        self,
        scene: InsertionScene,
        behaviors: CoStreamBehaviors,
        composer: ActionComposer,
        get_scene_summary: callable,
        dt: float = 0.005,
        sim_rate: int = 1,
    ) -> None:
        self.scene = scene
        self.behaviors = behaviors
        self.composer = composer
        self.get_scene_summary = get_scene_summary
        self.dt = dt
        self.sim_rate = sim_rate
        self.log: list[dict] = []

    def run_stage(
        self,
        stage: StageSpec,
        compose_spec: ComposeSpec,
        controller: CartesianCompliantController,
        task_z_in_world: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run a single stage to completion."""
        self.composer.reset()
        controller.profile = stage.profile

        # Initialize reactive reference with current tool-in-hand pose.
        obj_in_hand = self.scene.robot.object_in_hand_pose()
        self.behaviors.reactive.set_reference(obj_in_hand)

        n_steps = int(round(stage.duration / self.dt))
        stage_log = []
        for step in range(n_steps):
            t_stage = step * self.dt
            scene_summary = self.get_scene_summary()

            # Parallel behavior generation.
            WTI = self.behaviors.semantic.compute_anchor(
                stage, scene_summary, current_ee_pose=self.scene.robot.get_tcp_pose()
            )
            I_T_traj = self.behaviors.predictive.compute_nominal(
                stage, WTI, t_stage
            )

            contact_force = self.scene.robot.get_contact_force()
            observation = {
                "object_in_hand_pose": self.scene.robot.object_in_hand_pose(),
                "contact_force": contact_force,
            }
            T_tact, reactive_info = self.behaviors.reactive.compute_residual(
                stage,
                observation,
                task_z_in_world=task_z_in_world,
            )

            # Compose.
            W_T_cmd = self.composer.compose(
                WTI, I_T_traj, T_tact, compose_spec
            )

            # Execute compliant controller.
            q_target, ctrl_info = controller.step(
                W_T_cmd,
                contact_force,
                task_z_in_world=task_z_in_world,
            )

            # Update kinematic tool visualization / contact probe.
            self.scene.robot.update_tool_pose()

            # Step physics.
            for _ in range(self.sim_rate):
                self.scene.scene.step()

            # Diagnostics.
            entry = {
                "step": step,
                "t_stage": t_stage,
                "stage": stage.name,
                "W_T_cmd_pos": W_T_cmd[:3, 3].copy(),
                "contact_force": contact_force.copy(),
                **reactive_info,
                **ctrl_info,
            }
            stage_log.append(entry)

            # Simple guard checks.
            if self._check_recovery(stage, contact_force):
                return {
                    "success": False,
                    "reason": "recovery_triggered",
                    "log": stage_log,
                }

        self.log.extend(stage_log)
        return {"success": True, "log": stage_log}

    @staticmethod
    def _check_recovery(stage: StageSpec, contact_force: np.ndarray) -> bool:
        limits = stage.profile.max_force[:3]
        return bool(np.any(np.abs(contact_force) > limits))


class CoStreamRuntime:
    """High-level runner over a list of (stage, compose_spec) pairs."""

    def __init__(
        self,
        scene: InsertionScene,
        behaviors: CoStreamBehaviors,
        get_scene_summary: callable,
        dt: float = 0.005,
    ) -> None:
        self.scene = scene
        self.behaviors = behaviors
        self.get_scene_summary = get_scene_summary
        self.dt = dt
        self.composer = ActionComposer()
        self.stage_runtime = StageRuntime(
            scene=scene,
            behaviors=behaviors,
            composer=self.composer,
            get_scene_summary=get_scene_summary,
            dt=dt,
        )

    def run(
        self,
        stages: list[tuple[StageSpec, ComposeSpec]],
        controller: CartesianCompliantController,
        task_z_in_world: np.ndarray | None = None,
    ) -> dict[str, Any]:
        results = []
        for stage, compose_spec in stages:
            print(f"[CoStream] Stage: {stage.name} ({stage.duration}s)")
            result = self.stage_runtime.run_stage(
                stage,
                compose_spec,
                controller,
                task_z_in_world=task_z_in_world,
            )
            results.append((stage.name, result))
            if not result["success"]:
                print(f"[CoStream] Stage {stage.name} failed: {result['reason']}")
                break
            print(f"[CoStream] Stage {stage.name} completed")
        return {"results": results, "log": self.stage_runtime.log}
