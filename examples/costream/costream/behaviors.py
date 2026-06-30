"""Three core behaviors: semantic, predictive, reactive."""

from __future__ import annotations

import numpy as np

from .math_utils import (
    clip_translation,
    compose,
    identity,
    interpolate,
    matrix_to_pos_quat,
    pos_quat_to_matrix,
    translation_matrix,
)
from .specs import ComposeSpec, SceneSummary, StageSpec


class SemanticBehavior:
    """Produces a task-frame anchor WTI from scene summary.

    In the real system this involves an LLM/VLM parsing instructions into
    object-centric geometric constraints.  Here we use ground-truth object
    poses to emulate the result of that parsing.
    """

    def compute_anchor(
        self,
        stage: StageSpec,
        scene: SceneSummary,
        current_ee_pose: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return WTI for the active stage."""
        if stage.task_frame not in scene.objects:
            # Fallback: identity at world origin.
            return identity()

        obj = scene.objects[stage.task_frame]
        return pos_quat_to_matrix(obj.pos, obj.quat)


class PredictiveBehavior:
    """Produces nominal motion in the task frame.

    The paper uses a video world model + VLM critic + 3D keypoint tracker.
    Here we provide a parametric motion library as a drop-in replacement.
    """

    def __init__(self) -> None:
        self._stage_start: dict[str, np.ndarray] = {}
        self._stage_goal: dict[str, np.ndarray] = {}

    def compute_nominal(
        self,
        stage: StageSpec,
        WTI: np.ndarray,
        t_stage: float,
    ) -> np.ndarray:
        """Return I_T_traj(t) in the task frame.

        Args:
            stage: active stage spec.
            WTI: task-frame anchor.
            t_stage: elapsed time within the stage (seconds).
        """
        alpha = float(np.clip(t_stage / max(stage.duration, 1e-6), 0.0, 1.0))
        T_start = stage.relative_start
        T_goal = stage.relative_goal
        return interpolate(T_start, T_goal, alpha)


class ReactiveBehavior:
    """High-rate tactile/force corrections.

    The real system uses GelSight Mini NormalFlow for in-hand slip and the
    robot F/T sensor for contact regulation.  In simulation we approximate:

    - in-hand slip: compare current object-in-hand pose to reference.
    - contact: use net contact force on the tool to produce lateral compliance
      and detect axial overload.
    """

    def __init__(
        self,
        tactile_gain: float = 0.8,
        force_gain: float = 2e-4,
        max_lateral_correction: float = 0.015,
        max_axial_deflection: float = 0.01,
        contact_force_threshold: float = 2.0,
    ) -> None:
        self.tactile_gain = tactile_gain
        self.force_gain = force_gain
        self.max_lateral_correction = max_lateral_correction
        self.max_axial_deflection = max_axial_deflection
        self.contact_force_threshold = contact_force_threshold

        self.ref_in_hand: np.ndarray | None = None

    def set_reference(self, object_in_hand_pose: np.ndarray) -> None:
        """Call once a stable grasp/reference is established."""
        self.ref_in_hand = np.asarray(object_in_hand_pose, dtype=float).copy()

    def compute_residual(
        self,
        stage: StageSpec,
        observation: dict,
        task_z_in_world: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Return tactile residual transform and diagnostic dict.

        observation keys:
            - 'object_in_hand_pose': 4x4 transform of object relative to hand.
            - 'contact_force': 3-vector of net contact force in world frame.
        """
        T_tact = identity()
        info: dict = {"slip_detected": False, "contact": False}

        # --- Tactile slip correction (object-in-hand pose drift) ---
        obj_pose = observation.get("object_in_hand_pose")
        if obj_pose is not None and self.ref_in_hand is not None:
            obj_pose = np.asarray(obj_pose, dtype=float)
            # Measured drift = current relative pose - reference relative pose.
            drift = compose(obj_pose, np.linalg.inv(self.ref_in_hand))
            # We want to cancel the drift, so apply inverse.
            correction = np.linalg.inv(drift)
            correction[:3, :3] = np.eye(3)  # only translate for slip correction
            correction = clip_translation(correction, self.max_lateral_correction)
            # Scale correction by gain.
            correction[:3, 3] *= self.tactile_gain
            T_tact = correction
            slip_norm = float(np.linalg.norm(correction[:3, 3]))
            info["slip_detected"] = slip_norm > 1e-5
            info["slip_correction"] = correction[:3, 3].copy()

        # --- Force-based contact correction ---
        force_world = np.asarray(observation.get("contact_force", np.zeros(3)))
        info["contact_force"] = force_world.copy()
        force_norm = float(np.linalg.norm(force_world))
        info["contact_force_norm"] = force_norm
        info["contact"] = force_norm > self.contact_force_threshold

        if info["contact"]:
            # Determine task insertion axis (default world -z).
            if task_z_in_world is None:
                task_z_in_world = np.array([0.0, 0.0, -1.0])
            task_z = task_z_in_world / (np.linalg.norm(task_z_in_world) + 1e-9)

            # Axial component: used by controller to slow insertion.
            f_axial = float(np.dot(force_world, task_z))
            info["axial_force"] = f_axial

            # Lateral component: produce a corrective translation opposite to it.
            f_lateral = force_world - f_axial * task_z
            lateral_norm = float(np.linalg.norm(f_lateral))
            info["lateral_force"] = f_lateral.copy()
            info["lateral_force_norm"] = lateral_norm

            if lateral_norm > 0.2:
                # Move away from the contact force.
                delta = -self.force_gain * f_lateral
                delta_norm = float(np.linalg.norm(delta))
                if delta_norm > self.max_lateral_correction:
                    delta = delta * (self.max_lateral_correction / delta_norm)
                # Right-multiply a small translation.
                T_force = translation_matrix(delta)
                T_tact = compose(T_tact, T_force)
                info["lateral_correction"] = delta.copy()

            # Axial compliance: push back along insertion axis if overloaded.
            if abs(f_axial) > self.contact_force_threshold:
                axial_delta = -np.sign(f_axial) * min(
                    self.max_axial_deflection,
                    self.force_gain * 0.5 * abs(f_axial),
                )
                T_axial = translation_matrix(axial_delta * task_z)
                T_tact = compose(T_tact, T_axial)
                info["axial_correction"] = axial_delta * task_z

        return T_tact, info
