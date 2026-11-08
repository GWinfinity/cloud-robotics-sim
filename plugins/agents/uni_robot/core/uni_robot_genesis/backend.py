"""Genesis 后端：把 BuiltScene 包装成 uni-robot-agent 的 EnvironmentBackend。

协议语义（与 robodojo adapter 对齐）：
- 世界坐标单位米，四元数 wxyz；control_pose 是注册 grasp site（夹爪中心）
  的世界位姿。
- 一个 primitive step = 一次 IK/关节目标下发 + 一个控制周期的物理步进；
  多步运动循环由共享工具层拥有，这里不实现任何模型级语义。
- 坐标系约定：Genesis 相机为 OpenGL 约定（z 向后）；协议要求光学约定
  （z 向前），capture() 里显式做 world-from-cam -> world-from-optical
  的列变换（与 genesis Camera.extrinsics 同一约定，见源码注释）。
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from uni_robot_agent.services.environment.contracts import (
    ActionCommand,
    ArmCommand,
    ArmDescription,
    ArmState,
    CameraDescription,
    EnvironmentDescription,
    EnvironmentState,
    JointCommand,
    JointTarget,
    NativeCommand,
    Pose,
    PoseCommand,
    RequestRejected,
    StepReceipt,
)
from uni_robot_agent.services.environment.snapshots import (
    CameraFrame,
    CapturedObservation,
)

from .built import BuiltArm, BuiltScene
from .scene_spec import SceneSpec

_GRIPPER_NAME_HINTS = ("finger", "gripper", "grip")
_EEF_LINK_CANDIDATES = (
    "panda_hand",
    "panda_link8",
    "ee_link",
    "link_ee",
    "wrist_3_link",
    "link7",
    "link6",
    "tool0",
)


def _as_vec(value: Any, size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != size:
        raise ValueError(f"expected {size} values, got {arr.shape[0]}")
    return arr


def _to_numpy(value: Any) -> np.ndarray:
    """Torch 张量 / numpy / 序列统一转 float64 ndarray（避免 numpy2 的 copy 告警）。"""
    detach = getattr(value, "detach", None)
    if callable(detach):  # torch.Tensor
        return np.asarray(detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _link_name(link: Any) -> str:
    """RigidLink 只有 ``.name``；fake/其他实现可能只有 ``get_name()``。"""
    name = getattr(link, "name", None)
    if name:
        return str(name)
    get_name = getattr(link, "get_name", None)
    if callable(get_name):
        return str(get_name())
    raise TypeError(f"link object {type(link).__name__} exposes no name")


def _quaternion_distance(left: Any, right: Any) -> float:
    a = _as_vec(left, 4)
    b = _as_vec(right, 4)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return 2.0 * math.acos(float(np.clip(abs(np.dot(a, b)), 0.0, 1.0)))


class GenesisEnvironmentBackend:
    """单 episode 的原生边界。所有方法必须在 owner 线程调用（host 保证）。

    只依赖 BuiltScene 的 duck-typed 句柄，测试可注入 fake。
    """

    def __init__(self, built: BuiltScene, *, episode_id: str | None = None) -> None:
        self._built = built
        self._spec: SceneSpec = built.spec
        self._control = self._spec.control
        self.episode_id = episode_id or self._spec.episode_id
        self.description = self._build_description()
        self._steps = 0
        self._started = False
        self._sealed = False
        self._closed = False
        self._gripper_commands: dict[str, float] = {}
        self._outcome: dict[str, Any] | None = None

    # -- 生命周期 ------------------------------------------------------------

    def start(self) -> EnvironmentState:
        """准备 episode（场景已由 builder 构建好；不再重置）。"""
        if self._started:
            raise RequestRejected("Native initialization was already attempted")
        if self._closed:
            raise RequestRejected("Native episode is closed")
        self._started = True
        for name, arm in self._built.arms.items():
            qpos = self._qpos(arm)
            self._gripper_commands[name] = self._read_gripper(arm, qpos)
        return self.read_state()

    # -- 描述 ----------------------------------------------------------------

    def _build_description(self) -> EnvironmentDescription:
        arms = [
            ArmDescription(
                name=arm.name,
                control_frame=f"{arm.name}/{_link_name(arm.grasp_link)}",
                grasp_site_description=arm.grasp_site_description,
                rotation_radius_m=arm.rotation_radius_m,
                joint_names=list(arm.joint_names),
                joint_tracking_tolerance_rad=arm.joint_tracking_tolerance_rad,
            )
            for arm in self._built.arms.values()
        ]
        cameras = [
            CameraDescription(
                name=meta.name,
                role=meta.role,
                has_depth=meta.provide_depth,
                has_calibration=True,
            )
            for meta in self._built.camera_meta.values()
        ]
        return EnvironmentDescription(
            arms=arms,
            cameras=cameras,
            control_frequency_hz=self._control.frequency_hz,
            max_episode_steps=self._control.max_episode_steps,
            action_binding=None,
            metadata={
                "instruction": self._spec.instruction,
                "integration": "genesis",
                **self._spec.metadata,
            },
        )

    # -- 状态读取 ------------------------------------------------------------

    def _qpos(self, arm: BuiltArm) -> np.ndarray:
        return _to_numpy(arm.entity.get_qpos())

    def _read_gripper(self, arm: BuiltArm, qpos: np.ndarray) -> float:
        lower, upper = arm.gripper_limits
        if upper - lower < 1e-12:
            return 0.0
        value = float(qpos[arm.primary_gripper_dof])
        return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))

    def _arm_state(self, arm: BuiltArm) -> ArmState:
        qpos = self._qpos(arm)
        # relative=False：协议 control_pose 必须为世界系位姿
        position = _as_vec(_to_numpy(arm.grasp_link.get_pos(relative=False)), 3)
        quat = _as_vec(_to_numpy(arm.grasp_link.get_quat(relative=False)), 4)
        joints = [float(qpos[idx]) for idx in arm.arm_dofs]
        return ArmState(
            control_pose=Pose(position=tuple(position), quaternion=tuple(quat)),
            joints=joints,
            gripper=self._read_gripper(arm, qpos),
            gripper_command=float(self._gripper_commands.get(arm.name, 0.0)),
            metadata={
                "gripper_applied_command": float(qpos[arm.primary_gripper_dof]),
            },
        )

    def read_state(self) -> EnvironmentState:
        if not self._started:
            raise RequestRejected("Native episode has not started")
        arms = {name: self._arm_state(a) for name, a in self._built.arms.items()}
        return EnvironmentState(
            arms=arms,
            state_version=self._steps,
            native_steps=self._steps,
            ended=self._sealed,
            metadata={
                "instruction": self._spec.instruction,
                "remaining_steps": max(
                    0, self._control.max_episode_steps - self._steps
                ),
            },
        )

    # -- 校验 ----------------------------------------------------------------

    def _active(self) -> None:
        if not self._started or self._sealed or self._closed:
            raise RequestRejected("Native episode is not active")
        if self._steps >= self._control.max_episode_steps:
            raise RequestRejected("Episode step limit reached")

    def _require_all_arms(self, command: PoseCommand | JointCommand) -> None:
        if set(command.arms) != set(self._built.arms):
            raise RequestRejected(
                "A control step must contain all arms exactly once: "
                f"expected {sorted(self._built.arms)}"
            )

    def _check_joint_target(self, arm: BuiltArm, target: JointTarget) -> None:
        values = np.asarray(target.joints, dtype=np.float64)
        limits = arm.joint_limits
        if values.shape != (len(arm.arm_dofs),) or not np.isfinite(values).all():
            raise RequestRejected(f"Invalid {arm.name} joint target")
        if np.any(values < limits[:, 0] - 1e-5) or np.any(values > limits[:, 1] + 1e-5):
            raise RequestRejected(f"{arm.name} joint target exceeds limits")

    def validate_commands(self, commands: list[ActionCommand], mode: str) -> None:
        self._active()
        measured = self.read_state()
        previous = {
            name: np.asarray(arm.joints, dtype=np.float64)
            for name, arm in measured.arms.items()
        }
        for command in commands:
            if isinstance(command, NativeCommand):
                raise RequestRejected(
                    "Genesis adapter does not accept opaque native actions"
                )
            self._require_all_arms(command)
            for name, target in command.arms.items():
                arm = self._built.arms[name]
                if isinstance(target, JointTarget):
                    self._check_joint_target(arm, target)
                    values = np.asarray(target.joints, dtype=np.float64)
                    if (
                        np.max(np.abs(values - previous[name]))
                        > self._control.max_joint_jump_rad
                    ):
                        raise RequestRejected(
                            f"{name} action prefix has a joint jump over "
                            f"{self._control.max_joint_jump_rad} rad"
                        )
                    previous[name] = values
                elif mode == "eef":
                    pose = measured.arms[name].control_pose
                    displacement = float(
                        np.linalg.norm(
                            _as_vec(target.pose.position, 3) - _as_vec(pose.position, 3)
                        )
                    )
                    rotation = _quaternion_distance(
                        target.pose.quaternion, pose.quaternion
                    )
                    if (
                        displacement > self._control.max_step_translation_m + 1e-10
                        or rotation > self._control.max_step_rotation_rad + 1e-10
                    ):
                        raise RequestRejected(
                            f"{name} EEF target exceeds single-step bounds "
                            f"({self._control.max_step_translation_m} m / "
                            f"{self._control.max_step_rotation_rad} rad)"
                        )

    # -- IK 与执行 -----------------------------------------------------------

    def _solve_ik(
        self, arm: BuiltArm, target: ArmCommand
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """返回 (arm_qpos, diagnostics)；失败时 (None, 原因)。"""
        try:
            # init_qpos 不传：genesis 内部按当前状态自预热，且对
            # 单环境/批量的形状处理更稳（显式 (n,) 会在 envs_idx 校验处报错）。
            solved, error = arm.entity.inverse_kinematics(
                link=arm.grasp_link,
                pos=_as_vec(target.pose.position, 3),
                quat=_as_vec(target.pose.quaternion, 4),
                dofs_idx_local=list(arm.arm_dofs),
                return_error=True,
            )
        except Exception as error:  # noqa: BLE001 - 归一化为协议内的拒绝
            return None, {"reason": "ik_solver_error", "message": str(error)}
        solved = np.asarray(solved, dtype=np.float64).reshape(-1)
        # genesis IK 返回全量 qpos（n_dofs），不是所选 dof 的子向量。
        if solved.shape[0] != len(arm.arm_dofs):
            entity_n = self._qpos(arm).shape[0]
            if solved.shape[0] == entity_n:
                solved = solved[list(arm.arm_dofs)]
            else:
                return None, {
                    "reason": "ik_bad_solution_shape",
                    "solution_dim": int(solved.shape[0]),
                }
        error = np.asarray(error, dtype=np.float64).reshape(-1)
        pos_err = float(np.linalg.norm(error[:3]))
        rot_err = float(np.linalg.norm(error[3:]))
        if (
            pos_err > self._control.ik_pos_tolerance_m
            or rot_err > self._control.ik_rot_tolerance_rad
        ):
            return None, {
                "reason": "ik_not_converged",
                "position_error_m": pos_err,
                "rotation_error_rad": rot_err,
            }
        return solved, {
            "ik_position_error_m": pos_err,
            "ik_rotation_error_rad": rot_err,
        }

    def _gripper_to_joint(
        self, arm: BuiltArm, opening: float, qpos: np.ndarray
    ) -> np.ndarray:
        del qpos  # 所有夹爪 dof 共用主 dof 的线性跟随模型
        opening = float(np.clip(opening, 0.0, 1.0))
        lower, upper = arm.gripper_limits
        primary = lower + opening * (upper - lower)
        # 通用假设：所有夹爪 dof 以相同关节值跟随主 dof（franka 双指
        # prismatic 即如此，空间反向由 URDF 轴向表达）。特殊手爪请通过
        # ArmSpec.gripper 配置限位/主 dof，或在实体驱动层自行耦合。
        values = np.full(len(arm.gripper_dofs), primary, dtype=np.float64)
        return values

    def _apply(self, targets: dict[str, tuple[np.ndarray, float]]) -> None:
        # genesis 1.4.x 的 tensor_assigned 需要 torch 张量（list/numpy
        # 都会在其内部 `tensor[indices] = value` 处抛 TypeError）。
        import torch

        for name, (arm_q, gripper) in targets.items():
            arm = self._built.arms[name]
            grip_q = self._gripper_to_joint(arm, gripper, self._qpos(arm))
            values = np.concatenate([arm_q, grip_q])
            dofs = list(arm.arm_dofs) + list(arm.gripper_dofs)
            arm.entity.control_dofs_position(
                torch.as_tensor(values, dtype=torch.float32),
                dofs_idx_local=dofs,
            )
        for _ in range(self._built.ticks_per_control):
            self._built.scene.step()
        self._steps += 1
        for name, (_, gripper) in targets.items():
            self._gripper_commands[name] = float(np.clip(gripper, 0.0, 1.0))

    # -- 协议方法 ------------------------------------------------------------

    def step_pose(self, command: PoseCommand) -> StepReceipt:
        self._active()
        self._require_all_arms(command)
        diagnostics: dict[str, Any] = {"solver": command.solver}
        targets: dict[str, tuple[np.ndarray, float]] = {}
        for name, target in command.arms.items():
            arm = self._built.arms[name]
            if isinstance(target, JointTarget):
                self._check_joint_target(arm, target)
                arm_q = np.asarray(target.joints, dtype=np.float64)
                targets[name] = (arm_q, target.gripper)
            else:
                arm_q, ik_diag = self._solve_ik(arm, target)
                if arm_q is None:
                    return StepReceipt(
                        accepted=False,
                        state=self.read_state(),
                        diagnostics={**diagnostics, "arm": name, **ik_diag},
                    )
                targets[name] = (arm_q, target.gripper)
                diagnostics[f"{name}_ik"] = ik_diag
        self._apply(targets)
        diagnostics["commanded_joints"] = {
            name: values[0].tolist() for name, values in targets.items()
        }
        return StepReceipt(
            accepted=True, state=self.read_state(), diagnostics=diagnostics
        )

    def step_joints(self, command: JointCommand) -> StepReceipt:
        self._active()
        self._require_all_arms(command)
        targets: dict[str, tuple[np.ndarray, float]] = {}
        for name, target in command.arms.items():
            arm = self._built.arms[name]
            self._check_joint_target(arm, target)
            targets[name] = (
                np.asarray(target.joints, dtype=np.float64),
                target.gripper,
            )
        self._apply(targets)
        return StepReceipt(
            accepted=True,
            state=self.read_state(),
            diagnostics={
                "commanded_joints": {
                    name: values[0].tolist() for name, values in targets.items()
                }
            },
        )

    def step_native(self, command: NativeCommand) -> StepReceipt:
        raise RequestRejected("Genesis adapter does not accept opaque native actions")

    def preview_joints(self, commands: list[JointCommand]) -> dict[str, Any]:
        """无状态预览：做限位/跳变检查；不改动物理状态。

        Genesis 不暴露免状态切换的单臂 FK 预览接口，因此这里返回的是
        可执行性诊断而不是假想位姿（协议只要求返回 dict）。
        """
        self._active()
        results = []
        measured = self.read_state()
        previous = {
            name: np.asarray(arm.joints, dtype=np.float64)
            for name, arm in measured.arms.items()
        }
        for index, command in enumerate(commands):
            self._require_all_arms(command)
            row: dict[str, Any] = {"index": index, "arms": {}}
            for name, target in command.arms.items():
                arm = self._built.arms[name]
                values = np.asarray(target.joints, dtype=np.float64)
                limits = arm.joint_limits
                within = bool(
                    values.shape == (len(arm.arm_dofs),)
                    and np.isfinite(values).all()
                    and np.all(values >= limits[:, 0] - 1e-5)
                    and np.all(values <= limits[:, 1] + 1e-5)
                )
                jump = (
                    float(np.max(np.abs(values - previous[name])))
                    if values.shape == previous[name].shape
                    else None
                )
                row["arms"][name] = {
                    "within_limits": within,
                    "joint_jump_from_previous_rad": jump,
                }
                if within:
                    previous[name] = values
            results.append(row)
        return {"previews": results, "note": "limit diagnostics only; no state changed"}

    def capture(
        self, cameras: list[str] | None = None, *, high_resolution: bool = False
    ) -> CapturedObservation:
        del high_resolution  # Genesis 相机为固定分辨率；高低分辨率同图
        if not self._started:
            raise RequestRejected("Native episode has not started")
        available = set(self._built.cameras)
        selected = list(self._built.cameras) if cameras is None else list(cameras)
        if (
            not selected
            or len(set(selected)) != len(selected)
            or not set(selected) <= available
        ):
            raise RequestRejected("cameras must name distinct supported views")
        frames: dict[str, CameraFrame] = {}
        for name in selected:
            camera = self._built.cameras[name]
            meta = self._built.camera_meta[name]
            buffers = camera.render(rgb=True, depth=meta.provide_depth)
            rgb = np.asarray(buffers[0])
            if rgb.ndim != 3 or rgb.shape[2] < 3:
                raise RequestRejected(f"Camera {name!r} returned an invalid RGB image")
            rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
            depth = None
            if meta.provide_depth:
                if len(buffers) < 2 or buffers[1] is None:
                    raise RequestRejected(f"Camera {name!r} returned no depth image")
                depth = np.ascontiguousarray(buffers[1], dtype=np.float32)
            intrinsic = np.asarray(camera.intrinsics, dtype=np.float64)
            # Genesis 相机为 OpenGL 约定（z 向后）；协议要求光学约定
            # （z 向前, x 右, y 下）。对 world-from-cam 的旋转第 2、3 列
            # 取负即得 world-from-optical（与 genesis Camera.extrinsics
            # 的列翻转同一变换，但不做逆）。
            extrinsic = np.asarray(camera.transform, dtype=np.float64).copy()
            extrinsic[:3, 1:3] *= -1.0
            frames[name] = CameraFrame(
                rgb=rgb,
                depth_m=depth,
                intrinsic_K=intrinsic,
                T_world_from_optical=extrinsic,
                calibration={
                    "convention": "optical_frame_z_forward",
                    "depth_units": "m",
                    "source": "genesis",
                },
            )
        state = self.read_state()
        truncated = self._steps >= self._control.max_episode_steps
        return CapturedObservation(
            episode_id=self.episode_id,
            state_version=state.state_version,
            captured_at=time.time(),
            state=state.model_dump(mode="json"),
            cameras=frames,
            terminated=False,
            truncated=truncated,
        ).freeze()

    def finish(self, reason: str, status: str | None = None) -> dict[str, Any]:
        if not self._sealed:
            self._sealed = True
            success = self._built.judge.evaluate(self._built.objects)
            self._outcome = {
                "task_success": success,
                "native_score": None,
                "complete": True,
                "terminated": True,
                "truncated": bool(
                    self._steps >= self._control.max_episode_steps
                    and success is not True
                ),
                "steps": self._steps,
                "step_limit": self._control.max_episode_steps,
                "finish_request": {"reason": reason, "status": status},
                "stop_reason": reason,
            }
        return dict(self._outcome)

    def close(self) -> None:
        # 场景生命周期归 host/builder 所有（每进程一次 gs.init 约束），
        # 这里只标记句柄关闭，防止 episode 继续被执行。
        self._closed = True
