"""SceneSpec -> gs.Scene：通用场景构建。

genesis 为延迟导入（模块级不 import），保证插件在缺少 genesis 或
uni-robot-agent 时仍可被导入。所有 Genesis 调用集中在 build_scene；
构建失败时显式报错并保留原始异常链。

注意：dof 限位 / link 元数据在 scene.build() 之后才可用，因此臂的
原生句柄分两步拿到——build 前只登记实体，build 后由 _PendingArm
物化为 BuiltArm。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .built import BuiltArm, BuiltScene
from .judge import build_judge
from .scene_spec import ArmSpec, SceneSpec

logger = logging.getLogger(__name__)

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


def _import_genesis() -> Any:
    try:
        import genesis as gs
    except ImportError as error:  # pragma: no cover - 环境缺依赖时
        raise RuntimeError(
            "genesis-world is required to build a Genesis scene for the "
            "uni-robot agent adapter; install the base project dependencies"
        ) from error
    return gs


def _f64(value: Any) -> np.ndarray:
    """Torch 张量友好地转 float64 ndarray（避免 numpy2 copy 告警）。"""
    detach = getattr(value, "detach", None)
    if callable(detach):
        return np.asarray(detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _dof_names(entity: Any, n_dofs: int) -> list[str]:
    """尽力读取 dof 名称；不同 genesis 版本的命名访问器不一致，逐层回退。"""
    for attr in ("get_dofs_name", "get_joints_name"):
        accessor = getattr(entity, attr, None)
        if callable(accessor):
            try:
                names = list(accessor())
                if len(names) == n_dofs:
                    return [str(n) for n in names]
            except Exception:  # noqa: BLE001 - 回退到下一种来源
                pass
    dofs = getattr(entity, "dofs", None)
    if dofs is not None:
        try:
            return [str(dof.name) for dof in dofs]
        except Exception:  # noqa: BLE001
            pass
    return [f"dof_{i}" for i in range(n_dofs)]


def _detect_gripper_dofs(names: list[str]) -> tuple[int, ...]:
    hints = ("finger", "gripper", "grip")
    hits = tuple(
        i for i, name in enumerate(names) if any(h in name.lower() for h in hints)
    )
    if hits:
        return hits
    # 回退：约定最后两个 dof 为平行夹爪（franka/panda 等常见 URDF）
    return (len(names) - 2, len(names) - 1)


def _select_grasp_link(entity: Any, spec: ArmSpec, link_names: list[str]) -> Any:
    if spec.eef_link is not None:
        link = entity.get_link(spec.eef_link)
        if link is None:
            raise ValueError(f"eef_link {spec.eef_link!r} not found on robot")
        return link
    for candidate in _EEF_LINK_CANDIDATES:
        if candidate in link_names:
            return entity.get_link(candidate)
    # 回退：最后一个 link（Genesis links 末尾通常为工具链节）
    links = list(entity.links)
    if not links:
        raise ValueError("robot has no links to use as grasp site")
    return links[-1]


def _add_box(gs: Any, scene: Any, box: Any) -> Any:
    # genesis-world 1.4.x：surface 是 add_entity 的参数，不是 morph 的。
    surface = gs.surfaces.Default(color=tuple(box.color))
    return scene.add_entity(
        gs.morphs.Box(
            size=tuple(box.size),
            pos=tuple(box.position),
            quat=tuple(box.quaternion),
            fixed=box.fixed,
        ),
        surface=surface,
        name=box.name,
    )


def build_scene(spec: SceneSpec) -> BuiltScene:
    """按 spec 构建并 build() 一个 Genesis 场景（单环境、无 viewer）。"""
    gs = _import_genesis()
    try:
        from cloud_robotics_sim.utils import genesis_compat
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "cloud_robotics_sim is required to build a Genesis scene for the "
            "uni-robot agent adapter"
        ) from error

    genesis_compat.genesis_init(
        use_cuda=spec.sim.use_cuda if spec.sim.use_cuda is not None else True
    )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=spec.sim.dt, substeps=spec.sim.substeps),
        rigid_options=gs.options.RigidOptions(batch_links_info=True),
        show_viewer=False,
    )

    if spec.ground:
        scene.add_entity(gs.morphs.Plane())
    for fixture in spec.fixtures:
        _add_box(gs, scene, fixture)

    objects: dict[str, Any] = {}
    for obj in spec.objects:
        objects[obj.name] = _add_box(gs, scene, obj)

    pending: dict[str, _PendingArm] = {}
    for arm_spec in spec.arms:
        pending[arm_spec.name] = _add_robot(gs, scene, arm_spec)

    cameras: dict[str, Any] = {}
    camera_meta = {}
    for cam in spec.cameras:
        cameras[cam.name] = scene.add_camera(
            pos=tuple(cam.position),
            lookat=tuple(cam.lookat),
            res=tuple(cam.resolution),
            fov=cam.fov,
            GUI=False,
        )
        camera_meta[cam.name] = cam

    scene.build()

    arms = {name: pending_arm.materialize() for name, pending_arm in pending.items()}

    control_dt = 1.0 / spec.control.frequency_hz
    ticks = spec.control.ticks_per_control
    if ticks is None:
        ticks = max(1, int(round(control_dt / spec.sim.dt)))
        if not np.isclose(ticks * spec.sim.dt, control_dt, rtol=0, atol=spec.sim.dt):
            logger.warning(
                "control period %.4f s is not an integer multiple of dt %.4f s; "
                "using %d ticks (%.4f s)",
                control_dt,
                spec.sim.dt,
                ticks,
                ticks * spec.sim.dt,
            )
    judge = build_judge(spec.judge)
    return BuiltScene(
        scene=scene,
        dt=spec.sim.dt,
        ticks_per_control=int(ticks),
        arms=arms,
        cameras=cameras,
        camera_meta=camera_meta,
        objects=objects,
        judge=judge,
        spec=spec,
    )


def _add_robot(gs: Any, scene: Any, spec: ArmSpec) -> "_PendingArm":
    """登记机器人实体；dof 元数据待 scene.build() 后物化。"""
    if spec.urdf_path is not None:
        urdf_path = spec.urdf_path
    else:
        from cloud_robotics_sim.core.robot_assets import resolve_robot_model

        model = resolve_robot_model(spec.preset)
        if model is None:
            raise ValueError(
                f"cannot resolve robot model for preset {spec.preset!r}; "
                "set arm.urdf_path explicitly or prefetch assets "
                "(python -m cloud_robotics_sim.core.robot_assets)"
            )
        urdf_path = str(model.path)

    entity = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_path,
            pos=tuple(spec.base_position),
            quat=tuple(spec.base_quaternion),
            fixed=True,
            # 保留 grasp site 候选 link（固定关节默认会被合并掉，如 franka
            # 的 panda_hand）；links_to_keep 中不存在的名字会被忽略。
            links_to_keep=list(_EEF_LINK_CANDIDATES)
            + ([spec.eef_link] if spec.eef_link else []),
        ),
        name=f"{spec.name}_robot",
    )
    return _PendingArm(entity=entity, spec=spec)


class _PendingArm:
    """build() 之前无法读取 dof 限位；占位对象在 build 后物化为 BuiltArm。"""

    def __init__(self, *, entity: Any, spec: ArmSpec) -> None:
        self.entity, self.spec = entity, spec

    def materialize(self) -> BuiltArm:
        spec, entity = self.spec, self.entity
        n_dofs = int(entity.n_dofs)
        names = _dof_names(entity, n_dofs)

        if spec.gripper.dofs is not None:
            gripper_dofs = tuple(int(i) for i in spec.gripper.dofs)
        else:
            gripper_dofs = _detect_gripper_dofs(names)
        if spec.arm_dofs is not None:
            arm_dofs = tuple(int(i) for i in spec.arm_dofs)
        else:
            arm_dofs = tuple(i for i in range(n_dofs) if i not in gripper_dofs)
        if not arm_dofs or not gripper_dofs:
            raise ValueError(
                f"arm {spec.name!r}: arm_dofs/gripper_dofs detection failed "
                f"(n_dofs={n_dofs}, gripper={gripper_dofs})"
            )

        primary_local = spec.gripper.primary_dof
        if primary_local is None:
            primary_local = gripper_dofs[0]
        if primary_local not in gripper_dofs:
            raise ValueError("primary gripper dof must be one of the gripper dofs")

        lower, upper = entity.get_dofs_limit(dofs_idx_local=list(gripper_dofs))
        lower = _f64(lower)
        upper = _f64(upper)
        grip_index = gripper_dofs.index(primary_local)
        if spec.gripper.joint_limits is not None:
            gripper_limits = (
                float(spec.gripper.joint_limits[0]),
                float(spec.gripper.joint_limits[1]),
            )
        else:
            lo, hi = float(lower[grip_index]), float(upper[grip_index])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
                raise ValueError(
                    f"arm {spec.name!r}: invalid gripper joint limits {lo}..{hi}; "
                    "set gripper.joint_limits explicitly"
                )
            gripper_limits = (lo, hi)

        arm_lower, arm_upper = entity.get_dofs_limit(dofs_idx_local=list(arm_dofs))
        joint_limits = np.stack([_f64(arm_lower), _f64(arm_upper)], axis=1)

        links = list(entity.links)
        link_names = [str(link.name) for link in links]
        grasp_link = _select_grasp_link(entity, spec, link_names)

        # 可选的驱动增益覆盖（按 arm+gripper dof 顺序）
        all_dofs = list(arm_dofs) + list(gripper_dofs)
        if spec.kp is not None:
            if len(spec.kp) != len(all_dofs):
                raise ValueError(
                    f"arm {spec.name!r}: kp length {len(spec.kp)} != n_dofs {len(all_dofs)}"
                )
            entity.set_dofs_kp(list(spec.kp), dofs_idx_local=all_dofs)
        if spec.kv is not None:
            if len(spec.kv) != len(all_dofs):
                raise ValueError(
                    f"arm {spec.name!r}: kv length {len(spec.kv)} != n_dofs {len(all_dofs)}"
                )
            entity.set_dofs_kv(list(spec.kv), dofs_idx_local=all_dofs)

        return BuiltArm(
            name=spec.name,
            entity=entity,
            arm_dofs=arm_dofs,
            gripper_dofs=gripper_dofs,
            primary_gripper_dof=primary_local,
            grasp_link=grasp_link,
            joint_limits=joint_limits,
            gripper_limits=gripper_limits,
            joint_names=[names[i] for i in arm_dofs],
            rotation_radius_m=spec.rotation_radius_m,
            joint_tracking_tolerance_rad=spec.joint_tracking_tolerance_rad,
            grasp_site_description=spec.grasp_site_description,
        )
