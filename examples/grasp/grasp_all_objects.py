"""Grasp every RoboTwin object with the FR3 arm (suction) + hierarchical planner.

One Genesis scene is built once: FR3 (URDF), a table, a target marker, and
all object instances parked in a row outside the workspace. Each class is
then evaluated by teleporting its object onto the table and running::

    home -> pre-grasp (planned) -> descend (IK+interp) -> attach suction
    -> lift -> transport above target (planned) -> descend -> detach -> judge

Long moves prefer the hierarchical cuRobo planner and degrade to Genesis
OMPL ``plan_path`` (same routing policy as ``robotwin.curobo_planner``).

Usage::

    uv run python examples/grasp/grasp_all_objects.py --max-classes 3   # smoke
    uv run python examples/grasp/grasp_all_objects.py                   # full run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cloud_robotics_sim.robotwin.curobo_planner import (  # noqa: E402
    CuRoboPlannerConfig,
    CuRoboPlannerUnavailableError,
    HierarchicalCuRoboPlanner,
    PlannerError,
)
from cloud_robotics_sim.robotwin.grasp_report import (  # noqa: E402
    GraspRecord,
    write_report,
)
from cloud_robotics_sim.robotwin.object_library import (  # noqa: E402
    RoboTwinObjectLibrary,
)
from cloud_robotics_sim.robotwin.recorder import EpisodeRecorder  # noqa: E402
from cloud_robotics_sim.robotwin.suction_grasp import (  # noqa: E402
    GRASP_QUAT,
    HOME_QPOS,
    GraspPhase,
    SuctionGrasper,
    _to_numpy,
    follow_path,
    goto_joints,
)
from cloud_robotics_sim.utils.genesis_compat import genesis_init  # noqa: E402

logger = logging.getLogger("grasp_all")

# In-repo URDF (converted from the franka_fr3_v2 MJCF via
# tools/convert_fr3_mjcf_to_urdf.py); portable across machines.
FR3_URDF = REPO_ROOT / "assets_genesis" / "embodiments" / "franka-fr3-v2" / "fr3v2.urdf"

TABLE_Z = 0.40  # table top height
TABLE_CENTER = (0.48, 0.0)
TABLE_SIZE = (0.70, 1.00, TABLE_Z)
SPAWN_XY = (0.42, 0.18)  # object spawn on table
TARGET_XY = (0.55, -0.25)  # place target on table
EE_LINK = "fr3v2_link8"

# PD gains / joint dynamics from the original franka_fr3_v2 MJCF (fr3v2.xml).
# Genesis reads these from MJCF actuators/joint attributes, but URDF has no
# actuator concept, so they must be set explicitly when loading the URDF.
FR3_KP = np.array([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0])
FR3_KV = np.array([450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0])
# class default overwritten in joints 5-7 (see fr3v2.xml defaults)
FR3_DAMPING = np.array([0.21, 0.21, 0.21, 0.21, 0.003, 0.003, 0.003])
FR3_FRICTIONLOSS = np.array([1.137, 1.137, 1.137, 1.137, 0.2, 0.2, 0.2])
FR3_ARMATURE = np.full(7, 0.195)


# ----------------------------------------------------------------------
# Planner wrapper: cuRobo first, OMPL fallback
# ----------------------------------------------------------------------


class PlannerRouter:
    """Route long-range planning to cuRobo, falling back to Genesis OMPL."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.last_used = ""
        self._curobo: HierarchicalCuRoboPlanner | None = None
        if mode in ("auto", "curobo"):
            config = CuRoboPlannerConfig(
                urdf_path=str(FR3_URDF),
                base_link="fr3v2_link0",
                ee_link=EE_LINK,
                workspace_bounds=np.array(
                    [[0.0, 0.95], [-0.6, 0.6], [0.0, 1.1]], dtype=np.float64
                ),
                obstacles=[
                    {
                        "type": "cuboid",
                        "dims": list(TABLE_SIZE),
                        "pose": [TABLE_CENTER[0], TABLE_CENTER[1], TABLE_Z / 2],
                    }
                ],
            )
            self._curobo = HierarchicalCuRoboPlanner(config)

    def plan(
        self,
        robot,
        start_q: np.ndarray,
        goal_pos: np.ndarray,
        goal_quat: np.ndarray,
        num_waypoints: int = 40,
    ) -> np.ndarray:
        """Return a joint-space path; raises on total failure."""
        if self._curobo is not None and self.mode != "ompl":
            try:
                traj = self._curobo.plan_to_ee_pose(start_q, goal_pos, goal_quat)
                self.last_used = "hierarchical_curobo"
                return traj
            except (CuRoboPlannerUnavailableError, PlannerError) as exc:
                logger.debug("cuRobo unavailable/failed, using OMPL: %s", exc)
                if self.mode == "curobo":
                    raise
        ee = robot.get_link(EE_LINK)
        import torch

        q_goal = robot.inverse_kinematics(
            ee,
            pos=torch.as_tensor(goal_pos, dtype=torch.float64),
            quat=torch.as_tensor(goal_quat, dtype=torch.float64),
        )
        path = robot.plan_path(
            qpos_goal=_to_numpy(q_goal).reshape(-1),
            qpos_start=np.asarray(start_q, dtype=np.float64),
            num_waypoints=num_waypoints,
        )
        path = _to_numpy(path)
        if path.size == 0:
            raise PlannerError("OMPL plan_path returned an empty path")
        self.last_used = "ompl"
        return path


# ----------------------------------------------------------------------
# Single-class pick-and-place
# ----------------------------------------------------------------------


def ik_to(robot, pos: np.ndarray) -> np.ndarray:
    """Solve IK for the grasp orientation at ``pos``."""
    import torch

    ee = robot.get_link(EE_LINK)
    q = robot.inverse_kinematics(
        ee,
        pos=torch.as_tensor(pos, dtype=torch.float64),
        quat=torch.as_tensor(GRASP_QUAT, dtype=torch.float64),
    )
    return _to_numpy(q).reshape(-1)


def run_one_class(
    scene,
    robot,
    obj,
    target_marker_xy: tuple[float, float],
    router: PlannerRouter,
    record: GraspRecord,
    recorder: EpisodeRecorder | None = None,
    spawn_xy: tuple[float, float] = SPAWN_XY,
) -> None:
    """Execute the full pick-and-place for one object; mutate ``record``.

    When ``recorder`` is given, every sim step is captured (joint qpos, EE
    endpose, object pose, phase label, commanded action) as a RoboTwin-format
    demonstration episode — including partial episodes on failure.
    """
    grasper = SuctionGrasper(robot=robot, obj=obj, ee_link_name=EE_LINK)
    phase = {"value": int(GraspPhase.RESET)}

    def set_phase(p: GraspPhase) -> None:
        phase["value"] = int(p)

    hook = None
    if recorder is not None:

        def hook(target_q: np.ndarray) -> None:  # type: ignore[no-redef]
            q = _to_numpy(robot.get_qpos()).reshape(-1)[:7]
            ee_pos, ee_quat = grasper.ee_pose()
            obj_pos, obj_quat = grasper.obj_base_pose()
            recorder.capture(
                recorder.n_frames,
                qpos=q,
                endpose=np.concatenate([ee_pos, ee_quat]),
                extra={
                    "obj_pose": np.concatenate([obj_pos, obj_quat]),
                    "phase": np.array([phase["value"]], dtype=np.float64),
                    "action": np.asarray(target_q, dtype=np.float64).reshape(-1)[:7],
                },
            )

    def current_q() -> np.ndarray:
        return _to_numpy(robot.get_qpos()).reshape(-1)[:7]

    # -- reset robot to home -------------------------------------------
    robot.set_qpos(HOME_QPOS)
    for _ in range(10):
        robot.control_dofs_position(HOME_QPOS)
        scene.step()
        if hook is not None:
            hook(HOME_QPOS)

    # -- teleport object above the table and let it settle --------------
    qpos = _to_numpy(obj.get_qpos()).reshape(-1)
    qpos[:3] = [spawn_xy[0], spawn_xy[1], TABLE_Z + 0.30]
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    obj.set_qpos(qpos)
    for _ in range(60):
        scene.step()
        if hook is not None:
            hook(HOME_QPOS)

    aabb = _to_numpy(obj.get_AABB())
    obj_h = float(aabb[1, 2] - aabb[0, 2])
    center_xy = (aabb[0, :2] + aabb[1, :2]) / 2.0
    top_z = float(aabb[1, 2])
    if obj_h <= 1e-4 or obj_h > 0.6:
        record.status = "load_fail"
        record.message = f"implausible object height {obj_h:.3f} m"
        return
    # rest pose of the settled object (used for an upright release)
    rest_qpos = _to_numpy(obj.get_qpos()).reshape(-1).copy()

    tx, ty = target_marker_xy

    # -- pre-grasp (planned long move) ----------------------------------
    set_phase(GraspPhase.PRE_GRASP)
    try:
        path = router.plan(
            robot,
            current_q(),
            np.array([center_xy[0], center_xy[1], top_z + 0.12]),
            GRASP_QUAT,
        )
    except Exception as exc:
        record.status = "plan_fail"
        record.message = f"pre-grasp: {exc}"
        return
    record.planner = router.last_used
    follow_path(scene, robot, path, grasper=None, steps_per_waypoint=2, on_step=hook)

    # -- descend to grasp ------------------------------------------------
    set_phase(GraspPhase.DESCEND)
    q_grasp = ik_to(robot, np.array([center_xy[0], center_xy[1], top_z - 0.004]))
    goto_joints(scene, robot, q_grasp, grasper=None, n_steps=25, on_step=hook)
    ee_pos, _ = grasper.ee_pose()
    if (
        float(np.linalg.norm(ee_pos[:2] - center_xy)) > 0.04
        or abs(ee_pos[2] - top_z) > 0.05
    ):
        record.status = "grasp_fail"
        record.message = f"EE did not reach grasp point (ee={np.round(ee_pos, 3)})"
        return
    set_phase(GraspPhase.GRASP)
    grasper.attach()

    # -- lift -------------------------------------------------------------
    set_phase(GraspPhase.LIFT)
    obj_z_before, _ = grasper.obj_base_pose()
    q_pre_lift = ik_to(robot, np.array([center_xy[0], center_xy[1], top_z + 0.12]))
    goto_joints(scene, robot, q_pre_lift, grasper=grasper, n_steps=25, on_step=hook)
    obj_pos, _ = grasper.obj_base_pose()
    if obj_pos[2] < obj_z_before[2] + 0.05:
        record.status = "lift_fail"
        record.message = (
            f"object did not lift with the flange "
            f"(z {obj_z_before[2]:.3f} -> {obj_pos[2]:.3f})"
        )
        grasper.detach()
        return

    # -- transport above target (planned) ---------------------------------
    set_phase(GraspPhase.TRANSPORT)
    try:
        path = router.plan(
            robot,
            current_q(),
            np.array([tx, ty, TABLE_Z + obj_h + 0.15]),
            GRASP_QUAT,
        )
    except Exception as exc:
        record.status = "transport_fail"
        record.message = f"transport: {exc}"
        grasper.detach()
        return
    follow_path(scene, robot, path, grasper=grasper, steps_per_waypoint=2, on_step=hook)
    if not grasper.is_holding():
        record.status = "transport_fail"
        record.message = "object dropped during transport"
        grasper.detach()
        return

    # -- place -------------------------------------------------------------
    set_phase(GraspPhase.PLACE)
    q_place = ik_to(robot, np.array([tx, ty, TABLE_Z + obj_h + 0.005]))
    goto_joints(scene, robot, q_place, grasper=grasper, n_steps=25, on_step=hook)
    grasper.follow_step()
    # upright, zero-velocity release at the rest pose (suction place)
    set_phase(GraspPhase.RELEASE)
    grasper.detach()
    place_qpos = rest_qpos.copy()
    place_qpos[:3] = [tx, ty, rest_qpos[2]]
    obj.set_qpos(place_qpos)
    if hasattr(obj, "set_dofs_velocity"):
        try:
            import torch

            obj.set_dofs_velocity(torch.zeros(obj.n_dofs))
        except Exception:  # noqa: BLE001
            pass
    q_hold = current_q()
    for _ in range(50):
        scene.step()
        if hook is not None:
            hook(q_hold)
    set_phase(GraspPhase.RETREAT)
    q_retreat = ik_to(robot, np.array([tx, ty, TABLE_Z + obj_h + 0.15]))
    goto_joints(scene, robot, q_retreat, grasper=None, n_steps=15, on_step=hook)

    # -- judge --------------------------------------------------------------
    final = _to_numpy(obj.get_qpos()).reshape(-1)[:3]
    record.final_pos = [float(x) for x in final]
    xy_err = float(np.linalg.norm(final[:2] - np.array([tx, ty])))
    if xy_err < 0.08 and TABLE_Z - 0.03 < final[2] < TABLE_Z + obj_h + 0.15:
        record.status = "success"
    else:
        record.status = "place_fail"
        record.message = f"final pos {np.round(final, 3)}, xy_err {xy_err:.3f}"


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def run_chunked(args: argparse.Namespace) -> None:
    """Driver mode: run the benchmark in VRAM-safe chunks (subprocesses).

    Each chunk builds its own scene in a fresh subprocess (kernel cache is
    shared), so a CUDA crash only sacrifices the current chunk. Partial
    chunk reports are merged into ``args.out/report.{json,md}``.
    """
    import json
    import subprocess

    library = RoboTwinObjectLibrary(args.objects_dir)
    classes = args.classes or library.list_classes()
    if args.max_classes:
        classes = classes[: args.max_classes]
    chunks = [
        classes[i : i + args.chunk_size]
        for i in range(0, len(classes), args.chunk_size)
    ]
    logger.info(
        "chunked run: %d classes in %d chunks of <= %d",
        len(classes),
        len(chunks),
        args.chunk_size,
    )
    all_records: list[GraspRecord] = []
    chunk_root = args.out / "chunks"
    for k, chunk in enumerate(chunks):
        chunk_out = chunk_root / f"chunk_{k:03d}"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--objects-dir",
            str(args.objects_dir),
            "--out",
            str(chunk_out),
            "--planner",
            args.planner,
            "--seed",
            str(args.seed + k),
            "--classes",
            *chunk,
        ]
        if args.record:
            cmd += [
                "--record",
                "--episodes-per-class",
                str(args.episodes_per_class),
                "--jitter-xy",
                str(args.jitter_xy),
            ]
        logger.info("--- chunk %d/%d (%d classes) ---", k + 1, len(chunks), len(chunk))
        t0 = time.time()
        ret = subprocess.call(cmd)
        logger.info(
            "--- chunk %d/%d exit=%d (%.1f s) ---",
            k + 1,
            len(chunks),
            ret,
            time.time() - t0,
        )
        chunk_records: dict[str, GraspRecord] = {}
        report_path = chunk_out / "report.json"
        if report_path.is_file():
            try:
                payload = json.loads(report_path.read_text(encoding="utf-8"))
                for raw in payload.get("records", []):
                    chunk_records[raw["class_name"]] = GraspRecord(
                        **{
                            f: v
                            for f, v in raw.items()
                            if f in GraspRecord.__dataclass_fields__
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error("failed to read %s: %s", report_path, exc)
        for name in chunk:
            if name in chunk_records:
                all_records.append(chunk_records[name])
            else:
                all_records.append(
                    GraspRecord(
                        class_name=name,
                        status="error",
                        message=f"chunk process crashed (exit={ret})",
                    )
                )
        # incremental merged report (crash-safe)
        write_report(
            all_records,
            args.out,
            extra={"planner_mode": args.planner, "chunk_size": args.chunk_size},
        )

    # retry pass: physics-crash casualties (poisoned scenes, chunk crashes)
    # get a fair second attempt in isolated single-class subprocesses
    retry_idx = [i for i, r in enumerate(all_records) if r.status == "error"]
    if retry_idx:
        logger.info("retry pass: %d error records", len(retry_idx))
        for i in retry_idx:
            name = all_records[i].class_name
            chunk_out = chunk_root / f"retry_{name}"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--objects-dir",
                str(args.objects_dir),
                "--out",
                str(chunk_out),
                "--planner",
                args.planner,
                "--classes",
                name,
            ]
            logger.info("retry %s ...", name)
            ret = subprocess.call(cmd)
            report_path = chunk_out / "report.json"
            new_rec: GraspRecord | None = None
            if report_path.is_file():
                try:
                    payload = json.loads(report_path.read_text(encoding="utf-8"))
                    for raw in payload.get("records", []):
                        if raw["class_name"] == name:
                            new_rec = GraspRecord(
                                **{
                                    f: v
                                    for f, v in raw.items()
                                    if f in GraspRecord.__dataclass_fields__
                                }
                            )
                except Exception as exc:  # noqa: BLE001
                    logger.error("failed to read %s: %s", report_path, exc)
            if new_rec is not None:
                new_rec.message = (new_rec.message + " [retried]").strip()
                all_records[i] = new_rec
            else:
                all_records[i].message += f" [retry crashed exit={ret}]"
            write_report(
                all_records,
                args.out,
                extra={"planner_mode": args.planner, "chunk_size": args.chunk_size},
            )

    json_path, md_path = write_report(all_records, args.out)
    logger.info("CHUNKED RUN DONE: %s, %s", json_path, md_path)


def main() -> None:
    """Entry point: parse args and run the grasp-all benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objects-dir",
        type=Path,
        default=REPO_ROOT / "assets" / "robotwin" / "objects" / "objects",
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "outputs" / "grasp_all_objects"
    )
    parser.add_argument(
        "--classes", nargs="*", default=None, help="subset of class names"
    )
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--planner", choices=["auto", "curobo", "ompl"], default="auto")
    parser.add_argument(
        "--record",
        action="store_true",
        help="record every step as a RoboTwin HDF5 episode per class "
        "(saved under <out>/episodes/)",
    )
    parser.add_argument(
        "--episodes-per-class",
        type=int,
        default=1,
        help="episodes per class (>1 = online trajectory augmentation)",
    )
    parser.add_argument(
        "--jitter-xy",
        type=float,
        default=0.0,
        help="uniform +/- jitter (m) applied to spawn and target xy per episode",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="objects per scene (0 = all in one scene; >0 spawns one "
        "subprocess per chunk to bound GPU memory)",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.out / "run.log", mode="a", encoding="utf-8"),
        ],
    )

    if args.chunk_size > 0:
        run_chunked(args)
        return
    run_single_scene(args)


def run_single_scene(args: argparse.Namespace) -> None:
    """Single-process mode: one scene containing all requested objects."""
    library = RoboTwinObjectLibrary(args.objects_dir)
    classes = args.classes or library.list_classes()
    if args.max_classes:
        classes = classes[: args.max_classes]

    # Resolve instances up-front; load failures skip scene building for them.
    records: list[GraspRecord] = []
    spawnable: list[tuple[str, object]] = []
    for name in classes:
        try:
            inst = library.get_instance(name)
            spawnable.append((name, inst))
        except Exception as exc:
            records.append(
                GraspRecord(class_name=name, status="load_fail", message=str(exc))
            )
            logger.warning("load_fail %s: %s", name, exc)

    import genesis as gs

    genesis_init(backend="cuda", logging_level="warning")
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.Box(
            size=TABLE_SIZE,
            pos=(TABLE_CENTER[0], TABLE_CENTER[1], TABLE_Z / 2),
            fixed=True,
        )
    )
    # NOTE: no visible target marker — a fixed marker at the target point
    # would collide with the placed object (headless run, coords suffice).
    # links_to_keep: Genesis merges fixed-joint links by default, which would
    # drop the flange link ``fr3v2_link8`` (fixed-attached to link7).
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(FR3_URDF),
            fixed=True,
            links_to_keep=[EE_LINK],
        )
    )

    obj_entities: dict[str, object] = {}
    park_positions: dict[str, tuple[float, float, float]] = {}
    # generous 0.6 m grid spacing: tightly packed parked objects overlap and
    # blow up the contact solver (NaN / max-contact-pairs errors)
    for i, (name, inst) in enumerate(spawnable):
        park_pos = (0.05 + 0.6 * (i % 4), 2.0 + 0.6 * (i // 4), 0.5)
        try:
            obj_entities[name] = library.spawn_in_scene(scene, name, pos=park_pos)
            park_positions[name] = park_pos
        except Exception as exc:
            records.append(
                GraspRecord(class_name=name, status="load_fail", message=str(exc))
            )
            logger.warning("load_fail(spawn) %s: %s", name, exc)
    spawn_names = [n for n, _ in spawnable if n in obj_entities]

    logger.info("building scene with %d objects...", len(obj_entities))
    t0 = time.time()
    scene.build()
    logger.info("scene built in %.1f s", time.time() - t0)

    link_names = [link.name for link in robot.links]
    if EE_LINK not in link_names:
        raise RuntimeError(
            f"EE_LINK {EE_LINK!r} not found after scene.build(); "
            f"links_to_keep may have failed. Available: {link_names}"
        )

    # Restore the MJCF's actuator gains / joint dynamics (URDF carries none).
    dofs7 = list(range(7))
    robot.set_dofs_kp(FR3_KP, dofs_idx_local=dofs7)
    robot.set_dofs_kv(FR3_KV, dofs_idx_local=dofs7)
    robot.set_dofs_damping(FR3_DAMPING, dofs_idx_local=dofs7)
    robot.set_dofs_frictionloss(FR3_FRICTIONLOSS, dofs_idx_local=dofs7)
    robot.set_dofs_armature(FR3_ARMATURE, dofs_idx_local=dofs7)

    router = PlannerRouter(args.planner)
    rng = np.random.default_rng(args.seed)
    episodes_dir = args.out / "episodes"
    for name in spawn_names:
        record = GraspRecord(class_name=name)
        inst = library.get_instance(name)
        record.instance_index = inst.index
        record.kind = inst.kind
        t0 = time.time()
        logger.info("=== %s (%d/%d) ===", name, len(records) + 1, len(spawn_names))
        for ep_idx in range(args.episodes_per_class):
            if args.jitter_xy > 0:
                j = args.jitter_xy
                spawn_xy = tuple(np.asarray(SPAWN_XY) + rng.uniform(-j, j, 2))
                target_xy = tuple(
                    np.clip(
                        np.asarray(TARGET_XY) + rng.uniform(-j, j, 2),
                        [0.18, -0.45],
                        [0.78, 0.45],
                    )
                )
            else:
                spawn_xy, target_xy = SPAWN_XY, TARGET_XY
            recorder = None
            if args.record:
                recorder = EpisodeRecorder(
                    task_name=f"grasp_{name}",
                    fps=100.0,
                    metadata={
                        "class_name": name,
                        "kind": inst.kind,
                        "episode_idx": ep_idx,
                        "spawn_xy": list(spawn_xy),
                        "target_xy": list(target_xy),
                        "jitter_xy": args.jitter_xy,
                        "seed": args.seed,
                    },
                )
            ep_record = (
                record
                if ep_idx == args.episodes_per_class - 1
                else GraspRecord(class_name=name)
            )
            try:
                run_one_class(
                    scene,
                    robot,
                    obj_entities[name],
                    target_xy,
                    router,
                    ep_record,
                    recorder=recorder,
                    spawn_xy=spawn_xy,
                )
            except Exception as exc:  # noqa: BLE001 - keep the batch going
                ep_record.status = "error"
                ep_record.message = f"{type(exc).__name__}: {exc}"
                logger.error("%s crashed:\n%s", name, traceback.format_exc())
            if recorder is not None and recorder.n_frames > 0:
                recorder.metadata["status"] = ep_record.status
                recorder.metadata["planner"] = ep_record.planner or router.last_used
                ep_path = episodes_dir / f"{name}_ep{ep_idx}.hdf5"
                try:
                    recorder.save_hdf5(ep_path)
                    logger.info(
                        "saved episode %s (%d frames)", ep_path, recorder.n_frames
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("failed to save episode %s: %s", ep_path, exc)
            if ep_idx < args.episodes_per_class - 1:
                logger.info("    ep%d -> %s", ep_idx, ep_record.status)
        record.duration_s = time.time() - t0
        if record.planner == "":
            record.planner = router.last_used
        records.append(record)
        logger.info("=== %s -> %s (%.1f s) ===", name, record.status, record.duration_s)
        # park the used object again (out of the workspace)
        try:
            qpos = _to_numpy(obj_entities[name].get_qpos()).reshape(-1)
            qpos[:3] = park_positions[name]
            qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            obj_entities[name].set_qpos(qpos)
        except Exception:  # noqa: BLE001
            pass
        # incremental report (crash-safe)
        write_report(
            records,
            args.out,
            extra={
                "planner_mode": args.planner,
                "robot": "franka_fr3_v2 (suction)",
                "objects_dir": str(args.objects_dir),
            },
        )

    json_path, md_path = write_report(records, args.out)
    logger.info("DONE: %s, %s", json_path, md_path)


if __name__ == "__main__":
    main()
