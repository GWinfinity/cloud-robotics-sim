"""Home demo videos: 5 rooms, real RoboTwin-OD assets, planner-driven tasks.

Re-record of the apartment demo videos with the demo-quality fixes:

1. **Real tasks** — each room runs a full pick-and-place driven by the
   hierarchical cuRobo planner with Genesis OMPL fallback (same routing as
   ``scripts/grasp_all_objects.py``); outcomes are judged and reported via
   ``robotwin.grasp_report`` and shown as an end-of-video banner.
2. **Real assets** — room anchors, pick objects, and target containers come
   from the RoboTwin-OD object library (incl. the self-built parametric
   storage assets registered in ``data/recipes/asset_gap_list.yaml``); no
   ``obstacle_box`` placeholders. Large furniture that is still a registered
   gap (bed / sofa / toilet / washing machine) is represented by its zone
   anchor (wardrobe, storage ottoman, over-toilet shelf, slim cart).
3. **Higher render tier** — 1920x1080 offscreen camera, colored surfaces,
   H.264 encode at a configurable bitrate (default 4 Mbps).
4. **Camera work** — low, close slow-orbit camera; per-room title card,
   live phase label, and a SUCCESS/FAILED end banner.

One subprocess per room (default for multi-room runs) bounds VRAM usage and
isolates CUDA crashes; per-room ``record_<room>.json`` files are merged into
``report.{json,md}``.

Usage::

    # smoke: one room, CPU, small frames
    uv run python scripts/run_home_demo.py --room kitchen --quick --backend cpu \
        --planner ompl --resolution 640x360

    # full run on the 24G server
    uv run python scripts/run_home_demo.py --out outputs/home_demo
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from cloud_robotics_sim.robotwin.grasp_report import (  # noqa: E402
    GraspRecord,
    write_report,
)
from cloud_robotics_sim.robotwin.object_library import (  # noqa: E402
    RoboTwinObjectLibrary,
)
from cloud_robotics_sim.robotwin.suction_grasp import (  # noqa: E402
    GRASP_QUAT,
    HOME_QPOS,
    GraspPhase,
    SuctionGrasper,
    _to_numpy,
    follow_path,
    goto_joints,
)
from scripts.grasp_all_objects import (  # noqa: E402
    EE_LINK,
    FR3_ARMATURE,
    FR3_DAMPING,
    FR3_FRICTIONLOSS,
    FR3_KP,
    FR3_KV,
    FR3_URDF,
    PlannerRouter,
    ik_to,
)

logger = logging.getLogger("home_demo")

# ----------------------------------------------------------------------
# Room definitions
# ----------------------------------------------------------------------

COUNTER_CENTER = (0.48, 0.0)
COUNTER_SIZE = (0.70, 1.00, 0.40)  # matches PlannerRouter's table obstacle
COUNTER_Z = COUNTER_SIZE[2]
SPAWN_XY = (0.40, 0.22)  # pick object on the counter
TARGET_XY = (0.55, -0.25)  # target container on the counter

ROOMS: dict[str, dict] = {
    "kitchen": {
        "title": "KITCHEN",
        "task": "Soy-sauce bottle -> rotating spice tray",
        "pick": "065_soy-sauce",
        "target": "161_rotating_spice_tray",
        "anchors": [("125_fridge", (1.05, 1.15, 0.0))],
        "decor": [],
        "floor_color": (0.78, 0.72, 0.62, 1.0),
        "wall_color": (0.90, 0.88, 0.82, 1.0),
    },
    "living_room": {
        "title": "LIVING ROOM",
        "task": "Remote control -> remote caddy",
        "pick": "079_remotecontrol",
        "target": "136_remote_caddy",
        "anchors": [("163_storage_ottoman", (1.00, 1.10, 0.0))],
        "decor": [("023_tissue-box", (0.30, 0.42, COUNTER_Z))],
        "floor_color": (0.76, 0.64, 0.50, 1.0),
        "wall_color": (0.93, 0.91, 0.86, 1.0),
    },
    "bedroom": {
        "title": "BEDROOM",
        "task": "Alarm clock -> rotating desk tray",
        "pick": "046_alarm-clock",
        "target": "160_rotating_desk_tray",
        "anchors": [("126_wardrobe", (1.05, 1.15, 0.0))],
        "decor": [("115_perfume", (0.30, 0.42, COUNTER_Z))],
        "floor_color": (0.66, 0.53, 0.40, 1.0),
        "wall_color": (0.88, 0.86, 0.84, 1.0),
    },
    "bathroom": {
        "title": "BATHROOM",
        "task": "Toothpaste -> tube organizer",
        "pick": "118_tooth-paste",
        "target": "156_tube_organizer",
        "anchors": [("151_over_toilet_shelf", (1.05, 1.15, 0.0))],
        "decor": [("049_shampoo", (0.30, 0.42, COUNTER_Z))],
        "floor_color": (0.80, 0.85, 0.87, 1.0),
        "wall_color": (0.85, 0.90, 0.91, 1.0),
    },
    "laundry_room": {
        "title": "LAUNDRY ROOM",
        "task": "Soap -> detergent caddy",
        "pick": "107_soap",
        "target": "146_detergent_caddy",
        "anchors": [
            ("142_rolling_slim_cart", (1.00, 1.10, 0.0)),
            ("011_dustbin", (1.55, 0.55, 0.0)),
        ],
        "decor": [],
        "floor_color": (0.84, 0.84, 0.82, 1.0),
        "wall_color": (0.89, 0.90, 0.88, 1.0),
    },
}

PHASE_LABELS = {
    GraspPhase.RESET: "reset",
    GraspPhase.PRE_GRASP: "approach",
    GraspPhase.DESCEND: "descend",
    GraspPhase.GRASP: "grasp",
    GraspPhase.LIFT: "lift",
    GraspPhase.TRANSPORT: "transport",
    GraspPhase.PLACE: "place",
    GraspPhase.RELEASE: "release",
    GraspPhase.RETREAT: "retreat",
}


# ----------------------------------------------------------------------
# Video overlay (titles, phase chip, end banner)
# ----------------------------------------------------------------------


def _font_path() -> str | None:
    """Locate a bundled TrueType font (matplotlib ships DejaVu Sans)."""
    try:
        from matplotlib import font_manager

        return font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))
    except Exception:  # noqa: BLE001
        return None


_FONT_PATH = _font_path()


def _font(size: int):
    from PIL import ImageFont

    if _FONT_PATH:
        return ImageFont.truetype(_FONT_PATH, size)
    return ImageFont.load_default()


def overlay_frame(
    rgb: np.ndarray,
    title: str,
    task: str,
    phase: str | None = None,
    intro: bool = False,
    banner: str | None = None,
    banner_ok: bool = True,
) -> np.ndarray:
    """Draw demo overlays onto an RGB frame and return the new frame."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    s = h / 1080.0  # scale factor for font sizes

    if intro:
        # centered title card
        f_big = _font(int(84 * s))
        f_sub = _font(int(36 * s))
        tw = d.textlength(title, font=f_big)
        sw = d.textlength(task, font=f_sub)
        cx, cy = w / 2, h * 0.42
        d.rounded_rectangle(
            [cx - tw / 2 - 40 * s, cy - 100 * s, cx + tw / 2 + 40 * s, cy + 70 * s],
            radius=int(18 * s),
            fill=(0, 0, 0, 110),
        )
        d.text((cx - tw / 2, cy - 90 * s), title, font=f_big, fill=(255, 255, 255, 255))
        d.text((cx - sw / 2, cy + 10 * s), task, font=f_sub, fill=(220, 220, 220, 255))
    else:
        # top-left room chip
        f_title = _font(int(40 * s))
        f_sub = _font(int(28 * s))
        label = f"{title}  |  {task}"
        lw = d.textlength(label, font=f_sub)
        d.rounded_rectangle(
            [24 * s, 20 * s, 24 * s + lw + 32 * s, 96 * s],
            radius=int(12 * s),
            fill=(0, 0, 0, 110),
        )
        d.text((40 * s, 28 * s), title, font=f_title, fill=(255, 255, 255, 255))
        d.text((40 * s, 72 * s), task, font=f_sub, fill=(210, 210, 210, 255))

    if phase and not intro:
        f_phase = _font(int(30 * s))
        pw = d.textlength(phase, font=f_phase)
        d.rounded_rectangle(
            [w - pw - 56 * s, 20 * s, w - 24 * s, 68 * s],
            radius=int(12 * s),
            fill=(30, 90, 160, 140),
        )
        d.text(
            (w - pw - 40 * s, 28 * s), phase, font=f_phase, fill=(255, 255, 255, 255)
        )

    if banner:
        f_banner = _font(int(72 * s))
        bw = d.textlength(banner, font=f_banner)
        color = (30, 160, 60, 220) if banner_ok else (190, 40, 40, 220)
        cx, cy = w / 2, h * 0.82
        d.rounded_rectangle(
            [cx - bw / 2 - 36 * s, cy - 60 * s, cx + bw / 2 + 36 * s, cy + 40 * s],
            radius=int(16 * s),
            fill=color,
        )
        d.text(
            (cx - bw / 2, cy - 52 * s), banner, font=f_banner, fill=(255, 255, 255, 255)
        )

    return np.asarray(img.convert("RGB"))


# ----------------------------------------------------------------------
# Video writer: H.264 via imageio-ffmpeg, cv2 fallback (no subprocess)
# ----------------------------------------------------------------------


class VideoWriter:
    """Encode RGB frames to mp4; falls back to cv2 when ffmpeg can't spawn.

    imageio-ffmpeg spawns an ffmpeg subprocess for true H.264 + bitrate
    control. After ``gs.init`` on Windows, taichi's handle state can make
    subprocess creation fail (WinError 6); in that case we degrade to
    cv2.VideoWriter (mp4v, in-process encoder) so the demo still records.
    """

    def __init__(
        self, path: Path, fps: int, bitrate: str, size: tuple[int, int]
    ) -> None:
        self._backend = "ffmpeg"
        self._ffmpeg = None
        self._cv2 = None
        try:
            import imageio.v2 as imageio

            self._ffmpeg = imageio.get_writer(
                str(path),
                fps=fps,
                codec="libx264",
                bitrate=bitrate,
                macro_block_size=1,
                ffmpeg_log_level="error",
            )
            # force the ffmpeg subprocess to spawn now (lazy in imageio)
            self._ffmpeg.append_data(np.zeros((size[1], size[0], 3), dtype=np.uint8))
        except Exception as exc:  # noqa: BLE001
            logger.warning("ffmpeg writer unavailable (%s); using cv2 mp4v", exc)
            self._ffmpeg = None
            import cv2

            self._backend = "cv2"
            self._cv2 = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                size,
            )
            self._cv2.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)

    @property
    def backend(self) -> str:
        """Active encoder backend (``ffmpeg`` or ``cv2``)."""
        return self._backend

    def append(self, rgb: np.ndarray) -> None:
        """Append one RGB uint8 frame."""
        if self._ffmpeg is not None:
            self._ffmpeg.append_data(rgb)
        else:
            import cv2

            self._cv2.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        """Flush and finalize the video file."""
        if self._ffmpeg is not None:
            self._ffmpeg.close()
        elif self._cv2 is not None:
            self._cv2.release()


# ----------------------------------------------------------------------
# Camera: low, close, slow orbit
# ----------------------------------------------------------------------


class OrbitCamera:
    """Moving camera on a circle around the counter.

    The camera is render-only, so moving it never disturbs the physics.
    Modes:

    - ``arc``   – slow drift of ``span_deg`` across the take, fixed lookat.
    - ``orbit`` – continuous revolution at ``speed_deg_s`` around the
      workspace; every side is shown over one revolution (no blind spots).
    - ``track`` – same continuous revolution, but the lookat point follows
      the pick object so the action stays centred in frame.
    """

    def __init__(
        self,
        cam,
        mode: str = "arc",
        span_deg: float = 24.0,
        radius: float = 1.75,
        fps: int = 30,
        speed_deg_s: float = 15.0,
        lookat_fn=None,
    ) -> None:
        self.cam = cam
        self.mode = mode
        self.radius = radius if mode == "arc" else 1.30
        self.height = 1.02
        self.base_lookat = (COUNTER_CENTER[0] - 0.05, COUNTER_CENTER[1], 0.42)
        # arc starts on the -y side: a 3/4 front view that keeps the robot
        # from blocking the workspace (it stands between camera and backdrop)
        self.theta0 = np.deg2rad(247.0)
        self.span = np.deg2rad(span_deg)
        self.rate = np.deg2rad(speed_deg_s) / max(fps, 1)
        self.lookat_fn = lookat_fn
        self.progress = 0.0
        self.total = 400.0  # refined once the take length is known

    def pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Current ``(pos, lookat)`` on the camera circle."""
        if self.mode in ("orbit", "track"):
            theta = self.theta0 + self.rate * self.progress
        else:
            theta = self.theta0 + self.span * min(self.progress / self.total, 1.0)
        lookat = self.base_lookat
        if self.mode == "track" and self.lookat_fn is not None:
            try:
                lookat = self.lookat_fn()
            except Exception:  # noqa: BLE001 - fall back to the fixed lookat
                pass
        # the camera always circles the counter centre (never the object
        # itself) so it stays inside the room shell
        pos = (
            self.base_lookat[0] + self.radius * float(np.cos(theta)),
            self.base_lookat[1] + self.radius * float(np.sin(theta)),
            self.height,
        )
        return pos, lookat

    def step(self) -> None:
        """Advance the camera by one rendered frame."""
        self.progress += 1.0
        pos, lookat = self.pose()
        self.cam.set_pose(pos=pos, lookat=lookat)


# ----------------------------------------------------------------------
# Scene construction
# ----------------------------------------------------------------------


def build_room_scene(gs, room: dict, library: RoboTwinObjectLibrary):
    """Build the Genesis scene for one room; return (scene, robot, entities)."""
    scene = gs.Scene(
        show_viewer=False,
        vis_options=gs.options.VisOptions(
            ambient_light=(0.42, 0.42, 0.42),
            lights=[
                dict(
                    type="directional",
                    dir=(-1.0, -1.0, -1.6),
                    color=(1.0, 0.97, 0.92),
                    intensity=6.0,
                )
            ],
        ),
    )

    # room shell: thick colored floor slab (top exactly at z=0, doubles as
    # the collision ground so the default checkered plane stays hidden)
    # + full 4-wall enclosure so orbiting cameras never see the void
    scene.add_entity(
        gs.morphs.Box(
            size=(4.2, 4.2, 0.10),
            pos=(0.3, 0.0, -0.05),
            fixed=True,
        ),
        surface=gs.surfaces.Default(color=room["floor_color"], roughness=0.95),
    )
    for size, pos in [
        ((0.04, 4.2, 2.4), (1.80, 0.0, 1.20)),
        ((0.04, 4.2, 2.4), (-1.80, 0.0, 1.20)),
        ((4.2, 0.04, 2.4), (0.3, 1.90, 1.20)),
        ((4.2, 0.04, 2.4), (0.3, -1.90, 1.20)),
    ]:
        scene.add_entity(
            gs.morphs.Box(size=size, pos=pos, fixed=True),
            surface=gs.surfaces.Default(color=room["wall_color"], roughness=0.95),
        )

    # counter / worktop (the one procedural piece — large tables are a
    # registered asset gap, see data/recipes/asset_gap_list.yaml)
    scene.add_entity(
        gs.morphs.Box(
            size=COUNTER_SIZE,
            pos=(COUNTER_CENTER[0], COUNTER_CENTER[1], COUNTER_Z / 2),
            fixed=True,
        ),
        surface=gs.surfaces.Default(color=(0.64, 0.50, 0.36, 1.0), roughness=0.8),
    )

    # FR3 arm (URDF needs explicit actuator gains, restored after build)
    robot = scene.add_entity(
        gs.morphs.URDF(file=str(FR3_URDF), fixed=True, links_to_keep=[EE_LINK])
    )

    entities: dict[str, object] = {}
    for class_name, pos in room["anchors"]:
        entities[f"anchor:{class_name}"] = library.spawn_in_scene(
            scene, class_name, pos=(pos[0], pos[1], pos[2] + 0.02)
        )
    for class_name, pos in room["decor"]:
        entities[f"decor:{class_name}"] = library.spawn_in_scene(
            scene, class_name, pos=(pos[0], pos[1], pos[2] + 0.10)
        )
    # target container settles on the counter
    entities["target"] = library.spawn_in_scene(
        scene, room["target"], pos=(TARGET_XY[0], TARGET_XY[1], COUNTER_Z + 0.10)
    )
    # pick object and target container are pinned deterministically after
    # build (dropping from a height topples bottles over)
    entities["pick"] = library.spawn_in_scene(
        scene, room["pick"], pos=(SPAWN_XY[0], SPAWN_XY[1], COUNTER_Z + 0.20)
    )
    return scene, robot, entities


def pin_on_surface(scene, entity, xy: tuple[float, float], surface_z: float) -> None:
    """Teleport ``entity`` upright onto ``surface_z`` at ``xy`` with zero velocity.

    Deterministic alternative to dropping objects from a height (which
    topples tall objects like bottles). Identity quaternion is the upright
    orientation for RoboTwin-OD assets (same convention as grasp_all).
    """
    scene.step()  # make sure the AABB reflects the loaded pose
    aabb = _to_numpy(entity.get_AABB())
    qpos = _to_numpy(entity.get_qpos()).reshape(-1)
    qpos[:3] = [xy[0], xy[1], qpos[2] + (surface_z - aabb[0, 2]) + 0.001]
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    entity.set_qpos(qpos)
    logger.debug(
        "pin_on_surface %s: aabb %s -> qpos %s",
        xy,
        np.round(aabb, 3).tolist(),
        np.round(qpos[:3], 3).tolist(),
    )
    if hasattr(entity, "set_dofs_velocity"):
        try:
            import torch

            entity.set_dofs_velocity(torch.zeros(entity.n_dofs))
        except Exception:  # noqa: BLE001
            pass


# ----------------------------------------------------------------------
# Pick-and-place into a container (adapted from grasp_all_objects.run_one_class)
# ----------------------------------------------------------------------


def goto_cartesian(
    scene,
    robot,
    pos: np.ndarray,
    n_steps: int,
    grasper: SuctionGrasper | None = None,
    on_step=None,
    tol: float = 0.02,
    max_rounds: int = 3,
    gain: float = 0.6,
) -> tuple[np.ndarray, bool]:
    """Move the EE to a Cartesian target with IK-residual refinement.

    Genesis IK occasionally converges to a poor local branch (observed: EE
    6 cm short on descent, or 19 cm off / wrong direction on lift). Each
    round executes the motion, measures the true EE position, and feeds the
    residual back (damped by ``gain`` to avoid oscillation on far targets)
    into the next IK target until within ``tol``.
    """
    target = np.asarray(pos, dtype=np.float64)
    offset = np.zeros(3)
    ee_link = robot.get_link(EE_LINK)
    ee_pos = _to_numpy(ee_link.get_pos())
    for round_i in range(max_rounds):
        q = ik_to(robot, target + offset)
        goto_joints(scene, robot, q, grasper=grasper, n_steps=n_steps, on_step=on_step)
        ee_pos = _to_numpy(ee_link.get_pos())
        err = target - ee_pos
        if float(np.linalg.norm(err)) <= tol:
            return ee_pos, True
        offset = offset + gain * err
        logger.debug(
            "goto_cartesian round %d: target=%s ee=%s err=%.3f",
            round_i,
            np.round(target, 3).tolist(),
            np.round(ee_pos, 3).tolist(),
            float(np.linalg.norm(err)),
        )
    return ee_pos, False


def run_room_task(
    scene,
    robot,
    pick_obj,
    target_obj,
    router: PlannerRouter,
    record: GraspRecord,
    on_step=None,
    set_phase=None,
    quick: bool = False,
) -> None:
    """Pick ``pick_obj`` from the counter and place it into ``target_obj``.

    Unlike the bare-table benchmark, the release point is computed from the
    target container's AABB so the object lands *inside* the container.
    """
    grasper = SuctionGrasper(robot=robot, obj=pick_obj, ee_link_name=EE_LINK)
    n_descend = 12 if quick else 25
    n_settle = 20 if quick else 50

    def phase(p: GraspPhase) -> None:
        if set_phase is not None:
            set_phase(p)

    def current_q() -> np.ndarray:
        return _to_numpy(robot.get_qpos()).reshape(-1)[:7]

    # -- reset to home ---------------------------------------------------
    phase(GraspPhase.RESET)
    robot.set_qpos(HOME_QPOS)
    for _ in range(10):
        robot.control_dofs_position(HOME_QPOS)
        scene.step()
        if on_step is not None:
            on_step(HOME_QPOS)

    # -- measure the settled pick object ----------------------------------
    aabb = _to_numpy(pick_obj.get_AABB())
    obj_h = float(aabb[1, 2] - aabb[0, 2])
    center_xy = (aabb[0, :2] + aabb[1, :2]) / 2.0
    top_z = float(aabb[1, 2])
    logger.debug(
        "measure pick: aabb=%s qpos=%s",
        np.round(aabb, 3).tolist(),
        np.round(_to_numpy(pick_obj.get_qpos()).reshape(-1)[:3], 3).tolist(),
    )
    if obj_h <= 1e-4 or obj_h > 0.6:
        record.status = "load_fail"
        record.message = f"implausible object height {obj_h:.3f} m"
        return
    rest_qpos = _to_numpy(pick_obj.get_qpos()).reshape(-1).copy()

    # -- measure the target container --------------------------------------
    t_aabb = _to_numpy(target_obj.get_AABB())
    tx, ty = (float(v) for v in (t_aabb[0, :2] + t_aabb[1, :2]) / 2.0)
    tray_top = float(t_aabb[1, 2])
    half_xy = float(min(t_aabb[1, 0] - t_aabb[0, 0], t_aabb[1, 1] - t_aabb[0, 1])) / 2

    # -- pre-grasp (planned long move) -------------------------------------
    phase(GraspPhase.PRE_GRASP)
    try:
        path = router.plan(
            robot,
            current_q(),
            np.array([center_xy[0], center_xy[1], top_z + 0.12]),
            GRASP_QUAT,
            num_waypoints=15 if quick else 40,
        )
    except Exception as exc:
        record.status = "plan_fail"
        record.message = f"pre-grasp: {exc}"
        return
    record.planner = router.last_used
    follow_path(scene, robot, path, grasper=None, steps_per_waypoint=2, on_step=on_step)

    # -- descend + suction grasp -------------------------------------------
    phase(GraspPhase.DESCEND)
    ee_pos, _ = goto_cartesian(
        scene,
        robot,
        np.array([center_xy[0], center_xy[1], top_z - 0.004]),
        n_steps=n_descend,
        on_step=on_step,
        tol=0.03,
    )
    if (
        float(np.linalg.norm(ee_pos[:2] - center_xy)) > 0.04
        or abs(ee_pos[2] - top_z) > 0.05
    ):
        record.status = "grasp_fail"
        record.message = f"EE did not reach grasp point (ee={np.round(ee_pos, 3)})"
        return
    phase(GraspPhase.GRASP)
    grasper.attach()
    logger.debug(
        "grasp: ee=%s obj=%s offset=%s",
        np.round(grasper.ee_pose()[0], 3).tolist(),
        np.round(grasper.obj_base_pose()[0], 3).tolist(),
        np.round(grasper._offset_pos, 3).tolist(),
    )

    # -- lift ---------------------------------------------------------------
    phase(GraspPhase.LIFT)
    obj_z_before, _ = grasper.obj_base_pose()
    goto_cartesian(
        scene,
        robot,
        np.array([center_xy[0], center_xy[1], top_z + 0.12]),
        n_steps=n_descend,
        grasper=grasper,
        on_step=on_step,
    )
    obj_pos, _ = grasper.obj_base_pose()
    logger.debug(
        "lift: ee=%s obj=%s (before z=%.3f)",
        np.round(grasper.ee_pose()[0], 3).tolist(),
        np.round(obj_pos, 3).tolist(),
        obj_z_before[2],
    )
    if obj_pos[2] < obj_z_before[2] + 0.05:
        record.status = "lift_fail"
        record.message = (
            f"object did not lift (z {obj_z_before[2]:.3f} -> {obj_pos[2]:.3f})"
        )
        grasper.detach()
        return

    # -- transport above the container (planned) -----------------------------
    phase(GraspPhase.TRANSPORT)
    try:
        path = router.plan(
            robot,
            current_q(),
            np.array([tx, ty, tray_top + obj_h + 0.15]),
            GRASP_QUAT,
            num_waypoints=15 if quick else 40,
        )
    except Exception as exc:
        record.status = "transport_fail"
        record.message = f"transport: {exc}"
        grasper.detach()
        return
    follow_path(
        scene, robot, path, grasper=grasper, steps_per_waypoint=2, on_step=on_step
    )
    if not grasper.is_holding():
        record.status = "transport_fail"
        record.message = "object dropped during transport"
        grasper.detach()
        return

    # -- descend into the container + release --------------------------------
    phase(GraspPhase.PLACE)
    goto_cartesian(
        scene,
        robot,
        np.array([tx, ty, tray_top + obj_h + 0.02]),
        n_steps=n_descend,
        grasper=grasper,
        on_step=on_step,
    )
    grasper.follow_step()
    phase(GraspPhase.RELEASE)
    grasper.detach()
    # upright, zero-velocity release just above the container rim
    place_qpos = rest_qpos.copy()
    base_off = rest_qpos[2] - (top_z - obj_h)  # base offset above support
    place_qpos[:3] = [tx, ty, tray_top + base_off + 0.005]
    pick_obj.set_qpos(place_qpos)
    if hasattr(pick_obj, "set_dofs_velocity"):
        try:
            import torch

            pick_obj.set_dofs_velocity(torch.zeros(pick_obj.n_dofs))
        except Exception:  # noqa: BLE001
            pass
    q_hold = current_q()
    for _ in range(n_settle):
        scene.step()
        if on_step is not None:
            on_step(q_hold)
    phase(GraspPhase.RETREAT)
    goto_cartesian(
        scene,
        robot,
        np.array([tx, ty, tray_top + obj_h + 0.18]),
        n_steps=15,
        on_step=on_step,
        max_rounds=1,  # visual-only move: a single unrefined IK is fine
    )

    # -- judge: inside the container footprint, near its rest height ---------
    final = _to_numpy(pick_obj.get_qpos()).reshape(-1)[:3]
    record.final_pos = [float(x) for x in final]
    xy_err = float(np.linalg.norm(final[:2] - np.array([tx, ty])))
    in_footprint = xy_err < max(0.08, half_xy * 0.9)
    near_rest_height = COUNTER_Z - 0.03 < final[2] < tray_top + base_off + 0.06
    if in_footprint and near_rest_height:
        record.status = "success"
    else:
        record.status = "place_fail"
        record.message = f"final pos {np.round(final, 3)}, xy_err {xy_err:.3f}"


# ----------------------------------------------------------------------
# Per-room driver
# ----------------------------------------------------------------------


def run_room(args: argparse.Namespace, room_name: str) -> GraspRecord:
    """Build one room, execute its task, encode the video, return the record."""
    import genesis as gs

    room = ROOMS[room_name]
    record = GraspRecord(class_name=room_name)
    t0 = time.time()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"{room_name}.mp4"
    w, h = (int(v) for v in args.resolution.split("x"))

    library = RoboTwinObjectLibrary(args.objects_dir)
    # resolve pick/target instances up-front for the record
    pick_inst = library.get_instance(room["pick"])
    record.instance_index = pick_inst.index
    record.kind = pick_inst.kind

    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(backend=backend, logging_level="warning")

    scene, robot, entities = build_room_scene(gs, room, library)
    cam = scene.add_camera(
        res=(w, h),
        pos=(-0.25, -1.61, 1.02),
        lookat=(COUNTER_CENTER[0] - 0.05, COUNTER_CENTER[1], 0.42),
        fov=46,
    )
    logger.info("[%s] building scene...", room_name)
    scene.build()

    dofs7 = list(range(7))
    robot.set_dofs_kp(FR3_KP, dofs_idx_local=dofs7)
    robot.set_dofs_kv(FR3_KV, dofs_idx_local=dofs7)
    robot.set_dofs_damping(FR3_DAMPING, dofs_idx_local=dofs7)
    robot.set_dofs_frictionloss(FR3_FRICTIONLOSS, dofs_idx_local=dofs7)
    robot.set_dofs_armature(FR3_ARMATURE, dofs_idx_local=dofs7)

    cam_mode = "arc" if args.static_cam else args.cam_mode

    def pick_lookat() -> tuple[float, float, float]:
        """Centre of the pick object's AABB (track mode follows the action)."""
        aabb = _to_numpy(entities["pick"].get_AABB())
        c = (aabb[0] + aabb[1]) / 2.0
        return (float(c[0]), float(c[1]), float(c[2]))

    orbit = OrbitCamera(
        cam,
        mode=cam_mode,
        span_deg=0.0 if args.static_cam else 24.0,
        fps=args.fps,
        speed_deg_s=args.cam_speed,
        lookat_fn=pick_lookat if cam_mode == "track" else None,
    )
    state = {"phase": PHASE_LABELS[GraspPhase.RESET], "frame": 0}
    intro_frames = 8 if args.quick else 30
    writer = VideoWriter(video_path, args.fps, args.bitrate, (w, h))
    logger.info("[%s] video encoder: %s", room_name, writer.backend)

    def render_frame() -> None:
        orbit.step()
        out = cam.render()
        rgb = out[0] if isinstance(out, tuple) else out
        frame = overlay_frame(
            rgb,
            room["title"],
            room["task"],
            phase=state["phase"],
            intro=state["frame"] < intro_frames,
        )
        writer.append(frame)
        state["frame"] += 1

    def on_step(_q: np.ndarray) -> None:
        render_frame()

    def set_phase(p: GraspPhase) -> None:
        state["phase"] = PHASE_LABELS[p]

    try:
        # pin pick object, target container and decor deterministically,
        # then a short settle while the title card is on screen
        pin_on_surface(scene, entities["pick"], SPAWN_XY, COUNTER_Z)
        pin_on_surface(scene, entities["target"], TARGET_XY, COUNTER_Z)
        for class_name, pos in room["decor"]:
            pin_on_surface(
                scene, entities[f"decor:{class_name}"], (pos[0], pos[1]), pos[2]
            )
        for _ in range(15 if args.quick else 30):
            scene.step()
            render_frame()

        run_room_task(
            scene,
            robot,
            entities["pick"],
            entities["target"],
            PlannerRouter(args.planner),
            record,
            on_step=on_step,
            set_phase=set_phase,
            quick=args.quick,
        )
    except Exception as exc:  # noqa: BLE001 - a room crash must not kill the take
        record.status = "error"
        record.message = f"{type(exc).__name__}: {exc}"
        logger.error("[%s] crashed:\n%s", room_name, traceback.format_exc())

    # end banner: hold the final view with SUCCESS/FAILED overlay
    ok = record.status == "success"
    banner_frames = 15 if args.quick else 45
    state["phase"] = "done"
    try:
        for _ in range(banner_frames):
            orbit.step()
            out = cam.render()
            rgb = out[0] if isinstance(out, tuple) else out
            frame = overlay_frame(
                rgb,
                room["title"],
                room["task"],
                phase=state["phase"],
                banner="TASK SUCCESS" if ok else "TASK FAILED",
                banner_ok=ok,
            )
            writer.append(frame)
            state["frame"] += 1
    except Exception:  # noqa: BLE001
        logger.warning("[%s] banner render failed", room_name)
    writer.close()

    record.duration_s = time.time() - t0
    payload = asdict(record)
    payload.update(
        {
            "room": room_name,
            "task": room["task"],
            "pick": room["pick"],
            "target": room["target"],
            "video": video_path.name,
            "frames": state["frame"],
        }
    )
    (out_dir / f"record_{room_name}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "[%s] -> %s (%.1f s, %d frames, %s)",
        room_name,
        record.status,
        record.duration_s,
        state["frame"],
        video_path,
    )
    return record


# ----------------------------------------------------------------------
# Main: one subprocess per room for multi-room runs (VRAM bound + isolation)
# ----------------------------------------------------------------------


def _record_from_json(path: Path) -> GraspRecord | None:
    """Read a per-room record file back into a GraspRecord."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return GraspRecord(
            **{f: v for f, v in raw.items() if f in GraspRecord.__dataclass_fields__}
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("failed to read %s: %s", path, exc)
        return None


def main() -> None:
    """Entry point: parse args and produce per-room demo videos."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--room",
        choices=["all", *sorted(ROOMS)],
        default="all",
        help="Room to render (default: all five)",
    )
    parser.add_argument(
        "--objects-dir",
        type=Path,
        default=REPO_ROOT / "assets" / "robotwin" / "objects" / "objects",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "home_demo")
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--planner", choices=["auto", "curobo", "ompl"], default="auto")
    parser.add_argument("--resolution", default="1920x1080", help="WxH")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--bitrate", default="4M", help="H.264 bitrate")
    parser.add_argument(
        "--static-cam",
        action="store_true",
        help="Disable camera motion entirely (overrides --cam-mode)",
    )
    parser.add_argument(
        "--cam-mode",
        choices=["arc", "orbit", "track"],
        default="track",
        help=(
            "Camera motion: arc = slow 24 deg drift, orbit = continuous "
            "revolution around the workspace, track = orbit while following "
            "the pick object (default: track)"
        ),
    )
    parser.add_argument(
        "--cam-speed",
        type=float,
        default=15.0,
        help="Orbit/track camera angular speed in deg/s (default: 15)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke mode: fewer settle/plan/banner steps",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Run rooms sequentially in this process (debugging)",
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

    rooms = sorted(ROOMS) if args.room == "all" else [args.room]

    if len(rooms) == 1 or args.in_process:
        records = [run_room(args, name) for name in rooms]
    else:
        records = []
        for name in rooms:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--room",
                name,
                "--objects-dir",
                str(args.objects_dir),
                "--out",
                str(args.out),
                "--backend",
                args.backend,
                "--planner",
                args.planner,
                "--resolution",
                args.resolution,
                "--fps",
                str(args.fps),
                "--bitrate",
                args.bitrate,
            ]
            if args.static_cam:
                cmd.append("--static-cam")
            else:
                cmd += ["--cam-mode", args.cam_mode, "--cam-speed", str(args.cam_speed)]
            if args.quick:
                cmd.append("--quick")
            logger.info("=== room %s (subprocess) ===", name)
            ret = subprocess.call(cmd)
            rec = _record_from_json(args.out / f"record_{name}.json")
            if rec is None:
                rec = GraspRecord(
                    class_name=name,
                    status="error",
                    message=f"room process crashed (exit={ret})",
                )
            records.append(rec)

    json_path, md_path = write_report(
        records,
        args.out,
        extra={
            "rooms": rooms,
            "backend": args.backend,
            "planner_mode": args.planner,
            "resolution": args.resolution,
            "videos": [f"{name}.mp4" for name in rooms],
        },
    )
    ok = sum(1 for r in records if r.status == "success")
    logger.info(
        "DONE: %d/%d rooms succeeded -> %s, %s", ok, len(records), json_path, md_path
    )


if __name__ == "__main__":
    main()
