#!/usr/bin/env python3
"""Family-check resolver for the three RoboCasa EFE suites.

Each frozen instance carries its fully composed kitchen world (robot,
fixtures, objects) plus the pickled env state; ``tools/robocasa_worlds.py``
turns that into a world package (rewritten MJCF + qpos/qvel + meta with
compiled world poses). This module binds every inventory task to its
upstream ``_check_success`` transcription:

* fixtures are selected with the same rules upstream uses (type joints /
  sites, nearest to the robot base — upstream parks the robot at
  ``init_robot_base_pos``);
* object radii / bounding boxes come from the mirrored object-library XMLs
  (``horizontal_radius_site`` / ``bottom_site`` / ``top_site``), exactly the
  values ``MujocoObject.horizontal_radius`` and ``get_bbox_points`` read;
* joint ranges come from the world meta, so ``rc_joint_state`` normalises
  exactly like ``OU.normalize_joint_value``;
* appliance toggles (microwave / coffee machine) and step counters
  (washed / broiled / scrubbed) become ``EfeRobocasaTask`` config.

The entry point :func:`resolve_robocasa` returns the same knowledge-base
mapping ``tools/compile_efe_suite.py`` writes for the other suites.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RC_WORLDS = REPO_ROOT / "assets" / "robocasa"
TASKS_MIRROR = REPO_ROOT / "data" / "efe" / "upstream" / "tasks"

SUITES = ("robocasa", "robocasa365", "robocasa-gr1")
ROBOT_BY_SUITE = {
    "robocasa": "franka_panda_omron",
    "robocasa365": "franka_panda_omron",
    "robocasa-gr1": "fourier_gr1",
}
GRIPPER_BODY = {
    "robocasa": "gripper0_right",
    "robocasa365": "gripper0_right",
}
MAX_STEPS = 9000  # EFE standard-mode ceiling

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Per-task binding context
# ---------------------------------------------------------------------------


def _load_caches(suite: str, task: str) -> dict[str, Any]:
    instance = (
        TASKS_MIRROR / suite / f"{task}-privileged" / "environment" / "instance.npz"
    )
    with zipfile.ZipFile(instance) as archive:
        return json.loads(np.load(archive.open("caches.npy"), allow_pickle=True).item())


class RcBindingError(Exception):
    """A family/task binding failed; the task goes pending with this text."""


Reason = RcBindingError  # compact alias used throughout the builders


class Ctx:
    """Everything needed to bind one task's success spec."""

    def __init__(self, suite: str, task: str) -> None:
        self.suite = suite
        self.task = task
        world_dir = RC_WORLDS / suite / "worlds"
        self.meta = json.loads((world_dir / f"{task}.meta.json").read_text())
        self.xml_path = world_dir / f"{task}.xml"
        self.world_xml = self.xml_path.read_text(encoding="utf-8")
        self.caches = _load_caches(suite, task)
        self.env: dict[str, Any] = self.caches.get("env") or {}
        self.fixtures: dict[str, dict[str, Any]] = {
            k[len("fixture:") :]: v
            for k, v in self.caches.items()
            if k.startswith("fixture:")
        }
        self.joints: dict[str, dict[str, Any]] = {
            j["name"]: j for j in self.meta["joints"]
        }
        self.sites: dict[str, dict[str, Any]] = self.meta["sites"]
        self.bodies: dict[str, dict[str, Any]] = self.meta["bodies"]
        self.object_names: list[str] = list(self.env.get("obj_body_id") or [])
        self.appliances: list[dict[str, Any]] = []
        self.timers: list[dict[str, Any]] = []
        self.extra_sites: dict[str, list[Any]] = {}
        self.qpos_overrides: dict[str, float] = {}
        self._library: dict[str, dict[str, Any] | None] = {}
        self._parent_map: dict[ET.Element, ET.Element] | None = None

    # -- world geometry ----------------------------------------------------

    def world(self, body: str) -> np.ndarray:
        try:
            return np.asarray(self.bodies[body]["world"], dtype=float)
        except KeyError as exc:
            raise Reason(f"未知 body {body!r}") from exc

    def obj_body(self, obj: str) -> str:
        """World body for an env object name (``obj`` -> ``obj_main``)."""
        for candidate in (f"{obj}_main", obj):
            if candidate in self.bodies:
                return candidate
        raise Reason(f"物体 {obj!r} 无对应世界 body")

    def obj_world(self, obj: str) -> np.ndarray:
        return self.world(self.obj_body(obj))

    def robot_base_xy(self) -> np.ndarray:
        for name in ("robot0_base", "robot0_base_main"):
            if name in self.bodies:
                return self.world(name)[:2]
        raise Reason("robot0_base 缺失")

    # -- fixtures ------------------------------------------------------------

    def fx_body(self, fx: str) -> str:
        root = self.fixtures[fx].get("_root_body") or "main"
        for candidate in (f"{fx}_{root}", fx):
            if candidate in self.bodies:
                return candidate
        raise Reason(f"夹具 {fx!r} 无对应世界 body")

    def fx_world(self, fx: str) -> np.ndarray:
        return self.world(self.fx_body(fx))

    def fx_with_joints(self, *needles: str) -> list[str]:
        out = []
        for name, info in self.fixtures.items():
            joints = " ".join(info.get("_joints") or [])
            if all(n in joints for n in needles):
                out.append(name)
        return out

    def fx_named(self, needle: str) -> list[str]:
        return [n for n in self.fixtures if needle in n]

    def fx_nearest(self, cands: list[str], xy: np.ndarray) -> str:
        if not cands:
            raise Reason("无候选夹具")
        return min(
            cands, key=lambda f: float(np.linalg.norm(self.fx_world(f)[:2] - xy))
        )

    def site(self, fx: str, local: str) -> str:
        name = f"{fx}_{local}"
        if name not in self.sites:
            raise Reason(f"夹具 {fx!r} 缺站点 {local!r}")
        return name

    def counters(self) -> list[str]:
        out = [
            n
            for n, info in self.fixtures.items()
            if (
                n.startswith("counter")
                or ("island" in n and "stool" not in n and "cube" not in n)
            )
            and "int_p0" in (info.get("_sites") or [])
        ]
        if not out:
            out = [
                n
                for n, info in self.fixtures.items()
                if n.startswith("counter") or ("island" in n and "stool" not in n)
            ]
        if not out:
            raise Reason("无 counter 夹具")
        return out

    def cabinets(self) -> list[str]:
        out = [
            n
            for n, info in self.fixtures.items()
            if (
                "doorhinge" in " ".join(info.get("_joints") or [])
                or "leftdoorhinge" in " ".join(info.get("_joints") or [])
            )
        ]
        if not out:
            raise Reason("无带门柜夹具")
        return out

    def cabinets_with_interior(self) -> list[str]:
        out = [
            f
            for f in self.cabinets()
            if "int_p0" in (self.fixtures[f].get("_sites") or [])
            or self._reg_regions(f)
        ]
        if not out:
            raise Reason("无带内腔的柜夹具")
        return out

    def sink(self) -> str:
        cands = self.fx_with_joints("handle_joint")
        if not cands:
            raise Reason("无水槽夹具")
        with_int = [
            f for f in cands if "int_p0" in (self.fixtures[f].get("_sites") or [])
        ]
        pool = with_int or cands
        return self.fx_nearest(pool, self.robot_base_xy())

    def stove(self) -> str:
        cands = self.fx_with_joints("knob_")
        stovetops = [n for n in cands if "stovetop" in n]
        if stovetops:
            return stovetops[0]
        cands = [n for n in cands if "oven" not in n and "toaster" not in n]
        if not cands:
            raise Reason("无灶台夹具")
        return cands[0]

    def microwave(self) -> str:
        cands = self.fx_with_joints("microjoint")
        if not cands:
            raise Reason("无微波炉夹具")
        return cands[0]

    def coffee_machine(self) -> str:
        cands = [
            n
            for n, info in self.fixtures.items()
            if "receptacle_place_site" in (info.get("_sites") or [])
        ]
        if not cands:
            raise Reason("无咖啡机夹具")
        return cands[0]

    # -- geom-level pseudo-sites ----------------------------------------------

    def geom_site(self, geom_name: str) -> None:
        """Register a geom (microwave button) as a (body, local) pseudo-site.

        Several appliance buttons exist as geoms on a body, not as bodies;
        distance checks need their world position, so the resolver bakes
        the parent body and local offset as an extra site.
        """
        if geom_name in self.bodies:
            return  # already a body; nothing to bake
        if self._parent_map is None:
            root = ET.fromstring(self.world_xml)
            self._parent_map = {c: p for p in root.iter() for c in p}
        for el in self._parent_map:
            if el.tag == "geom" and el.get("name") == geom_name:
                parent = self._parent_map.get(el)
                while parent is not None and parent.tag != "body":
                    parent = self._parent_map.get(parent)
                if parent is None or parent.get("name") is None:
                    raise Reason(f"geom {geom_name!r} 无父 body")
                local = [float(v) for v in (el.get("pos") or "0 0 0").split()]
                self.extra_sites[geom_name] = [parent.get("name"), local]
                return
        raise Reason(f"geom {geom_name!r} 不存在")

    # -- joints --------------------------------------------------------------

    def joint_range(self, jname: str) -> tuple[float, float]:
        record = self.joints.get(jname)
        rng = (record or {}).get("range")
        if not rng:
            raise Reason(f"关节 {jname!r} 无 range")
        lo, hi = (float(v) for v in str(rng).split())
        return lo, hi

    def normalized_joint(self, jname: str, lo_th: float | None, hi_th: float | None):
        """rc_joint_state normalised over the joint's own range.

        Upstream normalises ``sign * qpos`` over a fixed window; a joint's
        compiled range is exactly that window (drawers ``[-d, 0]`` with
        ``sign=-1``, doors ``[0, pi/2]`` or ``[-pi/2, 0]`` with the matching
        sign), so range + sign reproduces ``OU.normalize_joint_value``.
        """
        lo, hi = self.joint_range(jname)
        sign = 1.0
        if lo < 0.0 <= hi:
            # drawer / door ranges like [-d, 0]: upstream multiplies by -1
            # and normalises over [0, d]
            sign, lo, hi = -1.0, 0.0, -lo
        params: dict[str, Any] = {
            "joint": jname,
            "mode": "normalized",
            "range": [round(lo, 6), round(hi, 6)],
            "sign": sign,
        }
        if lo_th is not None:
            params["min"] = lo_th
        if hi_th is not None:
            params["max"] = hi_th
        return {"type": "rc_joint_state", "params": params}

    def hinge_joints(self, fx: str) -> list[str]:
        prefix = f"{fx}_"
        out = [
            name
            for name in self.joints
            if name.startswith(prefix)
            and ("doorhinge" in name or "microjoint" in name or name.endswith("joint"))
            and "slide" not in name
            and "knob" not in name
            and "handle" not in name
            and "spout" not in name
            and "drawer" not in name
        ]
        return out

    def slide_joint(self, fx: str) -> str:
        prefix = f"{fx}_"
        out = [n for n in self.joints if n.startswith(prefix) and "slidejoint" in n]
        if not out:
            raise Reason(f"夹具 {fx!r} 无 slidejoint")
        return out[0]

    # -- object library (radii / bbox) ----------------------------------------

    def _object_dir(self, obj: str) -> Path | None:
        body = self.obj_body(obj)
        pattern = re.compile(rf'<body name="{re.escape(body)}"[^>]*>(.*?)</body>', re.S)
        block = pattern.search(self.world_xml)
        if block is None:
            return None
        mesh = re.search(r'mesh="([^"]+)"', block.group(1))
        if mesh is None:
            return None
        file_attr = re.search(
            rf'<mesh name="{re.escape(mesh.group(1))}"[^>]*file="([^"]+)"',
            self.world_xml,
        )
        if file_attr is None:
            return None
        path = Path(file_attr.group(1).replace("/", "\\"))
        # visual/model_x.obj -> object dir
        return path.parent.parent

    def library(self, obj: str) -> dict[str, Any] | None:
        """Parsed object-library XML values, cached per object."""
        if obj in self._library:
            return self._library[obj]
        result: dict[str, Any] | None = None
        obj_dir = self._object_dir(obj)
        if obj_dir is not None and obj_dir.is_dir():
            for xml in sorted(obj_dir.glob("*.xml")):
                try:
                    root = ET.fromstring(
                        xml.read_text(encoding="utf-8", errors="ignore")
                    )
                except ET.ParseError:
                    continue
                sites = {
                    (s.get("name") or ""): (s.get("pos") or "0 0 0").split()
                    for s in root.iter("site")
                }
                hr = sites.get("horizontal_radius_site")
                bottom = sites.get("bottom_site")
                top = sites.get("top_site")
                if hr and bottom and top:
                    hr_x, hr_y = float(hr[0]), float(hr[1] if len(hr) > 1 else hr[0])
                    bottom_v = np.array([float(v) for v in bottom])
                    top_v = np.array([float(v) for v in top])
                    center = (bottom_v + top_v) / 2.0
                    half = np.array([hr_x, hr_y, top_v[2] - center[2]])
                    corners = [
                        (center + half * np.array(signs)).round(6).tolist()
                        for signs in (
                            (-1, -1, -1),
                            (1, -1, -1),
                            (-1, 1, -1),
                            (-1, -1, 1),
                            (1, 1, 1),
                            (-1, 1, 1),
                            (1, -1, 1),
                            (1, 1, -1),
                        )
                    ]
                    result = {
                        "horizontal_radius": round(hr_x, 6),
                        "bbox": corners,
                    }
                    break
        self._library[obj] = result
        return result

    def obj_radius(self, obj: str) -> float:
        lib = self.library(obj)
        if lib is None:
            raise Reason(f"物体 {obj!r} 库 XML 缺失（horizontal_radius_site）")
        return float(lib["horizontal_radius"])

    def obj_bbox(self, obj: str) -> list[list[float]]:
        lib = self.library(obj)
        if lib is None:
            raise Reason(f"物体 {obj!r} 库 XML 缺失（bbox 站点）")
        return lib["bbox"]

    # -- shared spec fragments -------------------------------------------------

    def eef_sites(self) -> tuple[str, ...]:
        sites = [s for s in self.meta["eef_sites"] if "cylinder" not in s]
        if not sites:
            raise Reason("无 eef 站点")
        return tuple(sites)

    def gripper_far(self, obj: str, th: float = 0.25) -> dict[str, Any]:
        return {
            "type": "rc_gripper_far",
            "params": {
                "target": self.obj_body(obj),
                "sites": list(self.eef_sites()),
                "threshold": th,
            },
        }

    def contact(self, obj: str, fixture: str) -> dict[str, Any]:
        return {
            "type": "rc_contact",
            "params": {"object": self.obj_body(obj), "fixture": self.fx_body(fixture)},
        }

    def contact_any_counter(self, obj: str) -> dict[str, Any]:
        return {
            "type": "rc_contact_any",
            "params": {
                "object": self.obj_body(obj),
                "fixtures": [self.fx_body(c) for c in self.counters()],
            },
        }

    def in_receptacle(
        self, obj: str, recep: str, th: float | None = None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "object": self.obj_body(obj),
            "receptacle": self.obj_body(recep),
        }
        if th is None:
            params["radius"] = self.obj_radius(recep)
            params["factor"] = 0.7
        else:
            params["threshold"] = th
        return {"type": "rc_in_receptacle", "params": params}

    def inside(
        self,
        obj: str,
        fixture: str,
        partial: bool = False,
    ) -> dict[str, Any]:
        fx_sites = self.fixtures[fixture].get("_sites") or []
        params: dict[str, Any] = {
            "object": self.obj_body(obj),
            "fixture": self.fx_body(fixture),
        }
        if "int_p0" in fx_sites:
            params["fixture_sites"] = [
                self.site(fixture, s) for s in ("int_p0", "int_px", "int_py", "int_pz")
            ]
        else:
            regions = self._reg_regions(fixture)
            if not regions:
                raise Reason(f"夹具 {fixture!r} 无 int 站点也无 reg 区域")
            params["regions"] = regions
        if partial:
            params["tolerance"] = 0.0
        else:
            params["object_bbox"] = self.obj_bbox(obj)
            params["tolerance"] = 0.05
        return {"type": "rc_inside", "params": params}

    def _reg_regions(self, fixture: str) -> list[dict[str, Any]]:
        """Robocasa 1.0.1 ``reg_*`` region geoms as center/halfsize boxes."""
        prefix = f"{fixture}_reg_"
        regions: list[dict[str, Any]] = []
        for elem in re.finditer(r"<geom\b[^>]*>", self.world_xml):
            tag = elem.group(0)
            name = re.search(r'name="([^"]+)"', tag)
            if name is None or not name.group(1).startswith(prefix):
                continue
            pos = re.search(r'pos="([^"]+)"', tag)
            size = re.search(r'size="([^"]+)"', tag)
            if pos is None or size is None:
                continue
            center = [float(v) for v in pos.group(1).split()]
            half = [float(v) for v in size.group(1).split()]
            if 'type="cylinder"' in tag:
                half = [half[0], half[0], half[1]]
            regions.append(
                {
                    "center": [round(v, 6) for v in center],
                    "halfsize": [round(v, 6) for v in half[:3]],
                }
            )
        return regions

    def water_on(self) -> dict[str, Any]:
        sink = self.sink()
        return {
            "type": "rc_joint_state",
            "params": {
                "joint": f"{sink}_handle_joint",
                "mode": "mod2pi",
                "min": 0.40,
                "max": np.pi,
            },
        }

    def spout_ori(self, orientation: str) -> dict[str, Any]:
        sink = self.sink()
        return {
            "type": "rc_joint_state",
            "params": {
                "joint": f"{sink}_spout_joint",
                "mode": "spout_ori",
                "orientation": orientation,
            },
        }

    def appliance_spec(
        self,
        name: str,
        on: bool,
        *,
        button_far: str | None = None,
        button_th: float = 0.15,
    ) -> dict[str, Any]:
        conds: list[dict[str, Any]] = [
            {
                "type": "rc_appliance",
                "params": {"name": name, "state": "on" if on else "off"},
            }
        ]
        if button_far is not None:
            conds.append(
                {
                    "type": "rc_gripper_far",
                    "params": {
                        "target": button_far,
                        "sites": list(self.eef_sites()),
                        "threshold": button_th,
                    },
                }
            )
        return {"type": "all", "params": {"conditions": conds}}

    def register_microwave(self, initial_on: bool = False) -> str:
        fx = self.microwave()
        joint = f"{fx}_microjoint"
        lo, hi = self.joint_range(joint)
        self.appliances.append(
            {
                "name": fx,
                "door_joint": joint,
                "door_range": [lo, hi],
                "start_button": f"{fx}_start_button",
                "stop_button": f"{fx}_stop_button",
                "gripper_body": GRIPPER_BODY[self.suite],
                "initial_on": initial_on,
            }
        )
        return fx

    def register_coffee_machine(self) -> str:
        fx = self.coffee_machine()
        self.appliances.append(
            {
                "name": fx,
                "start_button": f"{fx}_start_button",
                "gripper_body": GRIPPER_BODY[self.suite],
                "initial_on": False,
            }
        )
        return fx


def _all(*conds: dict[str, Any]) -> dict[str, Any]:
    flat = [c for c in conds]
    if len(flat) == 1:
        return flat[0]
    return {"type": "all", "params": {"conditions": flat}}


def _any(*conds: dict[str, Any]) -> dict[str, Any]:
    return {"type": "any", "params": {"conditions": list(conds)}}


def _not(cond: dict[str, Any]) -> dict[str, Any]:
    return {"type": "not", "params": {"condition": cond}}


# ---------------------------------------------------------------------------
# Family builders
# ---------------------------------------------------------------------------


def _b_drawer(ctx: Ctx, open_: bool) -> dict[str, Any]:
    side = ctx.env.get("drawer_side")
    cands = ctx.fx_with_joints("slidejoint")
    if side:  # upstream: drawer on the robot's left/right
        robot = ctx.robot_base_xy()
        cands = [
            f
            for f in cands
            if np.sign(ctx.fx_world(f)[1] - robot[1]) == (1 if side == "left" else -1)
        ] or cands
    fx = ctx.fx_nearest(cands, ctx.robot_base_xy())
    joint = ctx.slide_joint(fx)
    if not open_:
        # 上游 _reset_internal：close 任务把抽屉打到 0.90–1.00 开度。
        lo, hi = ctx.joint_range(joint)
        ctx.qpos_overrides[joint] = 0.95 * hi if hi > 0 else 0.95 * lo
    return ctx.normalized_joint(joint, 0.95 if open_ else None, None if open_ else 0.05)


def _b_door(ctx: Ctx, open_: bool, double: bool) -> dict[str, Any]:
    if double:
        cands = [f for f in ctx.fx_with_joints("leftdoorhinge")]
    else:
        cands = [
            f
            for f in ctx.cabinets()
            if "leftdoorhinge" not in " ".join(ctx.fixtures[f].get("_joints") or [])
        ]
    fx = ctx.fx_nearest(cands, ctx.robot_base_xy())
    joints = ctx.hinge_joints(fx)
    if not joints:
        raise Reason(f"夹具 {fx!r} 无门铰链")
    if not open_:
        # 上游 _reset_internal：close 任务把门打到 0.90–1.00 开度。
        for joint in joints:
            lo, hi = ctx.joint_range(joint)
            ctx.qpos_overrides[joint] = 0.95 * hi if hi > 0 else 0.95 * lo
    specs = [
        ctx.normalized_joint(j, 0.90 if open_ else None, None if open_ else 0.05)
        for j in joints
    ]
    return _all(*specs)


def _b_microwave_press(ctx: Ctx, on: bool) -> dict[str, Any]:
    fx = ctx.register_microwave(initial_on=not on)
    button = f"{fx}_{'start' if on else 'stop'}_button"
    ctx.geom_site(button)
    return ctx.appliance_spec(fx, on, button_far=button)


def _b_sink_faucet(ctx: Ctx, on: bool) -> dict[str, Any]:
    joint = f"{ctx.sink()}_handle_joint"
    if not on:
        # 上游 _reset_internal：turn_off 任务每次 reset 把水龙头打到开
        # （rng.uniform(0.40, 0.50)），验证回放同样从开开始。
        ctx.qpos_overrides[joint] = 0.45
    cond: dict[str, Any] = {
        "type": "rc_joint_state",
        "params": {
            "joint": joint,
            "mode": "mod2pi",
            "min": 0.40,
            "max": np.pi,
        },
    }
    return cond if on else _not(cond)


def _b_sink_spout(ctx: Ctx) -> dict[str, Any]:
    behavior = ctx.env.get("behavior") or "center"
    return ctx.spout_ori(str(behavior))


def _b_stove(ctx: Ctx, on: bool) -> dict[str, Any]:
    stove = ctx.stove()
    knob = str(ctx.env.get("knob") or "")
    joints = [n for n in ctx.joints if n.startswith(f"{stove}_") and "knob" in n]
    if knob:
        joints = [j for j in joints if knob in j] or joints
    if not joints:
        raise Reason("无旋钮关节")
    if not on:
        # 上游 _reset_internal：turn_off 任务每次 reset 把旋钮打到开
        # （|q| ∈ [0.5, π/2] 或负对称区间），验证回放同样从开开始。
        for joint in joints:
            ctx.qpos_overrides[joint] = 0.6
    conds = [
        {
            "type": "rc_joint_state",
            "params": {"joint": j, "mode": "abs", "min": 0.35, "max": TWO_PI - 0.35},
        }
        for j in joints
    ]
    cond = _all(*conds)
    return cond if on else _not(cond)


def _b_pnp(ctx: Ctx, kind: str) -> dict[str, Any]:
    obj = "obj"
    if kind == "cab_to_counter":
        return _all(ctx.contact_any_counter(obj), ctx.gripper_far(obj))
    if kind == "counter_to_cab":
        fx = ctx.fx_nearest(ctx.cabinets(), ctx.robot_base_xy())
        return _all(ctx.inside(obj, fx), ctx.gripper_far(obj))
    if kind == "counter_to_microwave":
        fx = ctx.microwave()
        container = "container"
        return _all(
            {
                "type": "rc_contact",
                "params": {
                    "object": ctx.obj_body(obj),
                    "fixture": ctx.obj_body(container),
                },
            },
            ctx.contact(container, fx),
            ctx.gripper_far(obj),
        )
    if kind == "counter_to_sink":
        return _all(ctx.inside(obj, ctx.sink(), partial=True), ctx.gripper_far(obj))
    if kind == "counter_to_stove":
        return _all(ctx.in_receptacle(obj, "container", th=0.07), ctx.gripper_far(obj))
    if kind == "microwave_to_counter":
        return _all(ctx.in_receptacle(obj, "container"), ctx.gripper_far(obj))
    if kind == "sink_to_counter":
        return _all(
            ctx.in_receptacle(obj, "container"),
            ctx.contact_any_counter("container"),
            ctx.gripper_far(obj),
        )
    if kind == "stove_to_counter":
        return _all(ctx.in_receptacle(obj, "container", th=0.07), ctx.gripper_far(obj))
    raise Reason(f"未知 PnP 类型 {kind!r}")


def _b_coffee_pour(ctx: Ctx, to_machine: bool) -> dict[str, Any]:
    fx = ctx.register_coffee_machine()
    pour_site = ctx.site(fx, "receptacle_place_site")
    if to_machine:
        placed = {
            "type": "rc_near",
            "params": {
                "object": ctx.obj_body("obj"),
                "target": pour_site,
                "radius": 0.04,
            },
        }
    else:
        placed = ctx.contact_any_counter("obj")
    return _all(placed, ctx.gripper_far("obj"))


def _b_coffee_press(ctx: Ctx) -> dict[str, Any]:
    fx = ctx.register_coffee_machine()
    button = f"{fx}_start_button"
    ctx.geom_site(button)
    return ctx.appliance_spec(fx, True, button_far=button)


def _b_prepare_coffee(ctx: Ctx) -> dict[str, Any]:
    fx = ctx.register_coffee_machine()
    pour_site = ctx.site(fx, "receptacle_place_site")
    button = f"{fx}_start_button"
    ctx.geom_site(button)
    return _all(
        {
            "type": "rc_near",
            "params": {
                "object": ctx.obj_body("obj"),
                "target": pour_site,
                "radius": 0.04,
            },
        },
        ctx.gripper_far("obj"),
        {"type": "rc_appliance", "params": {"name": fx, "state": "on"}},
        {
            "type": "rc_gripper_far",
            "params": {
                "target": button,
                "sites": list(ctx.eef_sites()),
                "threshold": 0.15,
            },
        },
    )


def _b_microwave_thawing(ctx: Ctx) -> dict[str, Any]:
    fx = ctx.register_microwave()
    return _all(
        ctx.inside("obj", fx),
        ctx.gripper_far("obj"),
        {"type": "rc_appliance", "params": {"name": fx, "state": "on"}},
    )


def _b_pre_soak_pan(ctx: Ctx) -> dict[str, Any]:
    sink = ctx.sink()
    return _all(
        ctx.water_on(),
        ctx.inside("obj1", sink),
        ctx.inside("obj2", sink),
        ctx.gripper_far("obj1"),
        ctx.gripper_far("obj2"),
    )


def _b_restock_pantry(ctx: Ctx) -> dict[str, Any]:
    cab = ctx.fx_nearest(ctx.cabinets(), ctx.robot_base_xy())
    return _all(
        ctx.inside("obj1", cab),
        ctx.inside("obj2", cab),
        {
            "type": "rc_closer",
            "params": {
                "object": ctx.obj_body("obj1"),
                "anchor": ctx.obj_body("cab_obj1"),
                "other": ctx.obj_body("cab_obj2"),
            },
        },
        {
            "type": "rc_closer",
            "params": {
                "object": ctx.obj_body("obj2"),
                "anchor": ctx.obj_body("cab_obj1"),
                "other": ctx.obj_body("cab_obj1"),
            },
        },
        ctx.gripper_far("obj1"),
        ctx.gripper_far("obj2"),
    )


def _b_arrange_vegetables(ctx: Ctx) -> dict[str, Any]:
    return _all(
        ctx.in_receptacle("vegetable1", "cutting_board"),
        ctx.in_receptacle("vegetable2", "cutting_board"),
        ctx.gripper_far("cutting_board"),
    )


# ---- robocasa365 composite helpers -----------------------------------------


def _b365_two_recep(ctx: Ctx, obj1: str, obj2: str, recep: str) -> dict[str, Any]:
    return _all(
        ctx.in_receptacle(obj1, recep),
        ctx.in_receptacle(obj2, recep),
        ctx.gripper_far(recep),
    )


def _mk_timer(
    ctx: Ctx, name: str, count: int, conditions: list[dict[str, Any]]
) -> None:
    ctx.timers.append({"name": name, "count": count, "conditions": conditions})


# ---- builder table ------------------------------------------------------------

_BUILDERS = {
    "robocasa": {
        "OpenDrawer": lambda c: _b_drawer(c, True),
        "CloseDrawer": lambda c: _b_drawer(c, False),
        "OpenSingleDoor": lambda c: _b_door(c, True, False),
        "CloseSingleDoor": lambda c: _b_door(c, False, False),
        "OpenDoubleDoor": lambda c: _b_door(c, True, True),
        "CloseDoubleDoor": lambda c: _b_door(c, False, True),
        "TurnOnMicrowave": lambda c: _b_microwave_press(c, True),
        "TurnOffMicrowave": lambda c: _b_microwave_press(c, False),
        "TurnOnSinkFaucet": lambda c: _b_sink_faucet(c, True),
        "TurnOffSinkFaucet": lambda c: _b_sink_faucet(c, False),
        "TurnSinkSpout": _b_sink_spout,
        "TurnOnStove": lambda c: _b_stove(c, True),
        "TurnOffStove": lambda c: _b_stove(c, False),
        "PnPCabToCounter": lambda c: _b_pnp(c, "cab_to_counter"),
        "PnPCounterToCab": lambda c: _b_pnp(c, "counter_to_cab"),
        "PnPCounterToMicrowave": lambda c: _b_pnp(c, "counter_to_microwave"),
        "PnPCounterToSink": lambda c: _b_pnp(c, "counter_to_sink"),
        "PnPCounterToStove": lambda c: _b_pnp(c, "counter_to_stove"),
        "PnPMicrowaveToCounter": lambda c: _b_pnp(c, "microwave_to_counter"),
        "PnPSinkToCounter": lambda c: _b_pnp(c, "sink_to_counter"),
        "PnPStoveToCounter": lambda c: _b_pnp(c, "stove_to_counter"),
        "CoffeePressButton": _b_coffee_press,
        "CoffeeServeMug": lambda c: _b_coffee_pour(
            c, "machine" in str(c.env.get("behavior"))
        ),
        "CoffeeSetupMug": lambda c: _b_coffee_pour(
            c, "machine" in str(c.env.get("behavior"))
        ),
        "PrepareCoffee": _b_prepare_coffee,
        "MicrowaveThawing": _b_microwave_thawing,
        "PreSoakPan": _b_pre_soak_pan,
        "RestockPantry": _b_restock_pantry,
        "ArrangeVegetables": _b_arrange_vegetables,
    },
    "robocasa365": {
        "CloseDrawer": lambda c: _b_drawer(c, False),
        "OpenOven": lambda c: _b_door(c, True, False),
        "CloseOven": lambda c: _b_door(c, False, False),
        "OpenDishwasher": lambda c: _b_door(c, True, False),
        "CloseStandMixerHead": lambda c: _b365_mixer_close(c),
        "TurnOffMicrowave": lambda c: _b_microwave_press(c, False),
        "TurnOnMicrowave": lambda c: _b_microwave_press(c, True),
        "TurnOnSinkFaucet": lambda c: _b_sink_faucet(c, True),
        "TurnOnStove": lambda c: _b_stove(c, True),
        "TurnOnToasterOven": lambda c: _b365_toaster(c),
        "PackDessert": lambda c: _all(
            c.in_receptacle("cooked_food", "cooked_food_container"),
            c.in_receptacle("dessert", "cooked_food_container"),
            c.gripper_far("dessert"),
        ),
        "PickPlaceCabinetToCounter": lambda c: _all(
            c.contact_any_counter("obj"), c.gripper_far("obj")
        ),
        "PickPlaceSinkToCounter": lambda c: _all(
            c.in_receptacle("obj", "container"),
            c.contact_any_counter("container"),
            c.gripper_far("obj"),
        ),
        "AddLemonToFish": lambda c: _all(
            c.in_receptacle("lemon_wedge", "fish_container"),
            c.in_receptacle("fish", "fish_container"),
            c.contact_any_counter("fish_container"),
            c.gripper_far("lemon_wedge"),
        ),
        "AssembleCookingArray": lambda c: _all(
            c.in_receptacle("meat", "pan", th=0.07),
            c.gripper_far("vegetable"),
            c.gripper_far("condiment"),
            c.gripper_far("meat"),
            c.contact_any_counter("vegetable"),
            c.contact_any_counter("condiment"),
        ),
        "BeverageSorting": lambda c: _b365_beverage_sorting(c),
        "BreadAndCheese": lambda c: _all(
            c.in_receptacle("obj", "container"),
            c.in_receptacle("obj2", "container"),
            c.gripper_far("obj"),
        ),
        "CandleCleanup": lambda c: _b365_candle(c),
        "CerealAndBowl": lambda c: _b365_cereal(c),
        "ChooseMeasuringCup": lambda c: _b365_measuring_cup(c),
        "ClearFoodWaste": lambda c: _all(
            c.in_receptacle("food_0", "colander"),
            c.in_receptacle("food_1", "colander"),
            c.gripper_far("food_0"),
            c.gripper_far("food_1"),
        ),
        "ClearSink": lambda c: _all(
            _not(c.inside("food0", c.sink(), partial=True)),
            _not(c.inside("food1", c.sink(), partial=True)),
            c.inside("dish0", c.sink(), partial=True),
            c.inside("dish1", c.sink(), partial=True),
            c.gripper_far("food0"),
            c.gripper_far("food1"),
            c.gripper_far("dish0"),
            c.gripper_far("dish1"),
        ),
        "CookieDoughPrep": lambda c: _b365_cookie_dough(c),
        "DeliverBrewedCoffee": lambda c: _all(
            c.contact_any_counter("brewed_coffee"),
            c.contact_any_counter("breakfast_plate"),
            _near(c, "brewed_coffee", "breakfast_plate", 0.30),
            c.gripper_far("brewed_coffee"),
        ),
        "DessertUpgrade": lambda c: _b365_two_recep(
            c, "dessert1", "dessert2", "receptacle"
        ),
        "DisplayMeatVariety": lambda c: _all(
            c.in_receptacle("chicken", "serving_tray"),
            c.in_receptacle("shrimp", "serving_tray"),
            c.in_receptacle("beef", "serving_tray"),
            c.contact_any_counter("serving_tray"),
            c.gripper_far("chicken"),
            c.gripper_far("shrimp"),
            c.gripper_far("beef"),
        ),
        "DivideBasins": lambda c: _b365_divide_basins(c),
        "DrainVeggies": lambda c: _b365_drain_veggies(c),
        "DrinkwareConsolidation": lambda c: _b365_drinkware(c),
        "MakeIceLemonade": lambda c: _b365_ice_lemonade(c),
        "MakeLoadedPotato": lambda c: _all(
            c.gripper_far("potato"),
            c.gripper_far("condiment"),
            c.gripper_far("cheese"),
            c.in_receptacle("potato", "cutting_board"),
            c.in_receptacle("condiment", "cutting_board"),
            c.in_receptacle("cheese", "cutting_board"),
        ),
        "MealPrepStaging": lambda c: _b365_meal_prep(c),
        "MicrowaveDefrostMeat": lambda c: _b365_microwave_defrost(c),
        "OrganizeCoffeeCondiments": lambda c: _b365_coffee_condiments(c),
        "OvenBroilFish": lambda c: _b365_broil_fish(c),
        "PlateSteakMeal": lambda c: _all(
            c.in_receptacle("steak", "plate"),
            c.in_receptacle("asparagus", "plate"),
            c.in_receptacle("random_food", "tray"),
            c.gripper_far("steak"),
            c.gripper_far("asparagus"),
            c.gripper_far("random_food"),
        ),
        "PortionInTupperware": lambda c: _b365_portion(c),
        "PreRinseStation": lambda c: _b365_pre_rinse(c),
        "PrepForSanitizing": lambda c: _all(
            c.contact_any_counter("obj1"),
            c.contact_any_counter("obj2"),
            c.gripper_far("obj1"),
            c.gripper_far("obj2"),
        ),
        "PrepMarinatingMeat": lambda c: _all(
            c.gripper_far("condiment"),
            c.gripper_far("meat"),
            c.contact_any_counter("condiment"),
            c.in_receptacle("meat", "cutting_board"),
            c.contact_any_counter("cutting_board"),
        ),
        "ReheatMeal": lambda c: _b365_reheat(c),
        "RetrieveMeat": lambda c: _all(
            c.in_receptacle("meat", "receptacle"),
            c.gripper_far("meat"),
        ),
        "SanitizePrepCuttingBoard": lambda c: _all(
            _near(c, "spray", "cutting_board", 0.35),
            _near(c, "dish_brush", "cutting_board", 0.35),
            c.contact_any_counter("spray"),
            c.contact_any_counter("dish_brush"),
            c.contact_any_counter("cutting_board"),
            c.gripper_far("spray"),
        ),
        "ScrubBowl": lambda c: _b365_scrub_bowl(c),
        "SearingMeat": lambda c: _b365_searing(c),
        "SeasoningSpiceSetup": lambda c: _all(
            c.gripper_far("condiment1"),
            c.gripper_far("condiment2"),
            c.contact_any_counter("condiment1"),
            c.contact_any_counter("condiment2"),
        ),
        "SetUpCuttingStation": lambda c: _all(
            c.in_receptacle("meat", "receptacle"),
            c.in_receptacle("knife", "receptacle"),
            c.gripper_far("knife"),
            c.gripper_far("receptacle"),
        ),
        "StockingBreakfastFoods": lambda c: _b365_stocking(c),
        "SweetenCoffee": lambda c: _all(
            c.contact_any_counter("milk"),
            c.contact_any_counter("coffee"),
            _near(c, "milk", "coffee", 0.25),
            c.in_receptacle("sugar_cube", "coffee"),
            c.gripper_far("milk"),
            c.gripper_far("sugar_cube"),
        ),
        "ThawInSink": lambda c: _all(c.inside("obj", c.sink()), c.water_on()),
        "TongBuffetSetup": lambda c: _all(
            c.contact_any_counter("tongs"),
            c.contact_any_counter("food_container"),
            _near(c, "tongs", "food_container", 0.50),
            c.gripper_far("tongs", th=0.15),
            c.gripper_far("food_container", th=0.15),
        ),
    },
    "robocasa-gr1": {
        "PnPBottleToCabinetClose": lambda c: _gr1_inside_close(c, "cabinet"),
        "PnPWineToCabinetClose": lambda c: _gr1_inside_close(c, "cabinet"),
        "PnPCanToDrawerClose": lambda c: _gr1_inside_close(c, "drawer"),
        "PnPCupToDrawerClose": lambda c: _gr1_inside_close(c, "drawer"),
        "PnPMilkToMicrowaveClose": lambda c: _gr1_inside_close(c, "microwave"),
        "PnPPotatoToMicrowaveClose": lambda c: _gr1_inside_close(c, "microwave"),
        "PosttrainPnPNovelFromCuttingboardToBasketSplitA": lambda c: _gr1_splita(c),
        "PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA": lambda c: _gr1_splita(
            c
        ),
        "PosttrainPnPNovelFromCuttingboardToPanSplitA": lambda c: _gr1_splita(c),
        "PosttrainPnPNovelFromCuttingboardToPotSplitA": lambda c: _gr1_splita(c),
        "PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA": lambda c: _gr1_splita(
            c
        ),
        "PosttrainPnPNovelFromPlacematToBasketSplitA": lambda c: _gr1_splita(c),
        "PosttrainPnPNovelFromPlacematToBowlSplitA": lambda c: _gr1_splita(c),
        "PosttrainPnPNovelFromPlacematToPlateSplitA": lambda c: _gr1_splita(c),
        "PosttrainPnPNovelFromPlacematToTieredshelfSplitA": lambda c: _gr1_splita(c),
        "PosttrainPnPNovelFromPlateToBowlSplitA": lambda c: _gr1_splita(c),
        "PosttrainPnPNovelFromPlateToCardboardboxSplitA": lambda c: _gr1_splita(c),
        "PosttrainPnPNovelFromPlateToPanSplitA": lambda c: _gr1_splita(c),
        "PosttrainPnPNovelFromPlateToPlateSplitA": lambda c: _gr1_splita(c),
        "PosttrainPnPNovelFromTrayToCardboardboxSplitA": lambda c: _gr1_splita(c),
        "PosttrainPnPNovelFromTrayToPlateSplitA": lambda c: _gr1_splita(c),
        "PosttrainPnPNovelFromTrayToPotSplitA": lambda c: _gr1_splita(c),
        "PosttrainPnPNovelFromTrayToTieredbasketSplitA": lambda c: _gr1_splita(c),
        "PosttrainPnPNovelFromTrayToTieredshelfSplitA": lambda c: _gr1_splita(c),
    },
}


def _family(ctx: Ctx) -> str:
    return str(ctx.meta.get("env_name") or "")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def resolve_suite(suite: str, inventory: dict[str, Any]) -> dict[str, Any]:
    """Resolve one suite to the compile knowledge base."""
    builders = _BUILDERS[suite]
    tasks: dict[str, Any] = {}
    pending: list[str] = []
    pending_reasons: dict[str, str] = {}
    for entry in inventory["tasks"]:
        name = entry["name"]
        try:
            meta_path = RC_WORLDS / suite / "worlds" / f"{name}.meta.json"
            if not meta_path.is_file():
                raise Reason("世界包缺失（先跑 tools/robocasa_worlds.py --all）")
            ctx = Ctx(suite, name)
            family = _family(ctx)
            builder = builders.get(family)
            if builder is None:
                raise Reason(f"家族 {family!r} 无 builder（本批未覆盖）")
            success = builder(ctx)
            world_rel = f"assets/robocasa/{suite}/worlds/{name}.xml"
            tasks[name] = {
                "family": entry["family"],
                "goal": ctx.meta.get("language") or "",
                "robot": ROBOT_BY_SUITE[suite],
                "scene": "efe_world",
                "task_type": "efe_robocasa",
                "world": {
                    "mjcf_path": world_rel,
                    "qpos_path": f"assets/robocasa/{suite}/worlds/{name}.qpos.npy",
                    "qvel_path": f"assets/robocasa/{suite}/worlds/{name}.qvel.npy",
                    "meta_path": f"assets/robocasa/{suite}/worlds/{name}.meta.json",
                    "extra_sites": ctx.extra_sites,
                    "qpos_overrides": ctx.qpos_overrides,
                },
                "appliances": ctx.appliances,
                "timers": ctx.timers,
                "success": success,
                "max_episode_steps": MAX_STEPS,
                "efe": {
                    "fidelity": "spec",
                    "notes": (
                        f"判据逐字转写上游 {family}._check_success；"
                        f"instance_sha={str(ctx.meta.get('instance_sha', '?'))[:12]}"
                    ),
                },
                "assets_task_relevant": [
                    {
                        "name": "world",
                        "shape_type": "mjcf",
                        "mjcf_path": world_rel,
                        "position": [0.0, 0.0, 0.0],
                        "orientation": [1.0, 0.0, 0.0, 0.0],
                        "static": True,
                    }
                ],
            }
        except Reason as exc:
            pending.append(name)
            pending_reasons[name] = str(exc)
        except Exception as exc:  # noqa: BLE001 - one bad task must not kill the batch
            pending.append(name)
            pending_reasons[name] = f"{type(exc).__name__}: {exc}"
    return {
        "suite": suite,
        "robot": ROBOT_BY_SUITE[suite],
        "scene": "efe_world",
        "source": {"suite": suite},
        "pending_tasks": sorted(pending),
        "pending_reasons": pending_reasons,
        "tasks": tasks,
    }


# ---------------------------------------------------------------------------
# robocasa365 composite builders
# ---------------------------------------------------------------------------


def _near(
    ctx: Ctx, obj: str, target: str, radius: float, axes: int = 2
) -> dict[str, Any]:
    return {
        "type": "rc_near",
        "params": {
            "object": ctx.obj_body(obj),
            "target": ctx.obj_body(target) if target in ctx.object_names else target,
            "radius": radius,
            "axes": axes,
        },
    }


def _b365_mixer_close(ctx: Ctx) -> dict[str, Any]:
    cands = ctx.fx_named("stand_mixer")
    if not cands:
        raise Reason("无 stand mixer 夹具")
    fx = cands[0]
    head = [
        j for j in ctx.joints if j.startswith(f"{fx}_") and j.endswith("head_joint")
    ]
    if not head:
        raise Reason("stand mixer 无头关节")
    lo, hi = ctx.joint_range(head[0])
    # 上游 _setup_scene 在 reset 时把机头打到全开（set_head_pos(1.0)），
    # 验证回放同样从全开开始；冻结 qpos 需按此覆盖后才算有效初始态。
    ctx.qpos_overrides[head[0]] = hi
    return ctx.normalized_joint(head[0], None, 0.01)


def _b365_toaster(ctx: Ctx) -> dict[str, Any]:
    raise Reason("TurnOnToasterOven：time 状态机（realtime 动态任务），本批挂起")


def _b365_beverage_sorting(ctx: Ctx) -> dict[str, Any]:
    cabs = sorted(
        ctx.cabinets_with_interior(),
        key=lambda f: np.linalg.norm(ctx.fx_world(f)[:2] - ctx.robot_base_xy()),
    )[:2]
    if len(cabs) < 2:
        raise Reason("BeverageSorting 需要两个柜")
    cab1, cab2 = cabs

    def placed(c_alcohol, c_non):
        return _all(
            ctx.inside("alcohol1", c_alcohol),
            ctx.inside("alcohol2", c_alcohol),
            ctx.inside("non_alcohol1", c_non),
            ctx.inside("non_alcohol2", c_non),
        )

    return _all(
        ctx.gripper_far("alcohol1"),
        ctx.gripper_far("alcohol2"),
        ctx.gripper_far("non_alcohol1"),
        ctx.gripper_far("non_alcohol2"),
        _any(placed(cab1, cab2), placed(cab2, cab1)),
    )


def _b365_cereal(ctx: Ctx) -> dict[str, Any]:
    cab = ctx.fx_nearest(ctx.cabinets(), ctx.robot_base_xy())
    return _all(
        ctx.contact_any_counter("cereal"),
        ctx.contact_any_counter("bowl"),
        ctx.normalized_joint(ctx.hinge_joints(cab)[0], None, 0.05),
    )


def _b365_measuring_cup(ctx: Ctx) -> dict[str, Any]:
    use_tablespoon = bool(ctx.env.get("use_tablespoon"))
    large = "measuring_cup_large"
    small = "measuring_cup_small"
    chosen, other = (large, small) if use_tablespoon else (small, large)
    return _all(
        ctx.contact_any_counter(chosen),
        _not(ctx.contact_any_counter(other)),
        ctx.gripper_far(chosen),
    )


def _b365_cookie_dough(ctx: Ctx) -> dict[str, Any]:
    bowl_radius = ctx.obj_radius("bowl")
    return _all(
        ctx.contact_any_counter("egg"),
        ctx.contact_any_counter("butter"),
        _near(ctx, "egg", "bowl", bowl_radius + 0.25),
        _near(ctx, "butter", "bowl", bowl_radius + 0.25),
        ctx.gripper_far("egg", th=0.15),
        ctx.gripper_far("butter", th=0.15),
    )


def _basin(ctx: Ctx, obj: str, side: str) -> dict[str, Any]:
    sink = ctx.sink()
    body = ctx.fx_body(sink)
    region = f"{sink}_reg_basin_{side}"
    xml_block = ctx.world_xml
    pattern = re.compile(
        rf'<geom name="{re.escape(region)}"[^>]*pos="([^"]+)"[^>]*size="([^"]+)"'
    )
    hit = pattern.search(xml_block)
    if hit is None:
        # attributes may come in any order
        elem = re.search(rf'<geom name="{re.escape(region)}"[^>]*/?>', xml_block)
        if elem is None:
            raise Reason(f"水槽缺 basin 区域 geom {region!r}")
        tag = elem.group(0)
        pos = re.search(r'pos="([^"]+)"', tag)
        size = re.search(r'size="([^"]+)"', tag)
        if pos is None or size is None:
            raise Reason(f"basin 区域 geom {region!r} 缺 pos/size")
        center, half = pos.group(1), size.group(1)
    else:
        center, half = hit.group(1), hit.group(2)
    return {
        "type": "rc_basin",
        "params": {
            "object": ctx.obj_body(obj),
            "fixture_body": body,
            "side": side,
            "center_local": [float(v) for v in center.split()],
            "halfsize_local": [float(v) for v in half.split()],
            "object_bbox": ctx.obj_bbox(obj),
        },
    }


def _b365_divide_basins(ctx: Ctx) -> dict[str, Any]:
    return _all(
        _basin(ctx, "obj_to_wash", "left"),
        _basin(ctx, "obj_drying", "right"),
        ctx.gripper_far("obj_to_wash"),
        ctx.gripper_far("obj_drying"),
    )


def _b365_drain_veggies(ctx: Ctx) -> dict[str, Any]:
    sink = ctx.sink()
    _mk_timer(
        ctx,
        "drain_veggies_washed",
        11,
        [ctx.inside("obj", sink), ctx.water_on()],
    )
    return _all(
        {"type": "rc_timer", "params": {"name": "drain_veggies_washed", "count": 11}},
        ctx.in_receptacle("obj", "obj_container"),
        _not(ctx.water_on()),
    )


def _b365_drinkware(ctx: Ctx) -> dict[str, Any]:
    cab = ctx.fx_nearest(ctx.cabinets_with_interior(), ctx.robot_base_xy())
    names = sorted(n for n in ctx.object_names if re.fullmatch(r"obj_\d+", n))
    if not names:
        raise Reason("无 obj_N 物体")
    return _all(
        *[ctx.inside(n, cab) for n in names], *[ctx.gripper_far(n) for n in names]
    )


def _b365_ice_lemonade(ctx: Ctx) -> dict[str, Any]:
    return _all(
        ctx.in_receptacle("lemon_wedge", "glass_cup", th=0.5),
        _any(
            ctx.in_receptacle("ice_cube1", "glass_cup", th=0.5),
            ctx.in_receptacle("ice_cube2", "glass_cup", th=0.5),
        ),
        ctx.gripper_far("lemon_wedge", th=0.15),
        ctx.gripper_far("ice_cube1", th=0.15),
        ctx.gripper_far("ice_cube2", th=0.15),
    )


def _burner_sites(ctx: Ctx, stove: str) -> list[str]:
    sites = [
        s
        for s in ctx.sites
        if s.startswith(f"{stove}_") and re.search(r"burner", s) and "place_site" in s
    ]
    if not sites:
        raise Reason(f"灶台 {stove!r} 无 burner 站点")
    return sites


def _on_burner(ctx: Ctx, pan: str, burner: str, knobs: list[str]) -> dict[str, Any]:
    return {
        "type": "rc_on_burner",
        "params": {
            "object": ctx.obj_body(pan),
            "burner_site": burner,
            "threshold": 0.15,
            "knob_joint": knobs[0] if knobs else "",
            "require_knob_on": bool(knobs),
        },
    }


def _b365_meal_prep(ctx: Ctx) -> dict[str, Any]:
    stove = ctx.stove()
    burners = _burner_sites(ctx, stove)
    knob_joints = [n for n in ctx.joints if n.startswith(f"{stove}_") and "knob" in n]

    def food_on(pan_a: str, pan_b: str) -> dict[str, Any]:
        return _all(
            ctx.in_receptacle("vegetable", pan_a),
            ctx.in_receptacle("meat", pan_b),
        )

    pan_placements = [
        _all(
            _on_burner(ctx, "pan1", b1, knob_joints),
            _on_burner(ctx, "pan2", b2, knob_joints),
        )
        for b1 in burners
        for b2 in burners
        if b1 != b2
    ]
    return _all(
        _any(food_on("pan1", "pan2"), food_on("pan2", "pan1")),
        _any(*pan_placements),
    )


def _b365_microwave_defrost(ctx: Ctx) -> dict[str, Any]:
    fx = ctx.register_microwave()
    return _all(
        ctx.inside("tupperware", fx),
        ctx.in_receptacle("meat", "tupperware"),
        ctx.normalized_joint(f"{fx}_microjoint", None, 0.05),
        {"type": "rc_appliance", "params": {"name": fx, "state": "on"}},
        ctx.gripper_far("tupperware"),
    )


def _b365_coffee_condiments(ctx: Ctx) -> dict[str, Any]:
    mug_near = lambda obj: {  # noqa: E731 - 3D near the mug
        "type": "rc_near",
        "params": {
            "object": ctx.obj_body(obj),
            "target": ctx.obj_body("mug"),
            "radius": 0.30,
            "axes": 3,
        },
    }
    return _all(
        mug_near("obj1"),
        mug_near("obj2"),
        ctx.contact_any_counter("obj1"),
        ctx.contact_any_counter("obj2"),
        ctx.gripper_far("obj1", th=0.15),
        ctx.gripper_far("obj2", th=0.15),
    )


def _b365_broil_fish(ctx: Ctx) -> dict[str, Any]:
    oven = ctx.fx_nearest([f for f in ctx.fixtures if "oven" in f], ctx.robot_base_xy())
    _mk_timer(
        ctx,
        "oven_broil_fish",
        10,
        [
            ctx.contact("obj", oven),
            ctx.normalized_joint(ctx.hinge_joints(oven)[0], None, 0.05),
        ],
    )
    return {"type": "rc_timer", "params": {"name": "oven_broil_fish", "count": 10}}


def _b365_portion(ctx: Ctx) -> dict[str, Any]:
    return _all(
        _any(
            _all(
                ctx.in_receptacle("cooked_food1", "tupperware1"),
                ctx.in_receptacle("vegetable1", "tupperware1"),
            ),
            _all(
                ctx.in_receptacle("cooked_food2", "tupperware1"),
                ctx.in_receptacle("vegetable2", "tupperware1"),
            ),
        ),
        _any(
            _all(
                ctx.in_receptacle("cooked_food1", "tupperware2"),
                ctx.in_receptacle("vegetable1", "tupperware2"),
            ),
            _all(
                ctx.in_receptacle("cooked_food2", "tupperware2"),
                ctx.in_receptacle("vegetable2", "tupperware2"),
            ),
        ),
        ctx.gripper_far("cooked_food1"),
        ctx.gripper_far("cooked_food2"),
        ctx.gripper_far("vegetable1"),
        ctx.gripper_far("vegetable2"),
    )


def _b365_pre_rinse(ctx: Ctx) -> dict[str, Any]:
    left_obj = str(ctx.env.get("left_obj") or "left_obj")
    right_obj = str(ctx.env.get("right_obj") or "right_obj")
    return _all(
        _basin(ctx, left_obj, "left"),
        _basin(ctx, right_obj, "right"),
        ctx.gripper_far(left_obj),
        ctx.gripper_far(right_obj),
    )


def _b365_reheat(ctx: Ctx) -> dict[str, Any]:
    fx = ctx.register_microwave()
    return _all(
        ctx.inside("tupperware", fx),
        ctx.in_receptacle("meat", "tupperware"),
        ctx.in_receptacle("vegetable", "tupperware"),
        ctx.normalized_joint(f"{fx}_microjoint", None, 0.05),
        {"type": "rc_appliance", "params": {"name": fx, "state": "on"}},
        ctx.gripper_far("tupperware"),
    )


def _b365_scrub_bowl(ctx: Ctx) -> dict[str, Any]:
    _mk_timer(
        ctx,
        "scrub_bowl_contact",
        15,
        [
            {
                "type": "rc_contact",
                "params": {
                    "object": ctx.obj_body("sponge"),
                    "fixture": ctx.obj_body("bowl"),
                },
            }
        ],
    )
    return _all(
        {"type": "rc_timer", "params": {"name": "scrub_bowl_contact", "count": 15}},
        ctx.gripper_far("bowl"),
    )


def _b365_searing(ctx: Ctx) -> dict[str, Any]:
    stove = ctx.stove()
    knob = str(ctx.env.get("knob") or "")
    knob_joints = [
        n
        for n in ctx.joints
        if n.startswith(f"{stove}_") and "knob" in n and (not knob or knob in n)
    ]
    burners = _burner_sites(ctx, stove)
    return _all(
        ctx.gripper_far("meat"),
        {
            "type": "rc_on_burner",
            "params": {
                "object": ctx.obj_body("pan"),
                "burner_site": burners[0],
                "threshold": 0.15,
                "knob_joint": knob_joints[0] if knob_joints else "",
                "require_knob_on": bool(knob_joints),
            },
        },
        ctx.in_receptacle("meat", "pan", th=0.07),
    )


def _b365_stocking(ctx: Ctx) -> dict[str, Any]:
    cabs = sorted(
        ctx.cabinets_with_interior(),
        key=lambda f: np.linalg.norm(ctx.fx_world(f)[:2] - ctx.robot_base_xy()),
    )[:2]
    if len(cabs) < 2:
        raise Reason("StockingBreakfastFoods 需要两个柜")
    return _all(
        ctx.inside("obj1", cabs[0]),
        ctx.inside("obj2", cabs[1]),
        ctx.gripper_far("obj1"),
        ctx.gripper_far("obj2"),
    )


# ---------------------------------------------------------------------------
# robocasa-gr1 builders
# ---------------------------------------------------------------------------


def _gr1_inside_close(ctx: Ctx, fixture_kind: str) -> dict[str, Any]:
    behavior = str(ctx.env.get("behavior") or "close")
    open_target = behavior == "open"
    if fixture_kind == "cabinet":
        cands = ctx.cabinets()
        fx = ctx.fx_nearest(cands, ctx.robot_base_xy())
        joint = ctx.hinge_joints(fx)[0]
    elif fixture_kind == "drawer":
        cands = ctx.fx_with_joints("slidejoint")
        fx = ctx.fx_nearest(cands, ctx.robot_base_xy())
        joint = ctx.slide_joint(fx)
    elif fixture_kind == "microwave":
        fx = ctx.microwave()
        joint = f"{fx}_microjoint"
    else:
        raise Reason(f"未知 gr1 夹具类型 {fixture_kind!r}")
    lo, hi = ctx.joint_range(joint)
    sign = 1.0
    if lo < 0.0 <= hi:
        sign, lo, hi = -1.0, 0.0, -lo
    door = {
        "type": "rc_joint_state",
        "params": {
            "joint": joint,
            "mode": "normalized",
            "range": [round(lo, 6), round(hi, 6)],
            "sign": sign,
            **({"min": 0.5} if open_target else {"max": 0.005}),
        },
    }
    return _all(
        ctx.inside("obj", fx, partial=True),
        door,
    )


def _gr1_splita(ctx: Ctx) -> dict[str, Any]:
    """PosttrainPnPNovelFromXToYSplitA generated families."""
    target = str(ctx.env.get("target_container") or "")
    conds = [
        ctx.gripper_far("container"),
        ctx.gripper_far("obj"),
        ctx.in_receptacle("obj", "container"),
        _not(ctx.contact_any_counter("obj")),
        {
            "type": "rc_upright",
            "params": {"object": ctx.obj_body("container"), "threshold": 0.8},
        },
    ]
    if target in ("tiered_basket", "tiered_shelf"):
        sites = [
            (s, bool(re.search(r"spawn", s)))
            for s in ctx.sites
            if s.startswith(f"{ctx.obj_body('container').removesuffix('_main')}_")
            and "spawn" in s
        ]
        if not sites:
            raise Reason(f"容器无 spawn 站点（target={target!r}）")
        conds.append(
            {
                "type": "rc_at_spawn",
                "params": {
                    "object": ctx.obj_body("obj"),
                    "spawn_sites": [{"site": s, "target": t} for s, t in sites],
                },
            }
        )
    return _all(*conds)


def _b365_candle(ctx: Ctx) -> dict[str, Any]:
    cab = ctx.fx_nearest(ctx.cabinets_with_interior(), ctx.robot_base_xy())
    return _all(
        ctx.inside("obj1", cab),
        ctx.inside("obj2", cab),
        ctx.normalized_joint(ctx.hinge_joints(cab)[0], None, 0.05),
    )
