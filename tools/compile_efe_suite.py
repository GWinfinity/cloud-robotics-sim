#!/usr/bin/env python3
"""Compile an EFE suite's tasks into schema v1 task configs.

An EFE suite port is data, not code, wherever possible: this tool renders one
``configs/tasks/efe/<suite>/<task>.yaml`` per EFE task (plus a
``data/suites/efe_<suite>.yaml`` aggregate) from three committed inputs:

* the task inventory ``data/efe/tasks/<suite>.json`` (names, modes, budgets —
  generated from the upstream snapshot by ``tools/fetch_efe_inventory.py``);
* the suite knowledge base ``data/efe/compile/<suite>.yaml`` — one resolved
  record per task: robot, scene, success spec, assets, init ranges, EFE
  metadata. For MJCF-backed suites the KB is produced by ``--resolve`` from
  the curated predicate table (``data/efe/compile/<suite>.predicates.yaml``)
  plus the upstream model files under ``assets/``;
* the schema v1 contract of ``core/task_loader.py``.

``--check`` re-renders and diffs, making compiled configs CI-gated artifacts
(same pattern as the registry/embodiment manifest tools).

Usage::

    uv run python tools/compile_efe_suite.py --suite metaworldplus --resolve
    uv run python tools/compile_efe_suite.py --suite metaworldplus
    uv run python tools/compile_efe_suite.py --suite metaworldplus --check
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_EFE = REPO_ROOT / "data" / "efe"
COMPILE_DIR = DATA_EFE / "compile"
CONFIGS_OUT = REPO_ROOT / "configs" / "tasks" / "efe"
SUITES_OUT = REPO_ROOT / "data" / "suites"

EFAIL = "EFE 迁移编译物（{suite}）；由 tools/compile_efe_suite.py 生成，--check 门禁。"


# ---------------------------------------------------------------------------
# MetaWorld+ resolver: curated predicate table + env MJCF -> resolved KB
# ---------------------------------------------------------------------------

MW_ENV_XML = {
    # env name -> (xml filename under assets/metaworld/envs/assets/sawyer_xyz,
    #              goal body name in the xml)
}


def _mw_xml_dir() -> Path:
    return REPO_ROOT / "assets" / "metaworld" / "envs" / "assets"


def _load_mjcf_inlined(xml_path: Path, _depth: int = 0) -> ET.Element:
    """Parse an MJCF file with ``<include>`` files spliced in (cycle-guarded)."""
    if _depth > 8:
        raise ValueError(f"MJCF include depth exceeded at {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    _splice_includes(root, xml_path.parent, _depth)
    return root


def _splice_includes(element: ET.Element, base_dir: Path, depth: int) -> None:
    """Replace every ``<include>`` child with the included root's children.

    Spliced subtrees are tagged with ``_include_src`` so the resolver can
    tell task bodies from the robot model (``xyz_base.xml``) bodies.
    """
    for child in list(element):
        if child.tag == "include":
            rel = child.get("file")
            included_path = (base_dir / rel).resolve() if rel else None
            if included_path is None or not included_path.is_file():
                element.remove(child)
                continue
            included = _load_mjcf_inlined(included_path, depth + 1)
            idx = list(element).index(child)
            element.remove(child)
            for offset, grandchild in enumerate(list(included)):
                grandchild.set("_include_src", rel)
                element.insert(idx + offset, grandchild)
        else:
            _splice_includes(child, base_dir, depth)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by wxyz quaternion ``q``."""
    w, x, y, z = q
    u = np.array([x, y, z])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def resolve_mjcf(xml_path: Path) -> dict[str, Any]:
    """Resolve an env MJCF into world-frame bodies and sites.

    Returns ``{"bodies": {name: {...}}, "sites": {name: [x, y, z]}}`` where
    each body records its world position, geom description and whether it is
    movable (free joint or a mechanism slider like MetaWorld's buttons).
    Fixture composition (nested bodies, quaternions, includes) is honoured.
    """
    root = _load_mjcf_inlined(xml_path)
    bodies: dict[str, dict[str, Any]] = {}
    sites: dict[str, list[float]] = {}

    def walk(
        element: ET.Element,
        base_pos: np.ndarray,
        base_quat: np.ndarray,
        include_src: str | None = None,
    ) -> None:
        for child in list(element):
            if child.tag == "site" and include_src is None:
                # worldbody-level sites (MetaWorld goal markers) carry the
                # current base transform directly.
                spos = np.array(
                    [float(v) for v in (child.get("pos") or "0 0 0").split()]
                )
                sites[child.get("name")] = [
                    round(float(v), 4) for v in base_pos + _quat_rotate(base_quat, spos)
                ]
            if child.tag == "body":
                src = child.get("_include_src", include_src)
                local = np.array(
                    [float(v) for v in (child.get("pos") or "0 0 0").split()]
                )
                quat = np.array(
                    [float(v) for v in (child.get("quat") or "1 0 0 0").split()]
                )
                world = base_pos + _quat_rotate(base_quat, local)
                world_quat = _quat_mul(base_quat, quat)
                name = child.get("name") or f"body_{len(bodies)}"
                jointed = (
                    child.find("freejoint") is not None
                    or child.find("joint") is not None
                )
                is_robot = src is not None and "xyz_base" in src
                geom = child.find("geom")
                bodies[name] = {
                    "world_pos": [round(float(v), 4) for v in world],
                    "geom_type": (
                        geom.get("type", "sphere") if geom is not None else "sphere"
                    ),
                    "size": (
                        [float(v) for v in (geom.get("size") or "0.02").split()]
                        if geom is not None
                        else [0.02]
                    ),
                    "rgba": (
                        [float(v) for v in (geom.get("rgba") or "1 0 0 1").split()]
                        if geom is not None
                        else [1.0, 0.0, 0.0, 1.0]
                    ),
                    "movable": jointed and not is_robot,
                }
                for site in child.findall("site"):
                    spos = np.array(
                        [float(v) for v in (site.get("pos") or "0 0 0").split()]
                    )
                    sites[site.get("name")] = [
                        round(float(v), 4)
                        for v in world + _quat_rotate(world_quat, spos)
                    ]
                walk(child, world, world_quat, src)

    for worldbody in root.iter("worldbody"):
        walk(worldbody, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    return {"bodies": bodies, "sites": sites}


def parse_mjcf_objects(xml_path: Path) -> dict[str, dict[str, Any]]:
    """Movable bodies of an env MJCF (free or slide-jointed), world frame."""
    resolved = resolve_mjcf(xml_path)
    return {
        name: {**body, "pos": body["world_pos"]}
        for name, body in resolved["bodies"].items()
        if body["movable"]
    }


def parse_mjcf_sites(xml_path: Path) -> dict[str, list[float]]:
    """Named sites of an env MJCF in world coordinates."""
    return resolve_mjcf(xml_path)["sites"]


def resolve_metaworldplus(predicates: dict[str, Any]) -> dict[str, Any]:
    """Build the resolved knowledge base for MetaWorld+ from env MJCFs.

    Entries marked ``pending: true`` (predicate sentence not yet mirrored
    from upstream) are recorded in ``pending_tasks`` and skipped; the
    inventory gate accounts for them explicitly.
    """
    xml_dir = _mw_xml_dir() / "sawyer_xyz"
    table = predicates["envs"]
    tasks: dict[str, Any] = {}
    pending: list[str] = []
    for env_name, spec in table.items():
        if spec.get("pending"):
            pending.append(f"metaworldplus-{env_name}-i00")
            continue
        xml_file = spec.get("xml", f"sawyer_{env_name}.xml")
        xml_path = xml_dir / xml_file
        if not xml_path.is_file():
            logger.warning("metaworld+ env XML missing: %s", xml_path)
            pending.append(f"metaworldplus-{env_name}-i00")
            continue
        objects = parse_mjcf_objects(xml_path)
        sites = parse_mjcf_sites(xml_path)
        resolved = resolve_mjcf(xml_path)
        goal_body = spec.get("object")
        obj = objects.get(goal_body) if goal_body else None
        if goal_body is not None and obj is None:
            # Mechanism envs teleport bodies that carry no joint in the XML
            # (door/window/lever outer bodies): rebuild them as dynamic
            # entities tracking the same pose semantics.
            candidate = resolved["bodies"].get(goal_body)
            if candidate is not None:
                obj = {**candidate, "pos": candidate["world_pos"]}
            else:
                logger.warning(
                    "metaworld+ %s: object %r not found in %s",
                    env_name,
                    goal_body,
                    xml_file,
                )
        target = None
        if spec.get("goal_site"):
            target = sites.get(spec["goal_site"])
        elif spec.get("goal_body"):
            body_info = resolved["bodies"].get(spec["goal_body"])
            if body_info is not None:
                target = body_info["world_pos"]

        success = json.loads(json.dumps(spec["success"]))  # deep copy
        params = success.get("params", {})
        if spec.get("point") and params.get("offset") is None:
            point_world = sites.get(spec["point"])
            body_world = (obj or {}).get("pos")
            if point_world is not None and body_world is not None:
                params["offset"] = [
                    round(p - b, 4) for p, b in zip(point_world, body_world)
                ]
        if params.get("target_position") == "from_site":
            if target is None:
                raise SystemExit(
                    f"metaworldplus/{env_name}: success needs the goal site, "
                    f"but {xml_file} has none"
                )
            resolved = [round(float(v), 4) for v in target]
            for axis, value in (spec.get("target_override") or {}).items():
                resolved["xyz".index(axis)] = float(value)
            params["target_position"] = resolved
        elif "region" in params and params["region"] is None:
            if target is None:
                raise SystemExit(
                    f"metaworldplus/{env_name}: region needs the goal site"
                )
            cx, cy, cz = (float(v) for v in target)
            half = spec.get("goal_half_size", 0.05)
            params["region"] = [
                round(cx - half, 4),
                round(cx + half, 4),
                round(cy - half, 4),
                round(cy + half, 4),
                round(cz - half, 4),
                round(cz + half, 4),
            ]
        assets = []
        if obj is not None:
            shape_map = {"box": "box", "cylinder": "cylinder", "sphere": "sphere"}
            position = spec.get("init_override", obj["pos"])
            asset = {
                "name": goal_body,
                "shape_type": shape_map.get(obj["geom_type"], "sphere"),
                "size": obj["size"][:3],
                "position": position,
                "color": obj["rgba"],
                "mass": 0.75,
                "static": False,
                "friction": 1.0,
            }
            yaw_deg = spec.get("init_yaw_deg")
            if yaw_deg is not None:
                import math

                half = math.radians(float(yaw_deg)) / 2.0
                asset["orientation"] = [
                    round(math.cos(half), 6),
                    0.0,
                    0.0,
                    round(math.sin(half), 6),
                ]
            assets.append(asset)
        for extra in spec.get("extra_entities") or []:
            assets.append(
                {
                    "name": extra["name"],
                    "shape_type": extra.get("shape_type", "box"),
                    "size": extra.get("size", [0.02, 0.02, 0.02]),
                    "position": extra["position"],
                    "color": extra.get("color", [0.7, 0.7, 0.7, 1.0]),
                    "mass": extra.get("mass", 0.2),
                    "static": False,
                    "friction": 1.0,
                }
            )
        task_key = f"metaworldplus-{env_name}-i00"
        tasks[task_key] = {
            "family": f"metaworld/{env_name}",
            "goal": spec["goal"],
            "robot": "sawyer",
            "scene": "efe_tabletop",
            "task_type": "efe_static",
            "assets_task_relevant": assets,
            "success": success,
            "max_episode_steps": spec.get("max_episode_steps", 500),
            "efe": {"fidelity": "spec", "notes": spec.get("notes", "")},
        }
    return {
        "suite": "metaworldplus",
        "robot": "sawyer",
        "scene": "efe_tabletop",
        "source": predicates.get("source", {}),
        "pending_tasks": sorted(pending),
        "tasks": tasks,
    }


# ---------------------------------------------------------------------------
# VLABench resolver: frozen instance.json + component MJCF -> resolved KB
# ---------------------------------------------------------------------------

VLB_MIRROR = REPO_ROOT / "assets" / "vlabench"
VLB_INSTANCES = DATA_EFE / "upstream" / "tasks" / "vlabench"


def _vlb_worldbody_sites(
    xml_path: Path, names: tuple[str, ...]
) -> dict[str, list[float]]:
    """Local offsets of named sites that are direct ``worldbody`` children."""
    root = _load_mjcf_inlined(xml_path)
    found: dict[str, list[float]] = {}
    for worldbody in root.iter("worldbody"):
        for site in worldbody.findall("site"):
            name = site.get("name")
            if name in names:
                found[name] = [
                    round(float(v), 6) for v in (site.get("pos") or "0 0 0").split()
                ]
    return found


def _vlb_group_sites(xml_path: Path, group: str) -> dict[str, list[float]]:
    """Model-frame positions of sites with ``group`` set (transform-aware).

    Container keypoint sites (group="3") define the containment bbox upstream
    (``Entity.key_sites``); sites under positioned sub-bodies are transformed
    by the body chain. ``group`` may come from a site ``class=`` default
    (``dclass="keypoint"`` → ``<default class="keypoint"><site group="3">``),
    so defaults are resolved before filtering.
    """
    root = _load_mjcf_inlined(xml_path)
    defaults: dict[str, str] = {}
    for default_class in root.iter("default"):
        cls = default_class.get("class")
        if cls is None:
            continue
        site_default = default_class.find("site")
        if site_default is not None and site_default.get("group"):
            defaults[cls] = site_default.get("group")

    def site_group(site: ET.Element) -> str | None:
        if site.get("group"):
            return site.get("group")
        cls = site.get("class") or site.get("dclass")
        return defaults.get(cls) if cls else None

    found: dict[str, list[float]] = {}

    def walk(element: ET.Element, base_pos: np.ndarray, base_quat: np.ndarray) -> None:
        for child in list(element):
            if child.tag == "site" and site_group(child) == group:
                spos = np.array(
                    [float(v) for v in (child.get("pos") or "0 0 0").split()]
                )
                found[child.get("name") or f"site_{len(found)}"] = [
                    round(float(v), 6) for v in base_pos + _quat_rotate(base_quat, spos)
                ]
            if child.tag == "body":
                local = np.array(
                    [float(v) for v in (child.get("pos") or "0 0 0").split()]
                )
                quat = np.array(
                    [float(v) for v in (child.get("quat") or "1 0 0 0").split()]
                )
                world = base_pos + _quat_rotate(base_quat, local)
                world_quat = _quat_mul(base_quat, quat)
                walk(child, world, world_quat)

    for worldbody in root.iter("worldbody"):
        walk(worldbody, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    return found


def _euler_xyz_to_quat(euler: list[float]) -> list[float]:
    """Extrinsic XYZ euler radians (MuJoCo ``eulerseq`` default) to wxyz quat."""
    axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    def axis_quat(axis: tuple[float, float, float], angle: float) -> np.ndarray:
        half = angle / 2.0
        return np.array([np.cos(half), *(np.sin(half) * np.asarray(axis))], dtype=float)

    qx, qy, qz = (axis_quat(a, e) for a, e in zip(axes, euler))
    q = _quat_mul(qz, _quat_mul(qy, qx))
    return [round(float(v), 6) for v in q]


def _vlb_orientation(value: Any, task_label: str, entity: str) -> list[float]:
    """Normalise an instance orientation to wxyz (4-vec pass-through)."""
    vals = [float(v) for v in (value if value is not None else [1.0, 0.0, 0.0, 0.0])]
    if len(vals) == 4:
        return vals
    if len(vals) == 3:
        return _euler_xyz_to_quat(vals)
    raise SystemExit(
        f"{task_label}: orientation of {entity!r} must have 3 (euler xyz) or "
        f"4 (wxyz quat) numbers, got {vals}"
    )


def _vlb_body_positions(xml_path: Path) -> dict[str, list[float]]:
    """Model-frame positions of all named bodies (pos chain, quat-aware)."""
    root = _load_mjcf_inlined(xml_path)
    positions: dict[str, list[float]] = {}

    def walk(element: ET.Element, base_pos: np.ndarray, base_quat: np.ndarray) -> None:
        for child in list(element):
            if child.tag != "body":
                continue
            local = np.array([float(v) for v in (child.get("pos") or "0 0 0").split()])
            quat = np.array(
                [float(v) for v in (child.get("quat") or "1 0 0 0").split()]
            )
            world = base_pos + _quat_rotate(base_quat, local)
            world_quat = _quat_mul(base_quat, quat)
            name = child.get("name")
            if name:
                positions[name] = [round(float(v), 6) for v in world]
            walk(child, world, world_quat)

    for worldbody in root.iter("worldbody"):
        walk(worldbody, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    return positions


#: BilliardTable pocket bbox offsets, added at build time by the upstream
#: class (specific_entities/common_containers.py) — the frozen XMLs carry the
#: keypoint sites commented out, so the constants are transcribed here.
_BILLIARD_HOLE_MIN = (-0.07, -0.07, 0.0)
_BILLIARD_HOLE_MAX = (0.07, 0.07, 0.1)
_BILLIARD_TARGET_HOLE = "container_mid_1"


def _vlb_container_params(comp: dict[str, Any], task_label: str) -> dict[str, Any]:
    """Containment geometry for a container component (compile-time).

    Mirrors the upstream mixin dispatch: ``FlatContainer`` footprints come
    from ``horizontal_radius_site`` plus the 0.1 m z band (upstream
    ``FlatContainer.z_threshold`` default); ``BilliardTable`` checks the
    build-time target pocket; other ``CommonContainer`` subclasses use the
    bbox of their group-3 keypoint sites.
    """
    xml_path = VLB_MIRROR / comp["xml_path"]
    if comp.get("class") == "BilliardTable":
        world_holes = _vlb_body_positions(xml_path)
        target = world_holes.get(_BILLIARD_TARGET_HOLE)
        if target is None:
            raise SystemExit(
                f"{task_label}: billiard target hole {_BILLIARD_TARGET_HOLE!r} "
                f"not found in {comp['xml_path']}"
            )
        return {
            "mode": "boxes",
            "regions": [
                {
                    "center": target,
                    "min": [round(float(v), 6) for v in _BILLIARD_HOLE_MIN],
                    "max": [round(float(v), 6) for v in _BILLIARD_HOLE_MAX],
                }
            ],
        }
    sites = _vlb_worldbody_sites(xml_path, ("horizontal_radius_site",))
    if "horizontal_radius_site" in sites:
        x, y = sites["horizontal_radius_site"][:2]
        return {
            "mode": "circle",
            "circle": round(float(np.hypot(x, y)), 6),
            "z_band": [0.0, 0.1],
        }
    key_sites = _vlb_group_sites(xml_path, "3")
    if len(key_sites) >= 2:
        points = np.array(list(key_sites.values()), dtype=float)
        return {
            "mode": "boxes",
            "regions": [
                {
                    "center": [0.0, 0.0, 0.0],
                    "min": [round(float(v), 6) for v in points.min(axis=0)],
                    "max": [round(float(v), 6) for v in points.max(axis=0)],
                }
            ],
        }
    raise SystemExit(
        f"{task_label}: container {comp.get('name')!r} has neither a "
        f"horizontal_radius_site nor group-3 key sites in {comp['xml_path']}"
    )


def _vlb_map_condition(  # noqa: C901 - per-condition transcription branches
    cond_name: str,
    params: dict[str, Any],
    components: dict[str, dict[str, Any]],
    task_label: str,
) -> dict[str, Any]:
    """Map one upstream VLABench condition to a schema v1 success spec.

    Derived numbers (footprint radius, keypoint bboxes, pocket offsets,
    top/bottom site offsets) are parsed from the mirrored component MJCF at
    compile time — never hand-written.
    """
    if cond_name == "above":
        platform = params.get("platform")
        target = params.get("target_entity")
        comp = components.get(platform)
        if comp is None:
            raise SystemExit(
                f"{task_label}: above references unknown platform {platform!r}"
            )
        radius = None
        site = _vlb_worldbody_sites(
            VLB_MIRROR / comp["xml_path"], ("horizontal_radius_site",)
        )
        if "horizontal_radius_site" in site:
            x, y = site["horizontal_radius_site"][:2]
            radius = round(float(np.hypot(x, y)), 6)
        if radius is None:
            raise SystemExit(
                f"{task_label}: platform {platform!r} has no horizontal_radius_site; "
                f"box-footprint platforms are not mapped yet"
            )
        return {
            "type": "above_region",
            "params": {"object": target, "platform": platform, "circle": radius},
        }
    if cond_name == "pour":
        target = params.get("target_entity")
        comp = components.get(target)
        if comp is None:
            raise SystemExit(f"{task_label}: pour references unknown entity {target!r}")
        sites = _vlb_worldbody_sites(
            VLB_MIRROR / comp["xml_path"], ("top_site", "bottom_site")
        )
        missing = {"top_site", "bottom_site"} - set(sites)
        if missing:
            raise SystemExit(
                f"{task_label}: pour target {target!r} lacks sites {sorted(missing)}"
            )
        return {
            "type": "tipped",
            "params": {
                "object": target,
                "top_offset": sites["top_site"],
                "bottom_offset": sites["bottom_site"],
                "threshold": float(params.get("threshold", 0) or 0),
            },
        }
    if cond_name in ("contain", "not_contain"):
        container = params.get("container")
        entities = params.get("entities")
        comp = components.get(container)
        if comp is None:
            raise SystemExit(
                f"{task_label}: {cond_name} references unknown container {container!r}"
            )
        if not isinstance(entities, list) or not entities:
            raise SystemExit(
                f"{task_label}: {cond_name} needs a non-empty entities list"
            )
        geo = _vlb_container_params(comp, task_label)
        return {
            "type": "contains",
            "params": {
                "container": container,
                "objects": list(entities),
                "negate": cond_name == "not_contain",
                **geo,
            },
        }
    if cond_name == "lift":
        entities = params.get("entities")
        if not isinstance(entities, list) or not entities:
            raise SystemExit(f"{task_label}: lift needs a non-empty entities list")
        return {
            "type": "lift",
            "params": {
                "objects": list(entities),
                "min_height": float(params.get("target_height", 0.0)),
            },
        }
    if cond_name == "order":
        entities = params.get("entities")
        axis = params.get("axis", [0])
        if not isinstance(entities, list) or len(entities) < 2:
            raise SystemExit(f"{task_label}: order needs >= 2 entities")
        return {
            "type": "ordered",
            "params": {
                "objects": list(entities),
                "axes": [int(a) for a in axis],
                "offset": float(params.get("offset", 0.1) or 0.1),
            },
        }
    raise SystemExit(f"{task_label}: unmapped upstream condition {cond_name!r}")


def resolve_vlabench(predicates: dict[str, Any]) -> dict[str, Any]:
    """Build the resolved KB for VLABench from frozen instances.

    Iterates the committed inventory; a task is ``pending`` until its
    condition names are covered by ``condition_map`` and every referenced
    component/instance file is mirrored.
    """
    inv = load_inventory("vlabench")
    cmap = predicates.get("condition_map") or {}
    dynamic = set(predicates.get("dynamic_classes") or [])
    table = predicates.get("tasks") or {}
    tasks: dict[str, Any] = {}
    pending: list[str] = []
    pending_reasons: dict[str, str] = {}
    for entry in inv["tasks"]:
        name = entry["name"]
        native = entry["family"].split("/", 1)[-1]
        spec = table.get(native)
        if spec is None:
            pending.append(name)
            pending_reasons[name] = "predicate table 未策划该任务"
            continue
        if spec.get("pending"):
            pending.append(name)
            pending_reasons[name] = str(spec.get("note") or "pending")
            continue
        inst_path = (
            VLB_INSTANCES / f"{name}-privileged" / "environment" / "instance.json"
        )
        if not inst_path.is_file():
            pending.append(name)
            pending_reasons[name] = f"instance 未镜像: {inst_path.name}"
            continue
        data = json.loads(inst_path.read_text(encoding="utf-8"))
        task_cfg = data.get("episode", {}).get("task", {})
        all_components = task_cfg.get("components", [])
        components = {c["name"]: c for c in all_components}

        reasons: list[str] = []
        assets: list[dict[str, Any]] = []
        asset_overrides = spec.get("assets_override") or {}
        for comp in all_components:
            xml_rel = comp.get("xml_path")
            if not xml_rel:
                override = asset_overrides.get(comp["name"])
                if override is None:
                    reasons.append(
                        f"组件 {comp['name']} 无 xml_path（上游过程生成 RandomGeom），"
                        f"需在判据表 assets_override 给出规格"
                    )
                    continue
                assets.append(
                    {
                        "name": comp["name"],
                        "shape_type": override.get("shape_type", "box"),
                        "size": override.get("size", [0.04, 0.04, 0.04]),
                        "position": comp.get("position", [0.0, 0.0, 0.0]),
                        "orientation": _vlb_orientation(
                            comp.get("orientation"), name, comp["name"]
                        ),
                        "static": False,
                        "mass": override.get("mass", 0.2),
                        "friction": override.get("friction", 1.0),
                    }
                )
                continue
            if not (VLB_MIRROR / xml_rel).is_file():
                reasons.append(f"组件未镜像: {xml_rel}")
                continue
            if comp.get("class") not in dynamic:
                assets.append(
                    {
                        "name": comp["name"],
                        "shape_type": "mjcf",
                        "mjcf_path": f"assets/vlabench/{xml_rel}",
                        "position": comp.get("position", [0.0, 0.0, 0.0]),
                        "orientation": _vlb_orientation(
                            comp.get("orientation"), name, comp["name"]
                        ),
                        "static": True,
                    }
                )
                continue
            # Library-style components ship an asset library with an empty
            # worldbody (billiard balls) or a body without geoms (number
            # cubes); upstream code instantiates the actual geoms per class.
            # Their synthesis is a scoped work item — hold them pending.
            inlined = _load_mjcf_inlined(VLB_MIRROR / xml_rel)
            has_geom = any(
                elem.tag == "geom"
                for worldbody in inlined.iter("worldbody")
                for elem in worldbody.iter()
            )
            if not has_geom:
                reasons.append(
                    f"动态组件 {comp['name']}({comp.get('class')}) 为资产库式"
                    f"（worldbody 零 geom，几何由上游类代码合成），"
                    f"需类级实例化移植"
                )
                continue
            assets.append(
                {
                    "name": comp["name"],
                    "shape_type": "mjcf",
                    "mjcf_path": f"assets/vlabench/{xml_rel}",
                    "position": comp.get("position", [0.0, 0.0, 0.0]),
                    "orientation": _vlb_orientation(
                        comp.get("orientation"), name, comp["name"]
                    ),
                    "static": False,
                }
            )

        cond_specs: list[dict[str, Any]] = []
        conditions = task_cfg.get("conditions", {})
        if not conditions:
            reasons.append("instance 无 conditions 块")
        for cond_name, params in conditions.items():
            if cond_name not in cmap:
                reasons.append(f"condition '{cond_name}' 无映射")
                continue
            try:
                cond_specs.append(
                    _vlb_map_condition(cond_name, params, components, name)
                )
            except SystemExit as exc:
                reasons.append(str(exc))
        if not reasons:
            referenced: set[str] = set()
            for spec in cond_specs:
                p = spec.get("params", {})
                if isinstance(p.get("object"), str):
                    referenced.add(p["object"])
                referenced.update(p.get("objects") or ())
                if isinstance(p.get("platform"), str):
                    referenced.add(p["platform"])
                if isinstance(p.get("container"), str):
                    referenced.add(p["container"])
            missing_entities = sorted(referenced - set(components))
            if missing_entities:
                reasons.append(
                    f"条件引用未作为组件出现的实体: {missing_entities}"
                    f"（内嵌于容器/场景 MJCF，需子体抽取）"
                )
        if reasons:
            pending.append(name)
            pending_reasons[name] = "; ".join(reasons)
            continue

        if len(cond_specs) == 1:
            success: dict[str, Any] = cond_specs[0]
        else:
            success = {
                "type": "staged",
                "params": {
                    "stages": [
                        {"stage": "place_fail", "condition": cond}
                        for cond in cond_specs
                    ]
                },
            }
        tasks[name] = {
            "family": entry["family"],
            "goal": spec.get("goal", ""),
            "robot": "franka_panda",
            "scene": "efe_tabletop",
            "task_type": "efe_static",
            "assets_task_relevant": assets,
            "success": success,
            "max_episode_steps": int(spec.get("max_episode_steps", 4000)),
            "efe": {
                "fidelity": "spec",
                "notes": spec.get("note", "")
                or "判据句见 instruction.md；实例为上游冻结 seed 0",
            },
        }
    return {
        "suite": "vlabench",
        "robot": "franka_panda",
        "scene": "efe_tabletop",
        "source": predicates.get("source", {}),
        "pending_tasks": sorted(pending),
        "pending_reasons": pending_reasons,
        "tasks": tasks,
    }


RESOLVERS = {"metaworldplus": resolve_metaworldplus, "vlabench": resolve_vlabench}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_task_yaml(suite: str, name: str, record: dict[str, Any]) -> str:
    """Render one schema v1 task YAML with an EFE provenance header."""
    lines = [
        f"# {EFAIL.format(suite=suite)}",
        f"# family: {record.get('family', '?')} | goal: {record.get('goal', '?')}",
        f"# fidelity: {record.get('efe', {}).get('fidelity', 'spec')}"
        + (
            f" | notes: {record['efe']['notes']}"
            if record.get("efe", {}).get("notes")
            else ""
        ),
        "version: 1",
        "",
        "task:",
        f"  name: {name}",
        f"  type: {record.get('task_type', 'efe_static')}",
        f"  robot: {record['robot']}",
        f"  scene: {record['scene']}",
        f"  max_episode_steps: {record.get('max_episode_steps', 500)}",
        "",
    ]
    assets = record.get("assets_task_relevant") or []
    if assets:
        lines.append("assets:")
        lines.append("  task_relevant:")
        for asset in assets:
            lines.extend(
                "  " + line
                for line in yaml.safe_dump([asset], sort_keys=False).splitlines()
            )
        lines.append("")
    success = record.get("success")
    if success:
        lines.append("success:")
        lines.extend(
            "  " + line
            for line in yaml.safe_dump(success, sort_keys=False).splitlines()
        )
        lines.append("")
    lines.append("evaluation:")
    lines.append("  seeds: [0]")
    lines.append("  episodes_per_seed: 1")
    lines.append("")
    lines.append("simulation:")
    lines.append("  headless: true")
    lines.append("  device: cpu")
    lines.append("")
    return "\n".join(lines)


def render_suite_yaml(suite: str, task_names: list[str]) -> str:
    """Render the suite aggregate YAML listing every task config."""
    rel = f"../../configs/tasks/efe/{suite}"
    lines = [
        f"# EFE {suite} 套件汇总（编译物）；由 tools/compile_efe_suite.py 生成。",
        "version: 1",
        f"name: efe-{suite}",
        "tasks:",
    ]
    lines.extend(f"  - {rel}/{name}.yaml" for name in task_names)
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def load_kb(suite: str) -> dict[str, Any]:
    """Load the resolved knowledge base for a suite."""
    path = COMPILE_DIR / f"{suite}.yaml"
    if not path.is_file():
        raise SystemExit(f"knowledge base not found: {path} (run with --resolve)")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_inventory(suite: str) -> dict[str, Any]:
    """Load the committed upstream task inventory for a suite."""
    path = DATA_EFE / "tasks" / f"{suite}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def check_against_inventory(suite: str, kb: dict[str, Any]) -> list[str]:
    """KB tasks must match the EFE inventory, minus declared pending tasks."""
    errors: list[str] = []
    inv = load_inventory(suite)
    inv_names = {t["name"] for t in inv["tasks"]}
    kb_names = set(kb.get("tasks") or {})
    pending = set(kb.get("pending_tasks") or [])
    unexplained_pending = pending - inv_names
    for name in sorted(unexplained_pending):
        errors.append(f"{suite}: pending task not in inventory: {name}")
    for name in sorted(inv_names - kb_names - pending):
        errors.append(f"{suite}: inventory task missing from KB: {name}")
    for name in sorted(kb_names - inv_names):
        errors.append(f"{suite}: KB task not in inventory: {name}")
    return errors


def build_outputs(suite: str) -> dict[Path, str]:
    """Render the full output file set for a suite (path -> content)."""
    kb = load_kb(suite)
    errors = check_against_inventory(suite, kb)
    if errors:
        raise SystemExit("\n".join(errors))
    if not kb.get("tasks"):
        raise SystemExit(
            f"{suite}: knowledge base has no resolved tasks yet "
            f"({len(kb.get('pending_tasks') or [])} pending) — mirror the "
            f"upstream assets (tools/fetch_upstream_assets.py) and curate "
            f"the predicate table first; refusing to write an empty suite"
        )
    out_dir = CONFIGS_OUT / suite
    outputs: dict[Path, str] = {}
    names = sorted(kb["tasks"])
    for name in names:
        outputs[out_dir / f"{name}.yaml"] = render_task_yaml(
            suite, name, kb["tasks"][name]
        )
    outputs[SUITES_OUT / f"efe_{suite}.yaml"] = render_suite_yaml(suite, names)
    return outputs


def write_outputs(outputs: dict[Path, str]) -> None:
    """Write every rendered file to disk (UTF-8, parents created)."""
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def check_outputs(outputs: dict[Path, str]) -> list[str]:
    """Diff rendered content against what is on disk."""
    errors = []
    for path, content in outputs.items():
        if not path.is_file():
            errors.append(f"missing compiled file: {path}")
            continue
        on_disk = path.read_text(encoding="utf-8")
        if on_disk != content:
            diff = "\n".join(
                difflib.unified_diff(
                    on_disk.splitlines(), content.splitlines(), lineterm="", n=1
                )
            )
            errors.append(f"compiled file out of sync: {path}\n{diff[:2000]}")
    return errors


def extract_specs(suite: str) -> Path:
    """Dump every task's Scoring section for predicate-table curation.

    Reads ``data/efe/upstream/tasks/<suite>/*/instruction.md`` and writes
    ``data/efe/compile/<suite>.specs.md`` with one block per task: the check
    sentence(s) a curator transcribes into ``<suite>.predicates.yaml``.
    """
    upstream = DATA_EFE / "upstream" / "tasks" / suite
    out_path = COMPILE_DIR / f"{suite}.specs.md"
    blocks: list[str] = [f"# {suite} — Scoring sections (curation aid)", ""]
    for task_dir in sorted(upstream.glob("*-privileged")):
        instr = task_dir / "instruction.md"
        if not instr.is_file():
            continue
        text = instr.read_text(encoding="utf-8")
        body = None
        for heading in ("## Success", "## Scoring"):
            section = text.split(heading, 1)
            if len(section) == 2:
                body = section[1].split("\n## ", 1)[0]
                break
        if body is None:
            body = "(no ## Success / ## Scoring section)"
        blocks.append(f"## {task_dir.name.removesuffix('-privileged')}")
        blocks.append("".join(line + "\n" for line in body.strip().splitlines()))
        blocks.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(blocks), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; --check is the CI gate for compiled configs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument(
        "--resolve",
        action="store_true",
        help="rebuild the knowledge base from the curated predicate table",
    )
    parser.add_argument(
        "--extract-specs",
        action="store_true",
        help="dump upstream Scoring sections for predicate curation",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify compiled configs are in sync (CI gate); does not write",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.extract_specs:
        out_path = extract_specs(args.suite)
        print(f"specs -> {out_path}")
        return 0

    if args.resolve:
        pred_path = COMPILE_DIR / f"{args.suite}.predicates.yaml"
        if not pred_path.is_file():
            raise SystemExit(f"predicate table not found: {pred_path}")
        predicates = yaml.safe_load(pred_path.read_text(encoding="utf-8"))
        resolver = RESOLVERS.get(args.suite)
        if resolver is None:
            raise SystemExit(f"no resolver registered for suite {args.suite!r}")
        kb = resolver(predicates)
        kb_path = COMPILE_DIR / f"{args.suite}.yaml"
        kb_path.parent.mkdir(parents=True, exist_ok=True)
        kb_path.write_text(
            "# 编译知识库（生成物）：--resolve 从判据表+上游模型生成；"
            "人工修正会被下次 --resolve 覆盖，\n"
            "# 持久标注请改 .predicates.yaml。\n"
            + yaml.safe_dump(kb, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        logger.info("resolved KB -> %s (%d tasks)", kb_path, len(kb["tasks"]))

    outputs = build_outputs(args.suite)
    if args.check:
        errors = check_outputs(outputs)
        print(
            f"compile --check {args.suite} | files={len(outputs)} "
            f"errors={len(errors)}"
        )
        for error in errors:
            logger.error("%s", error)
        return 1 if errors else 0

    write_outputs(outputs)
    print(f"compiled {args.suite} -> {len(outputs)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
