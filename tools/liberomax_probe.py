#!/usr/bin/env python3
"""Compile-time probe for LIBERO-MAX cases (runs under the ``.lmxvenv`` probe venv).

Rebuilds one frozen case's MuJoCo scene with the pinned upstream stack
(LIBERO-PRO @ 2b910b5 + robosuite 1.4.0), applies the frozen ``.pruned_init``
sim state *and* the LIBERO-PRO substrate perturbations exactly like
LIBERO-MAX's ``ProSubstrateEnv.set_init_state``, then dumps everything the
Genesis port needs per case:

* world poses (pos + wxyz quat) of every movable object and fixture,
  with the source MJCF path and object class (thresholds transcribed from
  the mirrored LIBERO-PRO source, not re-derived);
* per-joint qpos of articulated fixtures plus their ``articulation`` property
  ranges (drawer close/open, knob turn-on/off thresholds);
* every BDDL region site (world pos, orientation matrix, half extents) —
  the geometry LIBERO's ``On``/``In`` predicates test against;
* the parsed BDDL goal, the scenario (setup + trigger + change), and a
  sanity ``success_at_init`` flag (must be false — same invariant EFE
  freezes).

The committed Genesis configs consume the JSON dumps; neither the probe nor
the runtime imports robosuite/LIBERO outside ``.lmxvenv``.

Usage (one case or all 64, gap-filling)::

    .lmxvenv/Scripts/python.exe tools/liberomax_probe.py --case data/efe/upstream/tasks/libero-max/<task>-standard/environment/sim/instance.json
    .lmxvenv/Scripts/python.exe tools/liberomax_probe.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT
PRO_SRC = REPO / "data" / "efe" / "upstream" / "libero-pro-src"
LMX_SRC = REPO / "data" / "efe" / "upstream" / "libero-max-src" / "src"
PRO_DATA = REPO / "data" / "efe" / "upstream" / "libero-pro-data"
ASSETS = REPO / "assets" / "liberomax"
TASKS = REPO / "data" / "efe" / "upstream" / "tasks" / "libero-max"
CONFIG_DIR = REPO / ".efe_stage" / "lmx-config"
DUMPS = REPO / "data" / "efe" / "upstream" / "liberomax-dumps"

# Physics-only probing: no GL context is ever created (see
# tools/setup_lmx_probe_venv.py for the robosuite macro override).
os.environ.setdefault("MUJOCO_GL", "disable")
os.environ["LIBERO_CONFIG_PATH"] = str(CONFIG_DIR)

sys.path.insert(0, str(LMX_SRC))
sys.path.insert(0, str(PRO_SRC))


def ensure_config() -> None:
    """Write the LIBERO config.yaml pointing at the local mirrors."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / "config.yaml").write_text(
        "assets: {assets}\n"
        "bddl_files: {bddl}\n"
        "benchmark_root: {root}\n"
        "datasets: {datasets}\n"
        "init_states: {init}\n".format(
            assets=ASSETS.as_posix(),
            bddl=(PRO_DATA / "bddl_files").as_posix(),
            root=(PRO_SRC / "libero" / "libero").as_posix(),
            datasets=(REPO / ".efe_stage" / "nonexistent").as_posix(),
            init=(PRO_DATA / "init_files").as_posix(),
        ),
        encoding="utf-8",
    )


def load_pruned_init(path: Path) -> np.ndarray:
    """Decode a torch-saved numpy array without torch (zip pickle)."""
    """Decode a torch-saved numpy array without torch (zip pickle)."""
    with zipfile.ZipFile(path) as zf:
        data = pickle.loads(zf.read("archive/data.pkl"))
    return np.asarray(data, dtype=np.float64)


def _xml_relpath(obj: object) -> str | None:
    """Source MJCF path of a robosuite/LIBERO object, relative to the asset root.

    LIBERO object classes resolve their MJCF relative to the *source tree*
    (``.../libero/libero/assets/<dir>/<obj>/<obj>.xml``), not the configured
    assets path; compiled configs reference the ``assets/liberomax`` mirror,
    which carries identical bytes from the same pinned commit.
    """
    file_attr = getattr(obj, "file", None)
    for cand in (file_attr, getattr(obj, "filename", None)):
        if isinstance(cand, str) and cand:
            norm = cand.replace("\\", "/")
            marker = "libero/libero/assets/"
            if marker in norm:
                return norm.split(marker, 1)[1]
            return Path(norm).name
    return None


def _flatten_goal(node: object) -> list[list[str]]:
    """Reduce LIBERO's parsed ``goal_state`` to a list of predicate calls.

    The parsed shape is a flat list of lower-case predicate tokens, e.g.
    ``[['on', 'bowl', 'cabinet_top_side'], ['and', ...]]``; nested ``and``
    nodes are flattened for robustness.
    """
    preds: list[list[str]] = []

    def walk(item: object) -> None:
        if isinstance(item, (list, tuple)) and item:
            head = item[0]
            if isinstance(head, str) and head.lower() == "and":
                for child in item[1:]:
                    walk(child)
            elif isinstance(head, str):
                preds.append([str(x) for x in item])
            else:
                for child in item:
                    walk(child)

    walk(node)
    return preds


def _joint_qpos(sim: object, joint: str) -> float | list[float]:
    value = np.atleast_1d(np.asarray(sim.data.get_joint_qpos(joint), dtype=float))
    if value.size == 1:
        return float(value[0])
    return value.tolist()


def probe_case(inst_path: Path) -> dict:
    """Rebuild one frozen case and dump its compiled-config facts."""
    from libero.libero.envs.env_wrapper import ControlEnv
    from libero_max import pro_runtime

    # Physics-only reference layouts: the stock helper builds an
    # OffScreenRenderEnv (OSMesa/EGL), which does not exist in this venv.
    def _reference_layout(bddl_path: str):
        ref_env = ControlEnv(
            bddl_file_name=bddl_path,
            camera_heights=64,
            camera_widths=64,
            camera_names=["agentview"],
            use_camera_obs=False,
            has_offscreen_renderer=False,
        )
        try:
            return pro_runtime._model_layout(ref_env.sim.model)
        finally:
            ref_env.close()

    pro_runtime._reference_layout = _reference_layout

    case = json.loads(inst_path.read_text(encoding="utf-8"))["case"]
    variant = case["substrate_variant"]
    bddl_path = PRO_DATA / "bddl_files" / variant["bddl_file"]
    init_path = PRO_DATA / "init_files" / variant["init_states_file"]
    initial_states = load_pruned_init(init_path)
    frozen = initial_states[int(case["init_state_index"])]

    env = ControlEnv(
        bddl_file_name=str(bddl_path),
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
    )
    wrapper = pro_runtime.ProSubstrateEnv(
        env,
        {
            "substrate_variant": variant,
            "task_suite_name": case["task_suite_name"],
            "task_name": case.get("task_name"),
            "scenario": {"seed": case["scenario"]["seed"]},
        },
    )
    try:
        env.reset()
        wrapper.set_init_state(np.asarray(frozen, dtype=np.float64))
        base = env.env
        sim = base.sim
        sim.forward()

        def body_state(name: str) -> dict:
            body_id = base.obj_body_id[name]
            return {
                "pos": sim.data.body_xpos[body_id].tolist(),
                "quat": sim.data.body_xquat[body_id].tolist(),
            }

        objects = {}
        for name, obj in sorted(base.objects_dict.items()):
            entry = body_state(name)
            entry["cls"] = type(obj).__name__
            entry["module"] = type(obj).__module__.rsplit(".", 1)[-1]
            entry["xml"] = _xml_relpath(obj)
            entry["joints"] = {
                j: _joint_qpos(sim, j) for j in getattr(obj, "joints", []) or []
            }
            objects[name] = entry

        fixtures = {}
        for name, obj in sorted(base.fixtures_dict.items()):
            entry = body_state(name)
            entry["cls"] = type(obj).__name__
            entry["xml"] = _xml_relpath(obj)
            props = getattr(obj, "object_properties", None) or {}
            articulation = props.get("articulation")
            if articulation:
                entry["articulation"] = json.loads(json.dumps(articulation))
            entry["joints"] = {
                j: _joint_qpos(sim, j) for j in getattr(obj, "joints", []) or []
            }
            fixtures[name] = entry

        sites = {}
        for name, site in sorted(base.object_sites_dict.items()):
            sites[name] = {
                "parent": getattr(site, "parent_name", None),
                "pos": sim.data.get_site_xpos(name).tolist(),
                "xmat": np.asarray(sim.data.get_site_xmat(name), dtype=float).tolist(),
                "size": [float(v) for v in site.size],
                "type": getattr(site, "site_type", "box"),
                # Region sites may carry the articulated joint they name (e.g.
                # a drawer slide); LIBERO's Close/TurnOn evaluate those qpos
                # against the *parent* fixture's articulation ranges.
                "joints": [str(j) for j in (getattr(site, "joints", None) or [])],
            }

        goal = _flatten_goal(base.parsed_problem.get("goal_state"))
        success_at_init = bool(env.check_success())

        # Arena support surface: every LIBERO problem builds its table as box
        # geoms on a body named "table" (robosuite TableArena). Floor scenes
        # have none — the efe_tabletop floor stands in there.
        arena_tables = []
        seen_boxes: set[tuple[float, ...]] = set()
        model = sim.model
        for geom_id in range(model.ngeom):
            body_id = int(model.geom_bodyid[geom_id])
            body_name = model.body_id2name(body_id)
            if body_name != "table" or int(model.geom_type[geom_id]) != 6:  # box
                continue
            local_pos = np.asarray(model.geom_pos[geom_id], dtype=float)
            body_pos = np.asarray(sim.data.body_xpos[body_id], dtype=float)
            body_mat = np.asarray(sim.data.body_xmat[body_id], dtype=float).reshape(
                3, 3
            )
            world_pos = body_pos + body_mat @ local_pos
            size = np.asarray(model.geom_size[geom_id], dtype=float)
            key = tuple(np.round(np.concatenate([world_pos, size]), 4))
            if key in seen_boxes:
                continue
            seen_boxes.add(key)
            arena_tables.append(
                {
                    "pos": world_pos.tolist(),
                    "quat": sim.data.body_xquat[body_id].tolist(),
                    "size": size.tolist(),
                }
            )

        scenario = case["scenario"]
        robot_model = base.robots[0].robot_model
        robot_base = base.sim.data.body_xpos[
            base.sim.model.body_name2id(robot_model.root_body)
        ].tolist()

        return {
            "task_name": case.get("task_name"),
            "task_suite_name": case["task_suite_name"],
            "language": variant.get("language"),
            "problem": base.__class__.__name__,
            "bddl_file": variant["bddl_file"],
            "init_states_file": variant["init_states_file"],
            "init_state_index": int(case["init_state_index"]),
            "policy_seed": int(case["policy_seed"]),
            "robot_base": robot_base,
            "arena_tables": arena_tables,
            "objects": objects,
            "fixtures": fixtures,
            "sites": sites,
            "goal": goal,
            "success_at_init": success_at_init,
            "event": {
                "setup": scenario.get("setup") or [],
                "trigger": scenario["trigger"],
                "change": scenario["change"],
                "change_type": scenario["change_type"],
            },
            "substrate": {
                "category": variant["category"],
                "applied_components": wrapper.substrate_info.get(
                    "applied_components", []
                ),
                "state_adapter": wrapper.substrate_info.get("state_adapter"),
            },
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    """Probe one case or every registered case (gap-fill)."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case", type=Path, help="one standard-mode instance.json")
    group.add_argument(
        "--all", action="store_true", help="probe every registered case (gap-fill)"
    )
    args = parser.parse_args(argv)

    ensure_config()
    DUMPS.mkdir(parents=True, exist_ok=True)

    if args.all:
        cases = sorted(TASKS.glob("*-standard/environment/sim/instance.json"))
    else:
        cases = [args.case]
    failures = 0
    for inst in cases:
        task_dir_name = inst.parts[-4]  # <task>-standard
        name = task_dir_name.removesuffix("-standard")
        out = DUMPS / f"{name}.json"
        if out.is_file() and out.stat().st_size > 0:
            print(f"present {name}", flush=True)
            continue
        try:
            dump = probe_case(inst)
        except Exception as exc:  # noqa: BLE001 - per-case report, keep going
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        out.write_text(
            json.dumps(dump, indent=1, sort_keys=False) + "\n", encoding="utf-8"
        )
        flag = "!! SUCCESS-AT-INIT" if dump["success_at_init"] else ""
        print(
            f"probed {name}: {len(dump['objects'])} obj, "
            f"{len(dump['fixtures'])} fix, {len(dump['sites'])} sites {flag}",
            flush=True,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
