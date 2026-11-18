#!/usr/bin/env python3
"""Extract Genesis-ready world packages from mirrored RoboCasa EFE instances.

Every EFE RoboCasa task ships ``environment/instance.npz``: the frozen MuJoCo
model (``xml.npy`` — the fully composed kitchen world, textures/fixtures/
robot/objects inlined by reference), the integration state (``integration.npy``
— ``qpos | qvel | act``), the rewritten model arrays and episode metadata.
This tool turns one instance (or all 103) into what the Genesis runtime needs:

* ``assets/robocasa/worlds/<suite>/<task>.xml`` — the world MJCF with every
  ``file=`` path rewritten from the image-absolute prefixes
  (``/opt/robocasa/robocasa/models/assets/...``,
  ``/usr/local/lib/python3.X/site-packages/robosuite/models/assets/...``) to
  this machine's mirrored roots (``data/efe/upstream/*-src``);
* ``<task>.qpos.npy`` / ``<task>.qvel.npy`` — the frozen generalised state
  (split sizes derived from the XML joint walk: free=7/6, ball=4/3, else 1/1);
* ``<task>.meta.json`` — the grading metadata the resolver bakes into the
  compile KB: body/joint/site inventories (joint ranges for the articulated
  checks), each object's ``horizontal_radius_site`` reading (robosuite's exact
  ``horizontal_radius``), eef site names, episode language and the instance
  digest.

No upstream runtime is involved: the frozen composed XML is the spec-level
truth, mirroring the EFE suites' "one frozen instance" definition.

Usage::

    uv run python tools/robocasa_worlds.py --suite robocasa --all
    uv run python tools/robocasa_worlds.py --suite robocasa-gr1 --task robocasa-gr1-pnp-bottle-to-cabinet-close-i00
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "data" / "efe" / "upstream"
WORLDS = ROOT / "assets" / "robocasa" / "worlds"

SUITES = ("robocasa", "robocasa365", "robocasa-gr1")

#: All rewritten asset paths land in one unified per-suite root
#: (``assets/robocasa/<suite>/models/assets``); the fetch tool populates it
#: with the XML-referenced members of the pinned archives plus the
#: referenced files of the suite's robosuite mirror.
WORLDS = ROOT / "assets" / "robocasa"

SUITES = ("robocasa", "robocasa365", "robocasa-gr1")


def _suite_root(suite: str) -> Path:
    """Unified assets root for one suite."""
    return WORLDS / suite / "models" / "assets"


_ROBOSUITE_PREFIX = re.compile(
    r"^/usr/local/lib/python3\.\d+/site-packages/robosuite/models/assets"
)
_IMAGE_ASSET_PREFIX = re.compile(r"^/opt/[^/]+/(?:robocasa/)?models/assets")

#: MuJoCo joint types -> (nq contribution, nv contribution)
_JOINT_Q = {"free": (7, 6), "ball": (4, 3), "slide": (1, 1), "hinge": (1, 1)}

_EEF_SITE_PATTERNS = ("eef", "grip_site", "gripper_site")


def _local_prefixes(suite: str) -> list[tuple[re.Pattern[str], str]]:
    """Ordered (pattern, replacement) path rewrites for one suite.

    Replacements are forward-slash normalised: MuJoCo (which Genesis
    delegates MJCF compilation to, via ``from_xml_string`` without a base
    dir) resolves ``D:/...`` drive paths on Windows but mangles
    backslashes and ``//drive`` UNC spellings.
    """
    root = _suite_root(suite).resolve().as_posix()
    return [
        (_IMAGE_ASSET_PREFIX, root),
        (_ROBOSUITE_PREFIX, root),
    ]


def rewrite_xml(xml: str, suite: str) -> tuple[str, list[str]]:
    """Rewrite image-absolute asset paths to the local mirrors.

    Returns the rewritten XML and the list of distinct rewritten file paths
    that do not exist on disk (the caller reports them as missing assets).
    """
    rewrites = _local_prefixes(suite)
    missing: list[str] = []

    def fix(match: re.Match[str]) -> str:
        path = match.group(1)
        for pattern, replacement in rewrites:
            hit = pattern.match(path)
            if hit is not None:
                new = replacement + path[hit.end() :]
                if not Path(new).is_file():
                    missing.append(new)
                return match.group(0).replace(path, new)
        return match.group(0)

    rewritten = re.sub(r'file="(/[^"]+)"', fix, xml)
    return rewritten, sorted(set(missing))


def _joint_spec(root: ET.Element) -> tuple[list[dict], int, int]:
    """Walk the XML joint tree: per-joint record plus (nq, nv) totals."""
    joints: list[dict] = []
    nq = nv = 0
    for element in root.iter("joint"):
        jtype = element.get("type", "hinge")
        dnq, dnv = _JOINT_Q.get(jtype, (1, 1))
        nq += dnq
        nv += dnv
        joints.append(
            {
                "name": element.get("name", ""),
                "type": jtype,
                "range": element.get("range"),
                "limited": element.get("limited"),
                "body": "",  # filled by caller walk
            }
        )
    return joints, nq, nv


def _inventory(root: ET.Element) -> dict:
    """Bodies / sites / per-object horizontal radii from the world XML."""
    bodies: dict[str, dict] = {}
    sites: dict[str, dict] = {}
    radii: dict[str, float] = {}
    for body in root.iter("body"):
        name = body.get("name")
        if name is None:
            continue
        bodies[name] = {"pos": body.get("pos"), "parent": None}
        for site in body.findall("site"):
            sname = site.get("name", "")
            spos = site.get("pos", "0 0 0")
            sites[sname] = {"body": name, "pos": spos}
            if "horizontal_radius_site" in sname:
                try:
                    radii[name] = round(float(spos.split()[0]), 6)
                except (ValueError, IndexError):
                    continue
    eef_sites = [n for n in sites if any(p in n for p in _EEF_SITE_PATTERNS)]
    return {"bodies": bodies, "sites": sites, "radii": radii, "eef_sites": eef_sites}


def _annotate_world_poses(
    inventory: dict,
    body_pos: np.ndarray,
    body_quat: np.ndarray,
) -> None:
    """Stamp compiled world poses onto the inventory (offline binding).

    MuJoCo numbers bodies in XML document order, so row ``i + 1`` of the
    instance's compiled ``model_body_*`` arrays is the i-th ``<body>``
    (row 0 is the worldbody). The resolver uses these to bind fixtures
    (nearest to the robot base) and initial containment without an engine.
    Site world poses are derived at resolve time from body pose + local
    offset, so only body rows are stamped here.
    """
    names = list(inventory["bodies"])
    for i, name in enumerate(names):
        row = i + 1
        if row >= len(body_pos):
            break
        inventory["bodies"][name]["world"] = [round(float(v), 6) for v in body_pos[row]]
        inventory["bodies"][name]["world_quat"] = [
            round(float(v), 6) for v in body_quat[row]
        ]


def caches_env_attrs(instance: Path) -> dict[str, Any]:
    """Small JSON-encodable env attributes from the frozen instance state."""
    with zipfile.ZipFile(instance) as archive:
        caches = json.loads(
            np.load(archive.open("caches.npy"), allow_pickle=True).item()
        )
    env = caches.get("env")
    return env if isinstance(env, dict) else {}


def extract_one(suite: str, task: str, instance: Path, check_assets: bool) -> dict:
    """Extract one instance.npz into the world package; return its meta."""
    with zipfile.ZipFile(instance) as archive:
        xml = np.load(archive.open("xml.npy"), allow_pickle=True).item()
        integration = np.load(archive.open("integration.npy")).reshape(-1)
        ep_meta = np.load(archive.open("ep_meta.npy"), allow_pickle=True)
        config = np.load(archive.open("config.npy"), allow_pickle=True)
        instance_sha = str(
            np.load(archive.open("instance_sha.npy"), allow_pickle=True).item()
        )
        body_pos = np.load(archive.open("model_body_pos.npy"))
        body_quat = np.load(archive.open("model_body_quat.npy"))

    cfg: dict[str, Any] = {}
    if isinstance(config, np.ndarray) and config.shape == ():
        raw = config.item()
        if isinstance(raw, dict):
            cfg = raw
        else:
            try:
                cfg = json.loads(str(raw))
            except (TypeError, ValueError):
                cfg = {}

    ep = {}
    if isinstance(ep_meta, np.ndarray) and ep_meta.shape == ():
        ep = ep_meta.item()
    if not isinstance(ep, dict):
        try:
            ep = json.loads(str(ep_meta))
        except (TypeError, ValueError):
            ep = {}

    root = ET.fromstring(xml)
    joints, nq, nv = _joint_spec(root)
    inventory = _inventory(root)
    _annotate_world_poses(inventory, body_pos, body_quat)
    qpos = integration[:nq].astype(np.float64)
    qvel = integration[nq : nq + nv].astype(np.float64)
    if qpos.size != nq or qvel.size != nv:
        raise ValueError(
            f"{task}: integration vector too short for nq={nq}, nv={nv} "
            f"(len={integration.size})"
        )

    rewritten, missing = rewrite_xml(xml, suite)

    out_dir = WORLDS / suite / "worlds"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{task}.xml").write_text(rewritten, encoding="utf-8")
    np.save(out_dir / f"{task}.qpos.npy", qpos)
    np.save(out_dir / f"{task}.qvel.npy", qvel)

    meta = {
        "suite": suite,
        "task": task,
        "instance_sha": instance_sha,
        "nq": nq,
        "nv": nv,
        "env_name": cfg.get("env_name", ""),
        "env_attrs": {
            k: v
            for k, v in (caches_env_attrs(instance) or {}).items()
            if isinstance(v, (str, int, float, bool, list))
        },
        "language": ep.get("lang", ""),
        "joints": joints,
        **inventory,
    }
    (out_dir / f"{task}.meta.json").write_text(
        json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8"
    )
    if check_assets and missing:
        print(
            f"WARNING {task}: {len(missing)} referenced files missing:", file=sys.stderr
        )
        for path in missing[:8]:
            print(f"   {path}", file=sys.stderr)
    return meta


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--task")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--no-check", action="store_true", help="skip missing-asset warnings"
    )
    args = parser.parse_args(argv)
    if not args.all and not args.task:
        raise SystemExit("give --task <name> or --all")

    tasks_dir = UPSTREAM / "tasks" / args.suite
    if args.all:
        instances = sorted(tasks_dir.glob("*-privileged/environment/instance.npz"))
    else:
        instances = [tasks_dir / f"{args.task}-privileged/environment/instance.npz"]
    failures = 0
    for instance in instances:
        task = instance.parent.parent.name.removesuffix("-privileged")
        try:
            meta = extract_one(
                args.suite, task, instance, check_assets=not args.no_check
            )
        except Exception as exc:  # noqa: BLE001 - batch keeps going
            failures += 1
            print(f"ERROR {task}: {exc}", file=sys.stderr)
            continue
        print(
            f"{task}: bodies={len(meta['bodies'])} joints={meta['nq']}/{meta['nv']} "
            f"eef={meta['eef_sites']} radii={len(meta['radii'])}"
        )
    print(f"done: {len(instances) - failures} extracted, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
