#!/usr/bin/env python3
"""RoboTwin asset conversion toolchain (migration document section 4).

Converts a RoboTwin-style asset tree (embodiments + RoboTwin-OD objects)
into a Genesis-loadable tree without polluting the source assets:

1. Copies the full source tree to the output directory (annotations and
   keypoint files are preserved verbatim, section 4.2).
2. Rewrites every URDF in place:
   - expands ``<mimic>`` joints into plain joints (Genesis issue #678) and
     records the mimic mapping so the control layer can replicate targets
     with the multiplier (section 4.1);
   - reports floating-base joints (embodiments must be loaded with
     ``fixed=True``, section 4.1 checklist);
   - reports links missing ``<inertial>`` blocks;
   - rewrites resolvable ROS ``package://`` mesh URIs to relative paths
     (RoboTwin piper ships ``package://piper_description/meshes/...`` with
     meshes at ``piper/meshes/...``), reporting the rest as missing.
3. Optionally runs a Genesis load smoke test per URDF (``--smoke-test``).
4. Writes ``conversion_report.json`` summarizing every asset.

For GLB-based object libraries (RoboTwin-OD: ``<class>/{collision,visual}/
base*.glb`` + ``model_data*.json``, no URDFs) use ``--mesh-smoke``: no tree
copy, samples one mesh per collision/visual group per object class and
verifies it loads in Genesis (CPU).

Usage:
    python tools/convert_assets.py <src_dir> <out_dir> [--smoke-test]
    python tools/convert_assets.py <src_dir> <out_dir> --mesh-smoke
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def parse_urdf(path: str | Path) -> ET.ElementTree:
    """Parse a URDF file into an ElementTree."""
    return ET.parse(str(path))


def find_root_link(tree: ET.ElementTree) -> str | None:
    """Return the name of the root link (no parent joint), if any."""
    root = tree.getroot()
    children = {
        child.text.strip()
        for joint in root.iter("joint")
        if (child := joint.find("child")) is not None and child.text
    }
    for link in root.iter("link"):
        name = link.get("name")
        if name and name not in children:
            return name
    return None


def check_fixed_base(tree: ET.ElementTree) -> bool:
    """Return True if the URDF has no floating joint (fixed-base robot)."""
    for joint in tree.getroot().iter("joint"):
        if joint.get("type") == "floating":
            return False
    return True


def expand_mimic_joints(tree: ET.ElementTree) -> dict[str, dict[str, Any]]:
    """Replace ``<mimic>`` joints with plain joints (Genesis issue #678).

    Mutates the tree in place. Returns the mimic mapping
    ``{slave_joint: {"master": ..., "multiplier": ..., "offset": ...}}``
    so the control layer can replicate master targets onto slave joints.
    """
    mapping: dict[str, dict[str, Any]] = {}
    for joint in tree.getroot().iter("joint"):
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        name = joint.get("name", "")
        mapping[name] = {
            "master": mimic.get("joint", ""),
            "multiplier": float(mimic.get("multiplier", "1") or "1"),
            "offset": float(mimic.get("offset", "0") or "0"),
        }
        joint.remove(mimic)
    return mapping


def find_missing_inertial(tree: ET.ElementTree) -> list[str]:
    """Return names of links that have geometry but no ``<inertial>``."""
    missing: list[str] = []
    root = tree.getroot()
    root_link = find_root_link(tree)
    for link in root.iter("link"):
        name = link.get("name", "")
        if name == root_link:
            continue  # fixed base links legitimately carry no inertial
        has_geometry = (
            link.find("visual") is not None or link.find("collision") is not None
        )
        if has_geometry and link.find("inertial") is None:
            missing.append(name)
    return missing


def find_mesh_references(tree: ET.ElementTree) -> list[str]:
    """Return all mesh filenames referenced by the URDF."""
    refs: list[str] = []
    for mesh in tree.getroot().iter("mesh"):
        filename = mesh.get("filename")
        if filename:
            refs.append(filename)
    return refs


def _build_file_index(src_root: Path) -> dict[str, Path]:
    """Index all files under ``src_root`` by their posix relative path."""
    return {
        p.relative_to(src_root).as_posix(): p
        for p in src_root.rglob("*")
        if p.is_file()
    }


def _lookup_package_uri(filename: str, index: dict[str, Path]) -> Path | None:
    """Resolve ``package://<pkg>/<path...>`` against the file index.

    ROS package names rarely match the extracted directory layout (e.g.
    RoboTwin piper ships ``package://piper_description/meshes/x.STL`` with
    meshes at ``piper/meshes/x.STL``), so match by progressively shorter
    path tails.
    """
    parts = filename[len("package://") :].replace("\\", "/").split("/")
    for drop in range(len(parts)):
        tail = "/".join(parts[drop:])
        for key, path in index.items():
            if key == tail or key.endswith("/" + tail):
                return path
    return None


def _resolve_mesh(
    filename: str,
    urdf_dir: Path,
    src_root: Path,
    index: dict[str, Path] | None = None,
) -> Path:
    """Resolve a mesh reference to a candidate path inside the source tree."""
    if filename.startswith("package://"):
        if index is not None:
            hit = _lookup_package_uri(filename, index)
            if hit is not None:
                return hit
        # Fall back to the unresolvable candidate (reported as missing).
        parts = filename[len("package://") :].split("/")
        return src_root.joinpath(*parts)
    return (urdf_dir / filename).resolve()


def convert_urdf(
    src_path: Path,
    dst_path: Path,
    src_root: Path,
    index: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Convert one URDF file and return its report entry."""
    if index is None:
        index = _build_file_index(src_root)
    tree = parse_urdf(src_path)
    mimic_map = expand_mimic_joints(tree)
    fixed_base = check_fixed_base(tree)
    missing_inertial = find_missing_inertial(tree)

    # Rewrite resolvable package:// mesh URIs to relative paths so Genesis
    # (and other non-ROS loaders) can find the meshes (finding #5).
    package_rewrites: dict[str, str] = {}
    for mesh in tree.getroot().iter("mesh"):
        filename = mesh.get("filename")
        if not filename or not filename.startswith("package://"):
            continue
        resolved = _lookup_package_uri(filename, index)
        if resolved is None:
            continue
        rel = os.path.relpath(resolved, src_path.parent.resolve())
        package_rewrites[filename] = Path(rel).as_posix()
        mesh.set("filename", package_rewrites[filename])

    missing_meshes = [
        ref
        for ref in find_mesh_references(tree)
        if not _resolve_mesh(ref, src_path.parent, src_root, index).exists()
    ]
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(dst_path), xml_declaration=True, encoding="UTF-8")
    joints = list(tree.getroot().iter("joint"))
    return {
        "urdf": str(dst_path),
        "fixed_base": fixed_base,
        "n_joints": len(joints),
        "mimic_map": mimic_map,
        "package_uri_rewrites": package_rewrites,
        "missing_inertial": missing_inertial,
        "missing_meshes": missing_meshes,
        "load_ok": None,
        "error": None,
    }


def _smoke_test_urdf(entry: dict[str, Any], urdf_path: Path) -> None:
    """Try loading a converted URDF in Genesis (CPU) and update the entry."""
    import genesis as gs  # lazy import: only needed for --smoke-test

    from cloud_robotics_sim.utils.genesis_compat import genesis_init

    genesis_init(use_cuda=False)
    try:
        scene = gs.Scene(show_viewer=False)
        robot = scene.add_entity(
            gs.morphs.URDF(file=str(urdf_path), fixed=True),
        )
        scene.build()
        entry["load_ok"] = True
        entry["n_dofs"] = int(getattr(robot, "n_dofs", 0))
    except Exception as exc:  # noqa: BLE001 - report any load failure
        entry["load_ok"] = False
        entry["error"] = f"{type(exc).__name__}: {exc}"


def _smoke_test_mesh(mesh_path: Path) -> dict[str, Any]:
    """Try loading one mesh (GLB/OBJ/...) in Genesis (CPU).

    ``convexify=False`` skips CoACD convex decomposition - it is prohibitively
    slow on dense visual meshes (100+s per mesh) and irrelevant for a load
    smoke test (finding: 002_bowl/visual/base1.glb crashed the loader).
    """
    import genesis as gs  # lazy import: only needed for mesh smoke tests

    from cloud_robotics_sim.utils.genesis_compat import genesis_init

    genesis_init(use_cuda=False)
    entry: dict[str, Any] = {"mesh": str(mesh_path), "load_ok": None, "error": None}
    try:
        scene = gs.Scene(show_viewer=False)
        scene.add_entity(
            gs.morphs.Mesh(file=str(mesh_path), fixed=True, convexify=False)
        )
        scene.build()
        entry["load_ok"] = True
    except Exception as exc:  # noqa: BLE001 - report any load failure
        entry["load_ok"] = False
        entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


def _smoke_test_meshes_batched(
    mesh_paths: list[Path],
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    """Load meshes in Genesis, building one scene per ``batch_size`` meshes.

    Scene ``build()`` dominates the cost (~50s per build on CPU) while mesh
    parsing is cheap (~1-9s), so batching amortizes the build. If a batch
    fails, fall back to per-mesh loads to attribute the failure.
    """
    import genesis as gs  # lazy import: only needed for mesh smoke tests

    from cloud_robotics_sim.utils.genesis_compat import genesis_init

    genesis_init(use_cuda=False)
    entries: list[dict[str, Any]] = []
    for start in range(0, len(mesh_paths), batch_size):
        batch = mesh_paths[start : start + batch_size]
        batch_entries: list[dict[str, Any]] = [
            {"mesh": str(p), "load_ok": None, "error": None} for p in batch
        ]
        try:
            scene = gs.Scene(show_viewer=False)
            for i, path in enumerate(batch):
                scene.add_entity(
                    gs.morphs.Mesh(
                        file=str(path),
                        fixed=True,
                        convexify=False,
                        pos=(i * 0.5, 0.0, 0.0),
                    )
                )
            scene.build()
            for entry in batch_entries:
                entry["load_ok"] = True
        except Exception:  # noqa: BLE001 - attribute failure per mesh below
            for path, entry in zip(batch, batch_entries):
                result = _smoke_test_mesh(path)
                entry["load_ok"] = result["load_ok"]
                entry["error"] = result["error"]
        entries.extend(batch_entries)
    return entries


def smoke_test_meshes(
    src_dir: str | Path,
    out_dir: str | Path,
    batch_size: int = 16,
    visual_sample: int | None = None,
) -> dict[str, Any]:
    """Mesh-load smoke test for GLB object libraries (RoboTwin-OD layout).

    Objects ship as ``<class>/{collision,visual}/base*.glb`` plus
    ``model_data*.json`` metadata - there are no URDFs to convert, so this
    mode does NOT copy the tree. It samples the first collision mesh of
    every object class plus the first visual mesh of the first
    ``visual_sample`` classes (None = all classes; visual GLBs parse
    ~10-90s each in Genesis due to texture conversion, so capping them
    keeps the run tractable). Scenes are built in batches to amortize the
    per-scene build cost, and the report is checkpointed after each batch.
    """
    src_dir = Path(src_dir).resolve()
    out_dir = Path(out_dir).resolve()
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source asset directory not found: {src_dir}")

    samples: list[Path] = []
    class_dirs = [d for d in sorted(src_dir.iterdir()) if d.is_dir()]
    roots = class_dirs or [src_dir]
    for class_idx, class_dir in enumerate(roots):
        groups = [class_dir / "collision"]
        if visual_sample is None or class_idx < visual_sample:
            groups.append(class_dir / "visual")
        groups = [g for g in groups if g.is_dir()] or [class_dir]
        for group in groups:
            meshes = sorted(
                p
                for p in group.rglob("*")
                if p.suffix.lower() in (".glb", ".obj", ".stl", ".dae")
            )
            if meshes:
                samples.append(meshes[0])

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "conversion_report.json"
    entries: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    for start in range(0, len(samples), batch_size):
        entries.extend(
            _smoke_test_meshes_batched(
                samples[start : start + batch_size], batch_size=batch_size
            )
        )
        report = {
            "src_dir": str(src_dir),
            "out_dir": str(out_dir),
            "mode": "mesh-smoke",
            "n_assets": len(entries),
            "n_samples_planned": len(samples),
            "n_classes": len(roots),
            "visual_sample": visual_sample,
            "complete": start + batch_size >= len(samples),
            "assets": entries,
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def convert_assets(
    src_dir: str | Path,
    out_dir: str | Path,
    *,
    smoke_test: bool = False,
) -> dict[str, Any]:
    """Convert an asset tree and write ``conversion_report.json``.

    Returns the report dict (also persisted under ``out_dir``).
    """
    src_dir = Path(src_dir).resolve()
    out_dir = Path(out_dir).resolve()
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source asset directory not found: {src_dir}")

    # 1. Mirror the full tree (meshes, annotations, configs) verbatim.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(src_dir, out_dir)

    # 2. Rewrite URDFs inside the output tree.
    index = _build_file_index(src_dir)
    assets: list[dict[str, Any]] = []
    for dst_urdf in sorted(out_dir.rglob("*.urdf")):
        rel = dst_urdf.relative_to(out_dir)
        entry = convert_urdf(src_dir / rel, dst_urdf, src_dir, index)
        entry["urdf"] = str(rel)
        assets.append(entry)

    # 3. Optional Genesis load smoke test (section 4.3).
    if smoke_test:
        for entry, dst_urdf in zip(assets, sorted(out_dir.rglob("*.urdf"))):
            _smoke_test_urdf(entry, dst_urdf)

    report: dict[str, Any] = {
        "src_dir": str(src_dir),
        "out_dir": str(out_dir),
        "n_assets": len(assets),
        "assets": assets,
    }
    report_path = out_dir / "conversion_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src_dir", help="Source RoboTwin asset directory")
    parser.add_argument("out_dir", help="Output directory for converted assets")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Load every converted URDF in Genesis (CPU) and report results",
    )
    parser.add_argument(
        "--mesh-smoke",
        action="store_true",
        help="Mesh-only smoke test for GLB object libraries (no tree copy): "
        "samples one mesh per collision/visual group per object class",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Meshes per scene build in --mesh-smoke mode (default: 16)",
    )
    parser.add_argument(
        "--visual-sample",
        type=int,
        default=None,
        help="In --mesh-smoke mode, only sample visual meshes of the first N "
        "classes (default: all; visual GLBs are slow to parse)",
    )
    args = parser.parse_args(argv)

    if args.mesh_smoke:
        report = smoke_test_meshes(
            args.src_dir,
            args.out_dir,
            batch_size=args.batch_size,
            visual_sample=args.visual_sample,
        )
        n_ok = sum(1 for a in report["assets"] if a["load_ok"])
        print(
            f"Mesh smoke test: {n_ok}/{report['n_assets']} samples loaded "
            f"({report['n_classes']} classes)"
        )
        for a in report["assets"]:
            if not a["load_ok"]:
                print(f"  FAILED: {a['mesh']}: {a['error']}")
        return 0 if n_ok == report["n_assets"] else 1

    report = convert_assets(args.src_dir, args.out_dir, smoke_test=args.smoke_test)
    n_mimic = sum(1 for a in report["assets"] if a["mimic_map"])
    n_missing_mesh = sum(1 for a in report["assets"] if a["missing_meshes"])
    print(f"Converted {report['n_assets']} URDF asset(s) -> {report['out_dir']}")
    print(f"  with mimic joints expanded: {n_mimic}")
    print(f"  with missing mesh refs:   {n_missing_mesh}")
    if args.smoke_test:
        n_ok = sum(1 for a in report["assets"] if a["load_ok"])
        print(f"  smoke test passed:        {n_ok}/{report['n_assets']}")
        if n_ok < report["n_assets"]:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
