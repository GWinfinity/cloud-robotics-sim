#!/usr/bin/env python3
"""Import externally generated 3D object meshes into the RoboTwin object library.

Takes meshes produced by image-to-3D generators (e.g. TRELLIS.2), Objaverse
downloads, or any trimesh-loadable file (``.glb`` / ``.obj`` / ``.ply`` /
``.stl``), and converts them into the RoboTwin-OD GLB layout understood by
:class:`cloud_robotics_sim.robotwin.object_library.RoboTwinObjectLibrary`::

    <objects_dir>/NNN_<class>/
        collision/base<N>.glb     # cleaned + normalized collision mesh
        model_data<N>.json        # center / extents / scale metadata

Per input mesh the importer:

1. Loads and merges all geometry into a single mesh (textures are kept when
   the mesh does not need decimation).
2. Optionally converts Z-up inputs (many OBJ exporters) to the glTF Y-up
   convention used by the library (``--z-up``).
3. Recenters the mesh: xz center at the origin, base at ``y = 0`` — the same
   convention as the shipped RoboTwin assets, so objects spawn resting on
   the support surface.
4. Uniformly rescales so the largest bounding-box edge equals
   ``--target-size`` (default 0.15 m), baking the scale into the vertices
   (``model_data`` scale stays ``[1, 1, 1]`` and the runtime
   ``normalize_scale`` fallback leaves the asset untouched).
5. Decimates with quadric simplification when the face count exceeds
   ``--max-faces`` (keeps the 4 GB GPU contact solver happy; note:
   decimation drops UVs/textures).
6. Exports ``collision/base<N>.glb`` and writes ``model_data<N>.json``.

Multiple input files become multiple *instances* of one class — the
"same category, different styles" expansion use case. Instance numbering
continues after the highest existing ``model_data<N>.json`` of the class,
so repeated invocations append more styles to the same class.

Usage:
    # one new class, three style instances
    python tools/import_generated_objects.py bottle_a.glb bottle_b.glb bottle_c.glb \
        --class-name bottle-gen

    # append more styles to an existing class
    python tools/import_generated_objects.py extra.glb --class-name 001_bottle

    # a whole directory of meshes, Z-up OBJ files
    python tools/import_generated_objects.py downloads/mugs/ --class-name mug --z-up
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import trimesh

logger = logging.getLogger("import_generated_objects")

MESH_SUFFIXES = {".glb", ".gltf", ".obj", ".ply", ".stl", ".off"}
_CLASS_DIR_RE = re.compile(r"^(\d{3})_(.+)$")


@dataclass
class ImportedInstance:
    """Record of one imported mesh instance."""

    class_name: str
    index: int
    glb_path: Path
    metadata_path: Path
    extents: tuple[float, float, float]
    faces_in: int
    faces_out: int
    watertight: bool
    warnings: list[str] = field(default_factory=list)


def collect_input_files(inputs: list[str]) -> list[Path]:
    """Expand files and directories into a sorted list of mesh files."""
    files: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            files.extend(
                sorted(p for p in path.iterdir() if p.suffix.lower() in MESH_SUFFIXES)
            )
        elif path.is_file() and path.suffix.lower() in MESH_SUFFIXES:
            files.append(path)
        elif path.exists():
            raise ValueError(f"unsupported mesh file type: {raw}")
        else:
            raise FileNotFoundError(f"not a mesh file or directory: {raw}")
    if not files:
        raise ValueError("no mesh files found in the given inputs")
    return files


def resolve_class_name(class_name: str, objects_dir: Path) -> str:
    """Return a ``NNN_name`` class dir name, auto-assigning the id if missing.

    A bare name (``bottle-gen``) gets the next free 3-digit id after the
    highest existing class. An already prefixed name (``001_bottle``) is
    returned unchanged.
    """
    if _CLASS_DIR_RE.match(class_name):
        return class_name
    max_id = 0
    if objects_dir.is_dir():
        for d in objects_dir.iterdir():
            m = _CLASS_DIR_RE.match(d.name)
            if d.is_dir() and m:
                max_id = max(max_id, int(m.group(1)))
    return f"{max_id + 1:03d}_{class_name}"


def next_instance_index(class_dir: Path) -> int:
    """First free instance index after the existing ``model_data<N>.json``."""
    max_idx = -1
    if class_dir.is_dir():
        for path in class_dir.glob("model_data*.json"):
            m = re.search(r"model_data(\d+)\.json$", path.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def load_mesh(path: Path, z_up: bool = False) -> "trimesh.Trimesh":
    """Load ``path`` as a single merged trimesh mesh (Y-up frame).

    Multi-geometry scenes are concatenated; visual attributes are preserved
    where trimesh supports it. With ``z_up=True`` the loaded geometry is
    rotated from Z-up to the glTF Y-up convention first.
    """
    import trimesh

    loaded = trimesh.load(str(path))
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"no triangle geometry found in {path}")
        mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError(f"unsupported geometry type in {path}: {type(loaded)}")
    if z_up:
        # rotate +90 deg about X: Z-up -> Y-up
        rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1.0, 0.0, 0.0])
        mesh = mesh.copy()
        mesh.apply_transform(rot)
    return mesh


def normalize_mesh(mesh: "trimesh.Trimesh", target_size: float) -> "trimesh.Trimesh":
    """Recenter (xz origin, base at y=0) and scale max edge to ``target_size``."""
    mesh = mesh.copy()
    bounds = mesh.bounds
    if bounds is None or not np.isfinite(bounds).all():
        raise ValueError("mesh has no finite bounds")
    center_xz = (bounds[0] + bounds[1]) / 2.0
    mesh.apply_translation([-center_xz[0], -bounds[0][1], -center_xz[2]])
    max_edge = float(max(mesh.extents))
    if max_edge <= 0:
        raise ValueError("degenerate mesh with zero extents")
    mesh.apply_scale(target_size / max_edge)
    return mesh


def enforce_min_thickness(
    mesh: "trimesh.Trimesh", min_thickness: float
) -> "trimesh.Trimesh":
    """Pad bounding-box axes thinner than ``min_thickness`` meters.

    Paper-thin meshes (low-poly keys, combs, ...) tunnel through the floor in
    the physics solver; scaling the thin axis up to ``min_thickness`` keeps
    contacts robust at the cost of a slightly thicker asset.
    """
    if min_thickness <= 0:
        return mesh
    factors = np.array(
        [min_thickness / e if 0 < e < min_thickness else 1.0 for e in mesh.extents]
    )
    if (factors == 1.0).all():
        return mesh
    mesh = mesh.copy()
    mesh.apply_transform(np.diag([*factors, 1.0]))
    return mesh


def decimate_mesh(mesh: "trimesh.Trimesh", max_faces: int) -> "trimesh.Trimesh":
    """Quadric-decimate to at most ``max_faces`` faces (drops UVs/textures)."""
    if len(mesh.faces) <= max_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=max_faces)
    except Exception as exc:  # noqa: BLE001 - fall back to the raw mesh
        logger.warning("decimation failed (%s); keeping %d faces", exc, len(mesh.faces))
        return mesh


def write_metadata(
    path: Path,
    mesh: "trimesh.Trimesh",
    source: Path,
    extra: dict | None = None,
) -> None:
    """Write a ``model_data<N>.json`` compatible with the object library."""
    bounds = mesh.bounds
    center = ((bounds[0] + bounds[1]) / 2.0).tolist()
    extents = mesh.extents.tolist()
    data = {
        "center": center,
        "extents": extents,
        "scale": [1.0, 1.0, 1.0],
        "stable": True,
        "source": source.name,
        "generator": "tools/import_generated_objects.py",
        "imported_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        data.update(extra)
    path.write_text(json.dumps(data, indent=4), encoding="utf-8")


def import_meshes(
    inputs: list[str],
    class_name: str,
    objects_dir: str | Path,
    target_size: float = 0.15,
    max_faces: int = 20000,
    z_up: bool = False,
    min_thickness: float = 0.0,
    provenance: dict | None = None,
) -> list[ImportedInstance]:
    """Import mesh files as instances of ``class_name`` into ``objects_dir``.

    Returns one :class:`ImportedInstance` per input mesh. Instance numbering
    appends after the class's existing instances. ``provenance`` (e.g.
    ``license`` / ``source_url`` / ``author``) is merged into every written
    ``model_data<N>.json`` so the asset manifest can audit per-asset IP.
    """
    objects_dir = Path(objects_dir)
    resolved = resolve_class_name(class_name, objects_dir)
    class_dir = objects_dir / resolved
    collision_dir = class_dir / "collision"
    collision_dir.mkdir(parents=True, exist_ok=True)

    files = collect_input_files(inputs)
    index = next_instance_index(class_dir)
    results: list[ImportedInstance] = []
    for path in files:
        warnings: list[str] = []
        mesh = load_mesh(path, z_up=z_up)
        faces_in = len(mesh.faces)
        if not mesh.is_watertight:
            warnings.append("mesh is not watertight (Genesis convexify will handle it)")
        mesh = normalize_mesh(mesh, target_size)
        thinned = min(mesh.extents) < min_thickness
        mesh = enforce_min_thickness(mesh, min_thickness)
        if thinned and min(mesh.extents) >= min_thickness:
            warnings.append(
                f"thin axis padded to {min_thickness:.3f} m (anti-tunneling)"
            )
        mesh = decimate_mesh(mesh, max_faces)
        if faces_in > max_faces and len(mesh.faces) > max_faces:
            warnings.append(f"decimation failed, still {len(mesh.faces)} faces")
        elif faces_in > max_faces:
            warnings.append("decimated; UVs/textures dropped")

        glb_path = collision_dir / f"base{index}.glb"
        mesh.export(str(glb_path))
        meta_path = class_dir / f"model_data{index}.json"
        write_metadata(meta_path, mesh, path, extra=provenance)

        results.append(
            ImportedInstance(
                class_name=resolved,
                index=index,
                glb_path=glb_path,
                metadata_path=meta_path,
                extents=tuple(float(e) for e in mesh.extents),  # type: ignore[arg-type]
                faces_in=faces_in,
                faces_out=len(mesh.faces),
                watertight=bool(mesh.is_watertight),
                warnings=warnings,
            )
        )
        index += 1
    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns 0 on success, 1 on input errors."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("inputs", nargs="+", help="mesh files or directories")
    parser.add_argument(
        "--class-name",
        required=True,
        help="target class (bare name gets an auto id; NNN_name to reuse/extend)",
    )
    parser.add_argument(
        "--objects-dir",
        default="assets/robotwin/objects/objects",
        help="RoboTwin object library root (default: %(default)s)",
    )
    parser.add_argument(
        "--target-size",
        type=float,
        default=0.15,
        help="largest bounding-box edge after normalization, meters (default: %(default)s)",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=20000,
        help="decimate meshes above this face count (default: %(default)s)",
    )
    parser.add_argument(
        "--z-up",
        action="store_true",
        help="inputs are Z-up (many OBJ exporters); rotate to glTF Y-up",
    )
    parser.add_argument(
        "--min-thickness",
        type=float,
        default=0.0,
        help="pad any bounding-box axis thinner than this, meters "
        "(anti-tunneling for paper-thin meshes; default: off)",
    )
    parser.add_argument(
        "--license",
        dest="license_",
        default=None,
        help="SPDX license id of the source meshes (e.g. CC0-1.0, CC-BY-4.0, MIT); "
        "recorded per instance for the asset manifest / IP audit",
    )
    parser.add_argument(
        "--source-url",
        default=None,
        help="URL or dataset id the meshes came from (e.g. an Objaverse uid)",
    )
    parser.add_argument(
        "--author",
        default=None,
        help="original author/creator of the source meshes",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    provenance = {
        k: v
        for k, v in {
            "license": args.license_,
            "source_url": args.source_url,
            "author": args.author,
        }.items()
        if v
    }
    if args.license_ is None:
        logger.warning(
            "no --license given; imported instances will be flagged as "
            "UNREGISTERED by tools/build_asset_manifest.py --check"
        )

    try:
        results = import_meshes(
            args.inputs,
            class_name=args.class_name,
            objects_dir=args.objects_dir,
            target_size=args.target_size,
            max_faces=args.max_faces,
            z_up=args.z_up,
            min_thickness=args.min_thickness,
            provenance=provenance or None,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    for r in results:
        warn = f" | warnings: {'; '.join(r.warnings)}" if r.warnings else ""
        print(
            f"[{r.class_name}] instance {r.index}: {r.glb_path} "
            f"({r.faces_in} -> {r.faces_out} faces, extents "
            f"{r.extents[0]:.3f} x {r.extents[1]:.3f} x {r.extents[2]:.3f} m){warn}"
        )
    print(f"imported {len(results)} instance(s) into {Path(args.objects_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
