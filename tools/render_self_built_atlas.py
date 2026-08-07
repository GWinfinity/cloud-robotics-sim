#!/usr/bin/env python3
"""Render a preview atlas of the self-built asset classes (121+).

Draws every self-built class (GLB instances + URDF furniture, parsed from
the PartNet-Mobility layout) into a labeled matplotlib grid — no GPU or
display needed. For eyeballing the parametric geometry quality.

Usage::

    uv run python tools/render_self_built_atlas.py
    uv run python tools/render_self_built_atlas.py --min-id 164 --out outputs/atlas_contents.png
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_glb_class_meshes(class_dir: Path, max_instances: int = 2) -> list:
    """Load up to ``max_instances`` collision meshes of a GLB class."""
    import trimesh

    meshes = []
    for glb in sorted(class_dir.glob("collision/base*.glb"))[:max_instances]:
        loaded = trimesh.load(str(glb), force="mesh")
        meshes.append(loaded)
    return meshes


def _box_faces(center, size) -> list:
    """Axis-aligned box as 6 quad faces (for URDF primitive rendering)."""
    cx, cy, cz = center
    sx, sy, sz = (v / 2 for v in size)
    v = np.array(
        [
            [cx - sx, cy - sy, cz - sz],
            [cx + sx, cy - sy, cz - sz],
            [cx + sx, cy + sy, cz - sz],
            [cx - sx, cy + sy, cz - sz],
            [cx - sx, cy - sy, cz + sz],
            [cx + sx, cy - sy, cz + sz],
            [cx + sx, cy + sy, cz + sz],
            [cx - sx, cy + sy, cz + sz],
        ]
    )
    quads = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (1, 2, 6, 5),
        (0, 3, 7, 4),
    ]
    return [[v[i] for i in q] for q in quads]


def _cylinder_faces(center, radius, length, axis="z", sections=24) -> list:
    """Vertical (z) or x-aligned cylinder as a quad strip + caps."""
    cx, cy, cz = center
    angles = np.linspace(0, 2 * math.pi, sections, endpoint=False)
    faces = []
    for i in range(sections):
        j = (i + 1) % sections
        a, b = angles[i], angles[j]
        ring_lo = [
            (cx + radius * math.cos(a), cy + radius * math.sin(a), cz - length / 2),
            (cx + radius * math.cos(b), cy + radius * math.sin(b), cz - length / 2),
        ]
        ring_hi = [(x, y, z + length) for x, y, z in ring_lo]
        quad = [ring_lo[0], ring_lo[1], ring_hi[1], ring_hi[0]]
        if axis == "x":  # rotate z->x: (x,y,z) -> (z,y,x) about origin then translate
            quad = [(cz + (z - cz), y, cx + (x - cx)) for x, y, z in quad]
        faces.append(quad)
    return faces


def _load_urdf_class_faces(class_dir: Path) -> list:
    """Parse mobility.urdf primitives into a flat face list (closed state)."""
    inst_dirs = [
        d for d in class_dir.iterdir() if d.is_dir() and (d / "mobility.urdf").is_file()
    ]
    root = ET.parse(inst_dirs[0] / "mobility.urdf").getroot()
    joint_offset: dict[str, np.ndarray] = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
        joint_offset[joint.find("child").get("link")] = np.array(
            [float(v) for v in xyz.split()]
        )
    faces = []
    for link in root.findall("link"):
        name = link.get("name")
        base = joint_offset.get(name, np.zeros(3))
        geom_parent = link.find("visual") or link.find("collision")
        origin = geom_parent.find("origin")
        xyz = (
            np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
            if origin is not None
            else np.zeros(3)
        )
        geom = geom_parent.find("geometry")
        center = base + xyz
        box = geom.find("box")
        cyl = geom.find("cylinder")
        if box is not None:
            size = [float(v) for v in box.get("size").split()]
            faces.extend(_box_faces(center, size))
        elif cyl is not None:
            axis = "z"
            rpy = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
            if abs(float(rpy.split()[1])) > 0.1:
                axis = "x"
            faces.extend(
                _cylinder_faces(
                    center,
                    float(cyl.get("radius")),
                    float(cyl.get("length")),
                    axis=axis,
                )
            )
    return faces


def _draw_trimesh(ax, mesh, offset=(0.0, 0.0, 0.0), color="#8ab4f8") -> tuple:
    """Draw a trimesh mesh (Y-up) as Poly3DCollection; return bounds."""
    verts = mesh.vertices + np.asarray(offset)
    polys = verts[mesh.faces]
    ax.add_collection3d(
        Poly3DCollection(
            polys, facecolor=color, edgecolor="#33415c", linewidths=0.15, alpha=0.95
        )
    )
    return verts.min(axis=0), verts.max(axis=0)


def _draw_faces(ax, faces, offset=(0.0, 0.0, 0.0), color="#f6bd60") -> tuple:
    """Draw raw face lists (URDF primitives, Z-up -> map to Y-up for display)."""
    off = np.asarray(offset)
    polys = [[np.asarray(v) + off for v in face] for face in faces]
    ax.add_collection3d(
        Poly3DCollection(
            polys, facecolor=color, edgecolor="#5c4333", linewidths=0.2, alpha=0.95
        )
    )
    all_v = np.vstack([np.asarray(v) + off for face in faces for v in face])
    return all_v.min(axis=0), all_v.max(axis=0)


def _frame(ax, lo, hi, title: str) -> None:
    """Equal-ish framing for one cell."""
    center = (lo + hi) / 2
    span = float(max(hi - lo).max() if hasattr(hi - lo, "max") else (hi - lo)) or 0.1
    r = span * 0.62 + 0.02
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r, center[2] + r)
    ax.set_title(title, fontsize=7, pad=1)
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-55)


def main() -> int:
    """Render the atlas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-dir", default="assets/robotwin/objects/objects")
    parser.add_argument(
        "--min-id", type=int, default=121, help="first self-built class id"
    )
    parser.add_argument("--out", default="outputs/asset_preview/self_built_atlas.png")
    args = parser.parse_args()

    objects_dir = Path(args.objects_dir)
    class_dirs = sorted(
        d
        for d in objects_dir.iterdir()
        if d.is_dir() and d.name[:3].isdigit() and int(d.name[:3]) >= args.min_id
    )
    if not class_dirs:
        print("no self-built classes found")
        return 1

    cols = 8
    rows = math.ceil(len(class_dirs) / cols)
    fig = plt.figure(figsize=(cols * 2.1, rows * 2.1), dpi=110)
    for i, class_dir in enumerate(class_dirs):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        is_urdf = not list(class_dir.glob("model_data*.json"))
        los, his = [], []
        if is_urdf:
            lo, hi = _draw_faces(ax, _load_urdf_class_faces(class_dir))
            los.append(lo)
            his.append(hi)
        else:
            meshes = _load_glb_class_meshes(class_dir)
            width = sum(float(m.extents[0]) for m in meshes) + 0.06 * (len(meshes) - 1)
            x = -width / 2
            for m in meshes:
                w = float(m.extents[0])
                cx = x + w / 2
                lo, hi = _draw_trimesh(ax, m, offset=(cx, 0.0, 0.0))
                los.append(lo)
                his.append(hi)
                x += w + 0.06
        lo = np.min(np.array(los), axis=0)
        hi = np.max(np.array(his), axis=0)
        _frame(ax, lo, hi, class_dir.name)
    fig.suptitle(
        f"Self-built parametric assets ({len(class_dirs)} classes, ids >= {args.min_id})",
        fontsize=11,
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"atlas -> {out} ({len(class_dirs)} classes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
