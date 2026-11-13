"""Household cloth mesh generation and Genesis PBD spawning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HouseholdClothSpec:
    """Metric surface dimensions, construction and PBD material parameters."""

    name: str
    width: float
    depth: float
    columns: int
    rows: int
    density: float = 4.0
    color: tuple[float, float, float, float] = (0.2, 0.5, 0.8, 1.0)
    shape: str = "grid"


CLOTH_SPECS = {
    "rag": HouseholdClothSpec("rag", 0.18, 0.18, 13, 13),
    "towel": HouseholdClothSpec(
        "towel", 0.25, 0.50, 17, 33, color=(0.8, 0.4, 0.2, 1.0)
    ),
    "bedsheet": HouseholdClothSpec(
        "bedsheet", 1.60, 2.20, 33, 45, color=(0.85, 0.88, 0.9, 1.0)
    ),
    "quilt": HouseholdClothSpec(
        "quilt", 1.50, 2.00, 31, 41, density=8.0, color=(0.65, 0.3, 0.45, 1.0)
    ),
    "blanket": HouseholdClothSpec(
        "blanket", 1.30, 1.80, 27, 37, density=5.0, color=(0.25, 0.55, 0.35, 1.0)
    ),
    "garment": HouseholdClothSpec(
        "garment",
        0.60,
        0.70,
        25,
        29,
        density=3.0,
        color=(0.25, 0.35, 0.75, 1.0),
        shape="tshirt",
    ),
    "garbage_bag": HouseholdClothSpec(
        "garbage_bag",
        0.50,
        0.60,
        21,
        25,
        density=1.0,
        color=(0.08, 0.08, 0.08, 1.0),
        shape="open_bag",
    ),
    "plastic_bag": HouseholdClothSpec(
        "plastic_bag",
        0.38,
        0.45,
        19,
        23,
        density=0.8,
        color=(0.85, 0.85, 0.78, 1.0),
        shape="open_bag",
    ),
    "pillow": HouseholdClothSpec(
        "pillow",
        0.60,
        0.40,
        25,
        21,
        density=6.0,
        color=(0.82, 0.78, 0.68, 1.0),
        shape="pillow",
    ),
}


def cloth_grid(spec: HouseholdClothSpec):
    """Create a Z-up XY triangle surface with no solid-shell approximation."""
    import trimesh

    if min(spec.width, spec.depth) <= 0.0:
        raise ValueError("cloth dimensions must be positive")
    if spec.columns < 2 or spec.rows < 2:
        raise ValueError("cloth grid needs at least two rows and columns")
    x = np.linspace(-spec.width / 2.0, spec.width / 2.0, spec.columns)
    y = np.linspace(-spec.depth / 2.0, spec.depth / 2.0, spec.rows)
    vertices = np.column_stack(
        [
            np.tile(x, spec.rows),
            np.repeat(y, spec.columns),
            np.zeros(spec.rows * spec.columns),
        ]
    )
    faces: list[list[int]] = []
    for row in range(spec.rows - 1):
        for column in range(spec.columns - 1):
            a = row * spec.columns + column
            b = a + 1
            c = a + spec.columns
            d = c + 1
            faces.extend(([a, b, d], [a, d, c]))
    return trimesh.Trimesh(
        vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False
    )


def _tshirt_mesh(spec: HouseholdClothSpec):
    """Cut a connected T-shirt outline with sleeves and a neck opening."""
    mesh = cloth_grid(spec)
    centers = mesh.triangles_center
    u = centers[:, 0] / (spec.width / 2.0)
    v = centers[:, 1] / (spec.depth / 2.0)
    body = np.abs(u) <= 0.62
    sleeve_top = 0.82 - 0.35 * np.maximum(np.abs(u) - 0.62, 0.0) / 0.38
    sleeves = (np.abs(u) > 0.55) & (v >= 0.12) & (v <= sleeve_top)
    neck = (u / 0.22) ** 2 + ((v - 0.98) / 0.22) ** 2 < 1.0
    mesh.update_faces((body | sleeves) & ~neck)
    mesh.remove_unreferenced_vertices()
    return mesh


def _panel_shell_mesh(spec: HouseholdClothSpec, *, close_top: bool):
    """Join two bulging panels along their seams, optionally closing the top."""
    import trimesh

    panel = cloth_grid(spec)
    count = len(panel.vertices)
    u = (panel.vertices[:, 0] + spec.width / 2.0) / spec.width
    v = (panel.vertices[:, 1] + spec.depth / 2.0) / spec.depth
    bulge_ratio = 0.20 if close_top else 0.12
    bulge = (
        min(spec.width, spec.depth)
        * bulge_ratio
        * np.sin(np.pi * u)
        * np.sin(np.pi * v)
    )
    front = panel.vertices.copy()
    back = panel.vertices.copy()
    front[:, 2] = bulge
    back[:, 2] = -bulge
    faces = panel.faces.tolist()
    faces.extend((panel.faces[:, ::-1] + count).tolist())

    def bridge(a: int, b: int) -> None:
        faces.extend(([b, a, a + count], [b, a + count, b + count]))

    for column in range(spec.columns - 1):
        bridge(column, column + 1)
        if close_top:
            top_left = (spec.rows - 1) * spec.columns
            bridge(top_left + column + 1, top_left + column)
    for row in range(spec.rows - 1):
        lower_left = row * spec.columns
        upper_left = lower_left + spec.columns
        bridge(upper_left, lower_left)
        lower_right = lower_left + spec.columns - 1
        upper_right = upper_left + spec.columns - 1
        bridge(lower_right, upper_right)
    return trimesh.Trimesh(
        vertices=np.vstack([front, back]),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )


def _open_bag_mesh(spec: HouseholdClothSpec):
    """Join two bulging panels at the sides and bottom, leaving the mouth open."""
    return _panel_shell_mesh(spec, close_top=False)


def _pillow_mesh(spec: HouseholdClothSpec):
    """Create a closed pillow shell; internal pressure is not modeled."""
    return _panel_shell_mesh(spec, close_top=True)


def cloth_mesh(spec: HouseholdClothSpec):
    """Create the selected flat, garment, bag or pillow cloth surface."""
    builders = {
        "grid": cloth_grid,
        "tshirt": _tshirt_mesh,
        "open_bag": _open_bag_mesh,
        "pillow": _pillow_mesh,
    }
    try:
        builder = builders[spec.shape]
    except KeyError as exc:
        raise ValueError(f"unknown cloth shape: {spec.shape}") from exc
    return builder(spec)


def export_cloth_mesh(spec: HouseholdClothSpec, path: str | Path) -> Path:
    """Write an OBJ cloth surface, creating its parent directory."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cloth_mesh(spec).export(output)
    return output


def export_cloth_grid(spec: HouseholdClothSpec, path: str | Path) -> Path:
    """Write a rectangular OBJ cloth grid, creating its parent directory."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cloth_grid(spec).export(output)
    return output


def spawn_pbd_cloth(
    scene: Any,
    spec: HouseholdClothSpec,
    file: str | Path,
    pos: tuple[float, float, float],
):
    """Spawn one sheet through Genesis's native PBD cloth solver."""
    import genesis as gs

    return scene.add_entity(
        material=gs.materials.PBD.Cloth(
            rho=spec.density,
            static_friction=0.4,
            kinetic_friction=0.3,
            stretch_compliance=1e-7,
            bending_compliance=1e-5,
        ),
        morph=gs.morphs.Mesh(
            file=str(file),
            file_meshes_are_zup=True,
            pos=pos,
        ),
        surface=gs.surfaces.Default(color=spec.color, double_sided=True),
    )


def particle_positions(entity: Any) -> np.ndarray:
    """Return single-environment PBD particle positions as a host array."""
    positions = entity.get_particles_pos()
    if hasattr(positions, "detach"):
        positions = positions.detach().cpu().numpy()
    return np.asarray(positions).reshape(-1, 3)
