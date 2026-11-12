"""Household cloth mesh generation and Genesis PBD spawning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HouseholdClothSpec:
    """Metric surface dimensions and PBD material parameters."""

    name: str
    width: float
    depth: float
    columns: int
    rows: int
    density: float = 4.0
    color: tuple[float, float, float, float] = (0.2, 0.5, 0.8, 1.0)


CLOTH_SPECS = {
    "rag": HouseholdClothSpec("rag", 0.18, 0.18, 13, 13),
    "towel": HouseholdClothSpec(
        "towel", 0.25, 0.50, 17, 33, color=(0.8, 0.4, 0.2, 1.0)
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


def export_cloth_grid(spec: HouseholdClothSpec, path: str | Path) -> Path:
    """Write an OBJ cloth surface, creating its parent directory."""
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
