"""Tests for household-physics parametric asset batches."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloud_robotics_sim.robotwin.object_library import (  # noqa: E402
    RoboTwinObjectLibrary,
)
from tools.build_home_physics_assets import (  # noqa: E402
    BUILDERS,
    GENERATOR,
    GLASS_WALL_THICKNESS_M,
    main,
)

RECIPE_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "recipes"
    / "home_physics_assets.yaml"
)


def _cls(lib: RoboTwinObjectLibrary, suffix: str) -> str:
    matches = [c for c in lib.list_classes() if c.endswith(f"_{suffix}")]
    assert len(matches) == 1, f"{suffix}: matches={matches}"
    return matches[0]


@pytest.fixture(scope="module")
def built_dir(tmp_path_factory) -> Path:
    """Build all eight classes once into an isolated object library."""
    objects_dir = tmp_path_factory.mktemp("home_physics_objects")
    assert main(["--objects-dir", str(objects_dir)]) == 0
    return objects_dir


def test_batch_matches_recipe_and_loads(built_dir: Path) -> None:
    """The declared batches resolve through the production library loader."""
    recipe = yaml.safe_load(RECIPE_PATH.read_text(encoding="utf-8"))
    declared = {item for batch in recipe["build_batches"] for item in batch["items"]}
    assert set(BUILDERS) == declared
    lib = RoboTwinObjectLibrary(built_dir)
    assert len(lib.list_classes()) == len(BUILDERS)
    for name in BUILDERS:
        inst = lib.get_instance(_cls(lib, name))
        assert inst.asset_path.is_file()
        assert inst.metadata["generator"] == GENERATOR
        assert inst.metadata["license"] == "Apache-2.0"
        assert inst.metadata["keep_scale"] is True


def test_second_batch_has_handle_and_open_basket(built_dir: Path) -> None:
    """The board has a compound handle and the basket is genuinely open gridwork."""
    lib = RoboTwinObjectLibrary(built_dir)
    board = trimesh.load(
        lib.get_instance(_cls(lib, "cutting_board")).asset_path,
        force="mesh",
        process=False,
    )
    basket = trimesh.load(
        lib.get_instance(_cls(lib, "laundry_basket")).asset_path,
        force="mesh",
        process=False,
    )
    assert len(board.split(only_watertight=False)) >= 8
    assert board.volume / np.prod(board.extents) > 0.3
    assert len(basket.split(only_watertight=False)) >= 30
    assert basket.volume / np.prod(basket.extents) < 0.02


def test_exported_geometry_is_closed_and_matches_recipe_dims(built_dir: Path) -> None:
    """Delivered GLBs retain closed component shells and exact metric sizes."""
    recipe = yaml.safe_load(RECIPE_PATH.read_text(encoding="utf-8"))
    items = {item["id"]: item for item in recipe["items"]}
    lib = RoboTwinObjectLibrary(built_dir)
    for name in BUILDERS:
        mesh = trimesh.load(
            lib.get_instance(_cls(lib, name)).asset_path,
            force="mesh",
            process=False,
        )
        assert np.isfinite(mesh.vertices).all(), name
        assert mesh.is_watertight, name
        assert mesh.is_winding_consistent, name
        assert all(part.is_watertight for part in mesh.split(only_watertight=False))
        assert mesh.volume > 0.0, name
        assert mesh.bounds[0][1] == pytest.approx(0.0, abs=1e-7), name
        dims_xyz = items[name]["dims_m"]
        expected_y_up = [dims_xyz[0], dims_xyz[2], dims_xyz[1]]
        assert np.allclose(mesh.extents, expected_y_up, atol=1e-6), (
            name,
            mesh.extents,
            expected_y_up,
        )


def test_glass_cup_has_real_two_mm_hollow_wall(built_dir: Path) -> None:
    """A mid-height section measures both sides of the hollow glass wall."""
    lib = RoboTwinObjectLibrary(built_dir)
    inst = lib.get_instance(_cls(lib, "glass_cup"))
    mesh = trimesh.load(inst.asset_path, force="mesh", process=False)
    section = mesh.section(plane_origin=[0.0, 0.10, 0.0], plane_normal=[0.0, 1.0, 0.0])
    assert section is not None
    radii = np.linalg.norm(section.vertices[:, [0, 2]], axis=1)
    assert radii.min() == pytest.approx(0.033, abs=1e-6)
    assert radii.max() == pytest.approx(0.035, abs=1e-6)
    assert radii.max() - radii.min() == pytest.approx(GLASS_WALL_THICKNESS_M, abs=1e-6)
    assert inst.metadata["physics_kind"] == "fragile_rigid"
    assert inst.metadata["wall_thickness_m"] == GLASS_WALL_THICKNESS_M


def test_idempotent_and_unknown_class(built_dir: Path) -> None:
    """Repeated builds reuse class ids; unknown names fail clearly."""
    lib = RoboTwinObjectLibrary(built_dir)
    before = lib.list_classes()
    assert main(["--objects-dir", str(built_dir)]) == 0
    assert lib.list_classes() == before
    assert main(["--objects-dir", str(built_dir), "--classes", "nope"]) == 1
