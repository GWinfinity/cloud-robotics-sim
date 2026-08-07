"""Tests for the P3 parametric asset builder (tools/build_p3_assets.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloud_robotics_sim.robotwin.object_library import (  # noqa: E402
    RoboTwinObjectLibrary,
)
from tools.build_p3_assets import RIGID_BUILDERS, URDF_BUILDERS, main  # noqa: E402


def _cls(lib: RoboTwinObjectLibrary, suffix: str) -> str:
    """Resolve a class dir name by suffix (ids depend on existing classes)."""
    matches = [c for c in lib.list_classes() if c.endswith(f"_{suffix}")]
    assert len(matches) == 1, f"{suffix}: matches={matches}"
    return matches[0]


@pytest.fixture(scope="module")
def built_dir(tmp_path_factory) -> Path:
    """Build all 20 P3 classes once into a temp library."""
    objects_dir = tmp_path_factory.mktemp("objects")
    assert main(["--objects-dir", str(objects_dir)]) == 0
    return objects_dir


def test_all_classes_built_and_loadable(built_dir: Path) -> None:
    """14 rigid + 6 urdf classes appear and resolve via the object library."""
    lib = RoboTwinObjectLibrary(built_dir)
    classes = lib.list_classes()
    assert len(classes) == 20
    assert [c[4:] for c in classes] == sorted(RIGID_BUILDERS) + sorted(URDF_BUILDERS)
    for cls in classes:
        assert lib.get_instance(cls).asset_path.is_file(), cls


def test_companion_instances(built_dir: Path) -> None:
    """Set classes bundle their contents as extra instances."""
    lib = RoboTwinObjectLibrary(built_dir)
    assert lib.instance_count(_cls(lib, "kitchen_utensil_set")) == 3
    assert lib.instance_count(_cls(lib, "hairdryer_set")) == 2
    assert lib.instance_count(_cls(lib, "toothbrush_set")) == 2
    assert lib.instance_count(_cls(lib, "umbrella_set")) == 2


def test_rigid_geometry_and_scale_flags(built_dir: Path) -> None:
    """Extents are plausible; tall items carry keep_scale."""
    lib = RoboTwinObjectLibrary(built_dir)
    shelf = lib.get_instance(_cls(lib, "over_toilet_shelf"), 0)
    assert 1.2 < max(shelf.extents) < 1.8
    assert shelf.metadata["keep_scale"] is True
    peg = lib.get_instance(_cls(lib, "pegboard"), 0)
    assert 0.3 < max(peg.extents) < 0.5
    tube = lib.get_instance(_cls(lib, "tube_organizer"), 0)
    assert 0.1 < max(tube.extents) < 0.2
    assert "keep_scale" not in tube.metadata


def test_provenance_registered(built_dir: Path) -> None:
    """P3 assets carry Apache-2.0 license and the p3 generator label."""
    lib = RoboTwinObjectLibrary(built_dir)
    inst = lib.get_instance(_cls(lib, "bookend_stand"), 0)
    assert inst.metadata["license"] == "Apache-2.0"
    assert inst.metadata["generator"] == "tools/build_p3_assets.py"


def test_urdf_joint_types(built_dir: Path) -> None:
    """Rotating trays/cart use continuous joints; lids revolute; drawer prismatic."""
    lib = RoboTwinObjectLibrary(built_dir)
    tray = lib.get_instance(_cls(lib, "rotating_spice_tray"))
    tray_xml = tray.asset_path.read_text(encoding="utf-8")
    assert tray_xml.count('type="continuous"') == 1
    ottoman = lib.get_instance(_cls(lib, "storage_ottoman"))
    assert ottoman.asset_path.read_text(encoding="utf-8").count('type="revolute"') == 1
    side = lib.get_instance(_cls(lib, "side_table_cabinet"))
    assert side.asset_path.read_text(encoding="utf-8").count('type="prismatic"') == 1
    cart = lib.get_instance(_cls(lib, "desk_cart"))
    cart_xml = cart.asset_path.read_text(encoding="utf-8")
    assert cart_xml.count('type="continuous"') == 4
    meta = json.loads((cart.asset_path.parent / "model_data.json").read_text("utf-8"))
    assert meta["license"] == "Apache-2.0"
    assert meta["generator"] == "tools/build_p3_assets.py"


def test_idempotent_and_unknown_class(built_dir: Path) -> None:
    """Second run skips; unknown class errors out."""
    assert main(["--objects-dir", str(built_dir)]) == 0
    assert main(["--objects-dir", str(built_dir), "--classes", "nope"]) == 1
