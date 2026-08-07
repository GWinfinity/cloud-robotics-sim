"""Tests for the P1 parametric asset builder (tools/build_p1_assets.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloud_robotics_sim.robotwin.object_library import (  # noqa: E402
    RoboTwinObjectLibrary,
)
from tools.build_p1_assets import main  # noqa: E402

ALL_CLASSES = [
    "clothes_hanger",
    "mop_broom_set",
    "mop_holder",
    "underwear_grid_box",
    "fridge",
    "wardrobe",
]


def _cls(lib: RoboTwinObjectLibrary, suffix: str) -> str:
    """Resolve a class dir name by suffix (ids depend on existing classes)."""
    matches = [c for c in lib.list_classes() if c.endswith(f"_{suffix}")]
    assert len(matches) == 1, f"{suffix}: matches={matches}"
    return matches[0]


@pytest.fixture(scope="module")
def built_dir(tmp_path_factory) -> Path:
    """Build all six P1 classes once into a temp library."""
    objects_dir = tmp_path_factory.mktemp("objects")
    assert main(["--objects-dir", str(objects_dir)]) == 0
    return objects_dir


def test_all_classes_built_and_loadable(built_dir: Path) -> None:
    """Six classes (121-126) appear and resolve via the object library."""
    lib = RoboTwinObjectLibrary(built_dir)
    classes = lib.list_classes()
    assert len(classes) == 6
    assert [c[4:] for c in classes] == ALL_CLASSES  # build order -> ids in order
    for cls in classes:
        inst = lib.get_instance(cls)
        assert inst.asset_path.is_file(), cls


def test_rigid_class_geometry(built_dir: Path) -> None:
    """Extents are plausible real-world sizes; keep_scale flags are set."""
    lib = RoboTwinObjectLibrary(built_dir)
    hanger = lib.get_instance(_cls(lib, "clothes_hanger"), 0)
    assert 0.3 < max(hanger.extents) < 0.5
    assert hanger.metadata["keep_scale"] is True
    mop = lib.get_instance(_cls(lib, "mop_broom_set"), 0)
    assert 1.0 < max(mop.extents) < 1.6  # long handle keeps real scale
    assert mop.metadata["keep_scale"] is True
    bucket = lib.get_instance(_cls(lib, "mop_broom_set"), 2)
    assert 0.2 < max(bucket.extents) < 0.45
    assert "keep_scale" not in bucket.metadata


def test_provenance_registered(built_dir: Path) -> None:
    """Self-built assets carry Apache-2.0 license + generator metadata."""
    lib = RoboTwinObjectLibrary(built_dir)
    inst = lib.get_instance(_cls(lib, "underwear_grid_box"), 0)
    assert inst.metadata["license"] == "Apache-2.0"
    assert inst.metadata["generator"] == "tools/build_p1_assets.py"
    assert inst.metadata["author"]


def test_urdf_furniture(built_dir: Path) -> None:
    """Fridge/wardrobe use the PartNet-Mobility layout with revolute doors."""
    lib = RoboTwinObjectLibrary(built_dir)
    for suffix, doors in (("fridge", 1), ("wardrobe", 2)):
        cls = _cls(lib, suffix)
        assert lib.class_kind(cls) == "urdf"
        inst = lib.get_instance(cls)
        assert inst.kind == "urdf"
        assert inst.scale == (1.0, 1.0, 1.0)
        xml = inst.asset_path.read_text(encoding="utf-8")
        assert xml.count('type="revolute"') == doors
        # provenance lands in the instance model_data.json for the manifest
        meta = json.loads(
            (inst.asset_path.parent / "model_data.json").read_text("utf-8")
        )
        assert meta["license"] == "Apache-2.0"


def test_idempotent_skip_and_force(built_dir: Path) -> None:
    """Second run skips existing classes; --force rebuilds them."""
    assert main(["--objects-dir", str(built_dir)]) == 0  # skip path
    assert (
        main(["--objects-dir", str(built_dir), "--classes", "fridge", "--force"]) == 0
    )
    assert main(["--objects-dir", str(built_dir), "--classes", "nope"]) == 1
