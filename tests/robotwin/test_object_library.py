"""Unit tests for ``cloud_robotics_sim.robotwin.object_library`` (no GPU needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cloud_robotics_sim.robotwin.object_library import (
    ObjectInstance,
    RoboTwinObjectLibrary,
    normalize_scale,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_OBJECTS_DIR = REPO_ROOT / "assets" / "robotwin" / "objects" / "objects"


def _make_class(root: Path, name: str, n_instances: int, with_glb: bool = True) -> Path:
    class_dir = root / name
    (class_dir / "collision").mkdir(parents=True)
    for i in range(n_instances):
        (class_dir / f"model_data{i}.json").write_text(
            json.dumps(
                {
                    "center": [0.0, 0.1 * i, 0.0],
                    "extents": [0.1, 0.2, 0.3],
                    "scale": [0.5, 0.5, 0.5],
                }
            ),
            encoding="utf-8",
        )
        if with_glb:
            (class_dir / "collision" / f"base{i}.glb").write_bytes(b"glb")
    return class_dir


@pytest.fixture()
def fake_library(tmp_path: Path) -> RoboTwinObjectLibrary:
    """Build a fake two-class object library in a temp dir."""
    _make_class(tmp_path, "002_bowl", 2)
    _make_class(tmp_path, "001_bottle", 3)
    (tmp_path / "readme.txt").write_text("not a class", encoding="utf-8")
    (tmp_path / "objaverse").mkdir()  # not matching NNN_ pattern
    return RoboTwinObjectLibrary(tmp_path)


def test_list_classes_sorted_and_filtered(fake_library: RoboTwinObjectLibrary) -> None:
    """Class listing is sorted and non-class dirs are skipped."""
    assert fake_library.list_classes() == ["001_bottle", "002_bowl"]


def test_instance_count(fake_library: RoboTwinObjectLibrary) -> None:
    """Instance count matches the number of model_data files."""
    assert fake_library.instance_count("001_bottle") == 3
    assert fake_library.instance_count("002_bowl") == 2


def test_get_instance_fields(fake_library: RoboTwinObjectLibrary) -> None:
    """Instance fields are parsed from model_data + glb path."""
    inst = fake_library.get_instance("001_bottle", 1)
    assert isinstance(inst, ObjectInstance)
    assert inst.class_name == "001_bottle"
    assert inst.index == 1
    assert inst.asset_path.name == "base1.glb"
    assert inst.scale == (0.5, 0.5, 0.5)
    assert inst.extents == (0.1, 0.2, 0.3)
    assert inst.kind == "glb"
    assert inst.scaled_extents == pytest.approx((0.05, 0.1, 0.15))


def test_get_instance_errors(fake_library: RoboTwinObjectLibrary) -> None:
    """Missing class and bad index raise errors."""
    with pytest.raises(FileNotFoundError):
        fake_library.get_instance("999_missing")
    with pytest.raises(IndexError):
        fake_library.get_instance("001_bottle", 99)


def test_first_instance_nonzero_start(tmp_path: Path) -> None:
    """Some classes start numbering at 1 (like the real 002_bowl)."""
    class_dir = tmp_path / "002_bowl"
    (class_dir / "collision").mkdir(parents=True)
    for i in (1, 2):
        (class_dir / f"model_data{i}.json").write_text(
            json.dumps({"center": [0, 0, 0], "extents": [1, 1, 1], "scale": [1, 1, 1]}),
            encoding="utf-8",
        )
        (class_dir / "collision" / f"base{i}.glb").write_bytes(b"glb")
    lib = RoboTwinObjectLibrary(tmp_path)
    assert lib.instance_indices("002_bowl") == [1, 2]
    inst = lib.get_instance("002_bowl")
    assert inst.index == 1
    assert inst.asset_path.name == "base1.glb"


def test_urdf_class(tmp_path: Path) -> None:
    """PartNet-Mobility layout: numbered subdirs with mobility.urdf."""
    class_dir = tmp_path / "009_kettle"
    for sub in ("102730", "102738"):
        d = class_dir / sub
        d.mkdir(parents=True)
        (d / "mobility.urdf").write_text("<robot name='x'/>", encoding="utf-8")
    (class_dir / "102730" / "model_data.json").write_text(
        json.dumps({"scale": [0.2]}), encoding="utf-8"
    )
    lib = RoboTwinObjectLibrary(tmp_path)
    assert lib.class_kind("009_kettle") == "urdf"
    inst = lib.get_instance("009_kettle")
    assert inst.kind == "urdf"
    assert inst.asset_path.name == "mobility.urdf"
    assert "102730" in str(inst.asset_path)
    assert inst.scale == (0.2, 0.2, 0.2)


def test_missing_glb(tmp_path: Path) -> None:
    """Missing collision mesh raises FileNotFoundError."""
    _make_class(tmp_path, "003_plate", 1, with_glb=False)
    lib = RoboTwinObjectLibrary(tmp_path)
    with pytest.raises(FileNotFoundError):
        lib.get_instance("003_plate", 0)


def test_missing_dir(tmp_path: Path) -> None:
    """Missing objects dir raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        RoboTwinObjectLibrary(tmp_path / "nope")


def test_normalize_scale() -> None:
    """Oversized meshes are shrunk, normal ones unchanged."""
    # normal object: unchanged
    assert normalize_scale(1.9, (0.132, 0.132, 0.132)) == (0.132, 0.132, 0.132)
    # missing metadata: 1.93 m raw mesh shrunk to 0.20 m target height
    out = normalize_scale(1.928, (1.0, 1.0, 1.0))
    assert out[1] * 1.928 == pytest.approx(0.20)
    assert out[0] == out[1] == out[2]
    # degenerate raw height: unchanged
    assert normalize_scale(0.0, (1.0, 1.0, 1.0)) == (1.0, 1.0, 1.0)


@pytest.mark.skipif(not REAL_OBJECTS_DIR.is_dir(), reason="real assets not present")
def test_real_assets_first_instances() -> None:
    """Smoke test against the real RoboTwin object database."""
    lib = RoboTwinObjectLibrary(REAL_OBJECTS_DIR)
    classes = lib.list_classes()
    assert len(classes) > 100
    for class_name in classes:
        inst = lib.get_instance(class_name)
        assert inst.asset_path.is_file(), class_name
        assert all(s > 0 for s in inst.scale), class_name
