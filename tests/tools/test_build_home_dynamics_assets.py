"""Integration tests for generated household fracture and cloth assets."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloud_robotics_sim.robotwin.object_library import (  # noqa: E402
    RoboTwinObjectLibrary,
)
from tests.optional_deps import fracture_only  # noqa: E402
from tools.build_home_dynamics_assets import GENERATOR, main  # noqa: E402


def test_cloth_assets_build_and_reload(tmp_path: Path) -> None:
    """Generated rag and towel OBJ files retain their metric dimensions."""
    assert main(["--output-dir", str(tmp_path), "--classes", "rag", "towel"]) == 0
    rag = trimesh.load(tmp_path / "cloth" / "rag.obj", process=False)
    towel = trimesh.load(tmp_path / "cloth" / "towel.obj", process=False)
    assert len(rag.vertices) == 169
    assert len(towel.vertices) == 561
    assert rag.extents[:2] == pytest.approx([0.18, 0.18])
    assert towel.extents[:2] == pytest.approx([0.25, 0.50])
    before = (tmp_path / "cloth" / "rag.obj").stat().st_mtime_ns
    assert main(["--output-dir", str(tmp_path), "--classes", "rag"]) == 0
    assert (tmp_path / "cloth" / "rag.obj").stat().st_mtime_ns == before


@fracture_only
def test_voronoi_fragments_build_reload_and_conserve_volume(tmp_path: Path) -> None:
    """All 12 delivered GLB fragments are watertight and conserve cup volume."""
    assert (
        main(["--output-dir", str(tmp_path), "--classes", "glass_cup_fragments"]) == 0
    )
    lib = RoboTwinObjectLibrary(tmp_path / "objects")
    class_name = lib.list_classes()[0]
    assert class_name.endswith("_glass_cup_fragments")
    assert lib.instance_count(class_name) == 12
    instances = [lib.get_instance(class_name, i) for i in range(12)]
    assert all(inst.metadata["generator"] == GENERATOR for inst in instances)
    assert all(inst.metadata["license"] == "Apache-2.0" for inst in instances)
    fragment_volume = sum(inst.metadata["fragment_volume_m3"] for inst in instances)
    parent_volume = instances[0].metadata["parent_volume_m3"]
    assert fragment_volume == pytest.approx(parent_volume, rel=1e-5)
    for inst in instances:
        mesh = trimesh.load(inst.asset_path, force="mesh", process=False)
        assert mesh.is_watertight, inst.asset_path
        assert mesh.volume > 0.0, inst.asset_path
    assert (
        main(["--output-dir", str(tmp_path), "--classes", "glass_cup_fragments"]) == 0
    )
    assert lib.list_classes() == [class_name]


def test_unknown_asset_fails(tmp_path: Path) -> None:
    """Unknown asset names return a nonzero status without writing files."""
    assert main(["--output-dir", str(tmp_path), "--classes", "nope"]) == 1
    assert list(tmp_path.iterdir()) == []
