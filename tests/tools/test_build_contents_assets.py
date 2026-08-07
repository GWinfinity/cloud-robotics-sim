"""Tests for the contents asset builder (tools/build_contents_assets.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloud_robotics_sim.robotwin.object_library import (  # noqa: E402
    RoboTwinObjectLibrary,
)
from tools.build_contents_assets import BUILDERS, main  # noqa: E402


def _cls(lib: RoboTwinObjectLibrary, suffix: str) -> str:
    """Resolve a class dir name by suffix (ids depend on existing classes)."""
    matches = [c for c in lib.list_classes() if c.endswith(f"_{suffix}")]
    assert len(matches) == 1, f"{suffix}: matches={matches}"
    return matches[0]


@pytest.fixture(scope="module")
def built_dir(tmp_path_factory) -> Path:
    """Build all 10 contents classes once into a temp library."""
    objects_dir = tmp_path_factory.mktemp("objects")
    assert main(["--objects-dir", str(objects_dir)]) == 0
    return objects_dir


def test_all_classes_built_and_loadable(built_dir: Path) -> None:
    """10 rigid contents classes appear and resolve via the object library."""
    lib = RoboTwinObjectLibrary(built_dir)
    classes = lib.list_classes()
    assert len(classes) == 10
    assert [c[4:] for c in classes] == sorted(BUILDERS)
    for cls in classes:
        assert lib.get_instance(cls).asset_path.is_file(), cls


def test_instance_counts(built_dir: Path) -> None:
    """Variants bundle as instances (pairs / sets / cartons)."""
    lib = RoboTwinObjectLibrary(built_dir)
    assert lib.instance_count(_cls(lib, "chopsticks")) == 2
    assert lib.instance_count(_cls(lib, "spoon")) == 2
    assert lib.instance_count(_cls(lib, "pot_lid")) == 2
    assert lib.instance_count(_cls(lib, "egg_set")) == 3  # 2 eggs + carton
    assert lib.instance_count(_cls(lib, "glasses")) == 1


def test_geometry_plausible(built_dir: Path) -> None:
    """Sizes are real-world plausible for daily items."""
    lib = RoboTwinObjectLibrary(built_dir)
    egg = lib.get_instance(_cls(lib, "egg_set"), 0)
    assert 0.03 < max(egg.extents) < 0.08  # ~5-6 cm long axis
    chop = lib.get_instance(_cls(lib, "chopsticks"), 0)
    assert 0.20 < max(chop.extents) < 0.30
    glasses = lib.get_instance(_cls(lib, "glasses"), 0)
    assert 0.10 < max(glasses.extents) < 0.16
    scissors = lib.get_instance(_cls(lib, "scissors"), 0)
    assert 0.15 < max(scissors.extents) < 0.25


def test_provenance_registered(built_dir: Path) -> None:
    """Contents assets carry Apache-2.0 license and the generator label."""
    lib = RoboTwinObjectLibrary(built_dir)
    inst = lib.get_instance(_cls(lib, "key_ring"), 0)
    assert inst.metadata["license"] == "Apache-2.0"
    assert inst.metadata["generator"] == "tools/build_contents_assets.py"


def test_idempotent_and_unknown_class(built_dir: Path) -> None:
    """Second run skips; unknown class errors out."""
    assert main(["--objects-dir", str(built_dir)]) == 0
    assert main(["--objects-dir", str(built_dir), "--classes", "nope"]) == 1
