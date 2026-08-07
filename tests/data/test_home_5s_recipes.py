"""Validate the home-5S task -> asset mapping (data/recipes/home_5s.yaml).

Guards the 移植方案设计书 §9.3 deliverable: every object class referenced by
the 5S task recipes must exist in the RoboTwin-OD object library, and the
suite index must point at real files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPES_PATH = REPO_ROOT / "data" / "recipes" / "home_5s.yaml"
SUITE_PATH = REPO_ROOT / "data" / "suites" / "home_5s_suite.yaml"
OBJECTS_DIR = REPO_ROOT / "assets" / "robotwin" / "objects" / "objects"

S5_CATEGORIES = {"整理", "整顿", "清扫", "清洁", "素养"}


@pytest.fixture(scope="module")
def recipes() -> dict:
    """Load the recipes yaml once per module."""
    return yaml.safe_load(RECIPES_PATH.read_text(encoding="utf-8"))


def test_recipes_schema(recipes: dict) -> None:
    """Top-level schema and per-task fields are well-formed."""
    assert recipes["version"] == 1
    assert recipes["s5_categories"]
    tasks = recipes["tasks"]
    assert len(tasks) >= 5, "设计书 §1.1 列举的五类隐形家务任务应全部覆盖"
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids)), "task ids must be unique"
    for task in tasks:
        assert task["name_zh"]
        assert task["s5_category"] in S5_CATEGORIES
        assert task["s5_category"] in recipes["s5_categories"]
        assert isinstance(task["object_classes"], list)
        assert isinstance(task.get("target_classes", []), list)
        assert isinstance(task.get("asset_gaps", []), list)


def test_recipes_cover_design_doc_tasks(recipes: dict) -> None:
    """设计书 §1.1 的隐形家务任务（归位/收纳/台面/床品/橱柜）全部有映射。"""
    names = {t["name_zh"] for t in recipes["tasks"]}
    assert {"物品归位", "收纳整理", "台面清洁", "床品整理", "橱柜归整"} <= names


def test_every_s5_category_used(recipes: dict) -> None:
    """Each declared 5S category has at least one task."""
    used = {t["s5_category"] for t in recipes["tasks"]}
    assert used == set(recipes["s5_categories"])


@pytest.mark.skipif(not OBJECTS_DIR.is_dir(), reason="RoboTwin assets not present")
def test_referenced_classes_exist(recipes: dict) -> None:
    """Every referenced object/target class exists in the object library."""
    from cloud_robotics_sim.robotwin.object_library import RoboTwinObjectLibrary

    available = set(RoboTwinObjectLibrary(OBJECTS_DIR).list_classes())
    referenced = set()
    for task in recipes["tasks"]:
        referenced.update(task["object_classes"])
        referenced.update(task.get("target_classes", []))
    missing = referenced - available
    assert not missing, f"classes referenced but missing from library: {missing}"


@pytest.mark.skipif(not OBJECTS_DIR.is_dir(), reason="RoboTwin assets not present")
def test_referenced_classes_loadable(recipes: dict) -> None:
    """First instance of every referenced class parses (glb and urdf kinds)."""
    from cloud_robotics_sim.robotwin.object_library import RoboTwinObjectLibrary

    lib = RoboTwinObjectLibrary(OBJECTS_DIR)
    for task in recipes["tasks"]:
        for cls in task["object_classes"] + task.get("target_classes", []):
            inst = lib.get_instance(cls)  # None -> first available (glb or urdf)
            assert inst.asset_path.is_file(), f"{cls}: {inst.asset_path}"


def test_suite_index_points_at_real_files() -> None:
    """The suite yaml references existing recipes / scripts / tests."""
    suite = yaml.safe_load(SUITE_PATH.read_text(encoding="utf-8"))
    assert suite["version"] == 1
    for recipe in suite["recipes"]:
        assert (SUITE_PATH.parent / recipe).resolve().is_file(), recipe
    assert (REPO_ROOT / suite["assets"]["object_library"]).is_dir()
    assert (REPO_ROOT / suite["assets"]["expansion_workflow"]).is_file()
    assert (REPO_ROOT / suite["validation"]["benchmark_runner"]).is_file()
    assert (REPO_ROOT / suite["validation"]["mapping_test"]).is_file()
    license_gate = suite["validation"]["license_gate"].split()[0]
    assert (REPO_ROOT / license_gate).is_file()
