"""Validate the storage asset gap registry (data/recipes/asset_gap_list.yaml).

The gap list is the 补库路线图 for the home-5S asset library: every entry
must be well-formed, priorities/feasibility must come from the declared
enums, covered_by classes must exist in the object library, and related
tasks must exist in data/recipes/home_5s.yaml.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]
GAP_LIST_PATH = REPO_ROOT / "data" / "recipes" / "asset_gap_list.yaml"
RECIPES_PATH = REPO_ROOT / "data" / "recipes" / "home_5s.yaml"
OBJECTS_DIR = REPO_ROOT / "assets" / "robotwin" / "objects" / "objects"

PRIORITIES = {"P1", "P2", "P3"}
FEASIBILITY = {"rigid", "articulated", "soft_body", "static_fixture", "out_of_scope"}
STATUSES = {"covered", "covered_partial", "gap"}


@pytest.fixture(scope="module")
def gap_list() -> dict:
    """Load the gap list yaml once per module."""
    return yaml.safe_load(GAP_LIST_PATH.read_text(encoding="utf-8"))


def test_gap_list_schema(gap_list: dict) -> None:
    """Unique ids; enums valid; required fields present."""
    assert gap_list["version"] == 1
    assert set(gap_list["priority_tiers"]) == PRIORITIES
    assert set(gap_list["feasibility_kinds"]) == FEASIBILITY
    items = gap_list["items"]
    assert len(items) >= 40, "收纳清单 8 大区应覆盖 40+ 条目"
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids)), "item ids must be unique"
    for item in items:
        assert item["name_zh"] and item["zone"]
        assert item["priority"] in PRIORITIES
        assert item["feasibility"] in FEASIBILITY
        assert item["status"] in STATUSES
        assert isinstance(item["covered_by"], list)
        assert isinstance(item["related_tasks"], list)
        assert item["note"]


def test_status_consistent_with_covered_by(gap_list: dict) -> None:
    """covered/covered_partial must name the stand-in classes."""
    for item in gap_list["items"]:
        if item["status"] in ("covered", "covered_partial"):
            assert item[
                "covered_by"
            ], f"{item['id']}: {item['status']} needs covered_by"


def test_p1_items_actionable(gap_list: dict) -> None:
    """P1 (入住即买) items must be sim-feasible, never out_of_scope."""
    for item in gap_list["items"]:
        if item["priority"] == "P1":
            assert item["feasibility"] != "out_of_scope", item["id"]


def test_related_tasks_exist(gap_list: dict) -> None:
    """related_tasks must reference task ids from home_5s.yaml."""
    recipes = yaml.safe_load(RECIPES_PATH.read_text(encoding="utf-8"))
    task_ids = {t["id"] for t in recipes["tasks"]}
    for item in gap_list["items"]:
        unknown = set(item["related_tasks"]) - task_ids
        assert not unknown, f"{item['id']}: unknown tasks {unknown}"


def test_zones_cover_checklist(gap_list: dict) -> None:
    """清单 8 大区全部有条目。"""
    zones = {i["zone"] for i in gap_list["items"]}
    assert {
        "玄关",
        "厨房",
        "客厅",
        "卧室",
        "卫生间",
        "阳台储物",
        "书房儿童",
        "通用",
    } <= zones


@pytest.mark.skipif(not OBJECTS_DIR.is_dir(), reason="RoboTwin assets not present")
def test_covered_by_classes_exist(gap_list: dict) -> None:
    """Every covered_by class exists in the object library."""
    from cloud_robotics_sim.robotwin.object_library import RoboTwinObjectLibrary

    available = set(RoboTwinObjectLibrary(OBJECTS_DIR).list_classes())
    for item in gap_list["items"]:
        missing = set(item["covered_by"]) - available
        assert not missing, f"{item['id']}: covered_by classes missing {missing}"
