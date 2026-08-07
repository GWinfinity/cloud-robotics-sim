"""Validate the asset upgrade list (data/recipes/asset_upgrade_list.yaml).

The upgrade list drives the Poly Pizza / PartNet-Mobility fidelity pass
(Objaverse declared but unreachable from this network since 2026-08):
every referenced class must exist in the object library, tiers and upgrade
paths must come from the declared enums, and related tasks must exist in
the home-5S recipes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]
UPGRADE_PATH = REPO_ROOT / "data" / "recipes" / "asset_upgrade_list.yaml"
RECIPES_PATH = REPO_ROOT / "data" / "recipes" / "home_5s.yaml"
OBJECTS_DIR = REPO_ROOT / "assets" / "robotwin" / "objects" / "objects"

TIERS = {"U1", "U2", "U3"}
PATHS = {"objaverse", "poly_pizza", "partnet_mobility"}


@pytest.fixture(scope="module")
def upgrade() -> dict:
    """Load the upgrade list yaml once per module."""
    return yaml.safe_load(UPGRADE_PATH.read_text(encoding="utf-8"))


def _entries(upgrade: dict) -> list[dict]:
    """Flatten container + contents entries of every scene."""
    out = []
    for scene in upgrade["scenes"]:
        out.append(scene["container"])
        out.extend(scene.get("contents", []))
    return out


def test_upgrade_schema(upgrade: dict) -> None:
    """Scenes well-formed; enums valid; every entry has a note."""
    assert upgrade["version"] == 1
    assert set(upgrade["tiers"]) == TIERS
    assert set(upgrade["upgrade_paths"]) == PATHS
    scenes = upgrade["scenes"]
    assert len(scenes) >= 15, "8 大生活区的收纳场景应基本覆盖"
    ids = [s["id"] for s in scenes]
    assert len(ids) == len(set(ids)), "scene ids must be unique"
    for scene in scenes:
        assert scene["zone"] and scene["related_task"]
        for entry in [scene["container"], *scene.get("contents", [])]:
            assert entry["tier"] in TIERS, entry
            assert entry["path"] in PATHS, entry
            assert entry["note"], entry


def test_related_tasks_exist(upgrade: dict) -> None:
    """related_task references must exist in home_5s.yaml."""
    recipes = yaml.safe_load(RECIPES_PATH.read_text(encoding="utf-8"))
    task_ids = {t["id"] for t in recipes["tasks"]}
    for scene in upgrade["scenes"]:
        assert scene["related_task"] in task_ids, scene["id"]


def test_articulated_uses_partnet(upgrade: dict) -> None:
    """Furniture-scale articulated classes must go the PartNet path."""
    articulated = {
        "125_fridge",
        "126_wardrobe",
        "163_storage_ottoman",
        "162_side_table_cabinet",
        "158_desk_cart",
        "060_kitchenpot",
    }
    for entry in _entries(upgrade):
        cls = entry["class"]
        if cls in articulated:
            assert entry["path"] == "partnet_mobility", cls


def test_u1_nonempty_and_bounded(upgrade: dict) -> None:
    """U1 (优先升级) is a small, focused set — not everything at once."""
    u1 = sorted({e["class"] for e in _entries(upgrade) if e["tier"] == "U1"})
    assert 5 <= len(u1) <= 12, f"U1 should stay focused, got {len(u1)}: {u1}"


@pytest.mark.skipif(not OBJECTS_DIR.is_dir(), reason="RoboTwin assets not present")
def test_referenced_classes_exist(upgrade: dict) -> None:
    """Every container/contents class exists in the object library."""
    from cloud_robotics_sim.robotwin.object_library import RoboTwinObjectLibrary

    available = set(RoboTwinObjectLibrary(OBJECTS_DIR).list_classes())
    for entry in _entries(upgrade):
        assert entry["class"] in available, entry["class"]
