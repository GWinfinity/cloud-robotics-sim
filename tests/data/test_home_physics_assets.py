"""Validate the household-item physics classification recipe
(data/recipes/home_physics_assets.yaml).

Guards the 80 平 housework asset inventory: every item is classified as
rigid / fragile_rigid / soft_body, gap items carry a generation route
(trimesh / mesh_import / articulated_urdf / cloth_sim) with size parameters
where applicable, and the scene-level SDF layer stays out of the item layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE_PATH = REPO_ROOT / "data" / "recipes" / "home_physics_assets.yaml"
HOME_5S_PATH = REPO_ROOT / "data" / "recipes" / "home_5s.yaml"
OBJECTS_DIR = REPO_ROOT / "assets" / "robotwin" / "objects" / "objects"

PHYSICS_KINDS = {"rigid", "fragile_rigid", "soft_body"}
ROUTES = {"trimesh", "mesh_import", "articulated_urdf", "cloth_sim"}
STATUSES = {"covered", "gap"}
SCENE_ROUTE = "xuvdb_sdf_bake"


@pytest.fixture(scope="module")
def recipe() -> dict:
    """Load the recipe yaml once per module."""
    return yaml.safe_load(RECIPE_PATH.read_text(encoding="utf-8"))


def test_schema(recipe: dict) -> None:
    """Top-level schema and per-item required fields are well-formed."""
    assert recipe["version"] == 1
    assert set(recipe["physics_kinds"]) == PHYSICS_KINDS
    assert set(recipe["generation_routes"]) == ROUTES
    items = recipe["items"]
    assert len(items) >= 40, "80 平高频家务物品应基本覆盖"
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), "item ids must be unique"
    for it in items:
        assert it["name_zh"]
        assert it["zone"] in recipe["zones"]
        assert it["physics_kind"] in PHYSICS_KINDS
        assert it["status"] in STATUSES


def test_every_physics_kind_represented(recipe: dict) -> None:
    """三类物理（刚体/易碎刚体/柔软体）各有足够样本，不是摆设分类。"""
    counts = {kind: 0 for kind in PHYSICS_KINDS}
    for it in recipe["items"]:
        counts[it["physics_kind"]] += 1
    for kind, n in counts.items():
        assert n >= 3, f"{kind} 仅 {n} 项，分类覆盖不足"


def test_covered_vs_gap_fields(recipe: dict) -> None:
    """字段规则：已覆盖项必须有 covered_by 且无 route；缺口项必须有合法 route 且无 covered_by。"""
    for it in recipe["items"]:
        if it["status"] == "covered":
            assert it.get("covered_by"), f"{it['id']}: covered item needs covered_by"
            assert "route" not in it, f"{it['id']}: covered item must not set route"
        else:
            assert it.get("route") in ROUTES, f"{it['id']}: gap item needs a route"
            assert (
                "covered_by" not in it
            ), f"{it['id']}: gap item must not set covered_by"


def test_trimesh_items_carry_sane_dims(recipe: dict) -> None:
    """程序化生成项给出 [x,y,z] 米制包围盒，数量级合理。"""
    for it in recipe["items"]:
        if it.get("route") != "trimesh":
            continue
        dims = it.get("dims_m")
        assert dims is not None, f"{it['id']}: trimesh route requires dims_m"
        assert len(dims) == 3, f"{it['id']}: dims_m must be [x, y, z]"
        for d in dims:
            assert isinstance(d, float), f"{it['id']}: dims must be floats, got {d!r}"
            assert 0.0 < d <= 3.0, f"{it['id']}: dim {d} out of household range"


def test_item_layer_never_uses_scene_sdf_route(recipe: dict) -> None:
    """物品层禁止走 xuvdb SDF：薄壁/破碎/抓取语义都不支持（评估结论固化为约束）。"""
    for it in recipe["items"]:
        assert (
            it.get("route") != SCENE_ROUTE
        ), f"{it['id']}: items must not use {SCENE_ROUTE}"


def test_scene_statics_use_sdf_bake(recipe: dict) -> None:
    """静态场景层（墙/台面/水槽）统一走 xuvdb 整屋烘焙，体素尺寸有限制。"""
    statics = recipe["scene_statics"]
    assert len(statics) >= 3
    for st in statics:
        assert st["route"] == SCENE_ROUTE, f"{st['id']}: statics must bake to xuvdb SDF"
        vox = st["voxel_size_m"]
        assert 0.0 < vox <= 0.05, f"{st['id']}: voxel {vox} too coarse for contact"


def test_related_tasks_exist_in_home_5s(recipe: dict) -> None:
    """related_tasks 交叉引用 home_5s.yaml 中真实存在的任务 id。"""
    home_5s = yaml.safe_load(HOME_5S_PATH.read_text(encoding="utf-8"))
    task_ids = {t["id"] for t in home_5s["tasks"]}
    for it in recipe["items"]:
        for task in it.get("related_tasks", []):
            assert task in task_ids, f"{it['id']}: unknown related_task {task}"


def _reference_assets_present() -> bool:
    """Partial local generated libraries are not the complete RoboTwin reference set."""
    if not OBJECTS_DIR.is_dir():
        return False
    recipe = yaml.safe_load(RECIPE_PATH.read_text(encoding="utf-8"))
    return all(
        (OBJECTS_DIR / covered).is_dir()
        for item in recipe["items"]
        for covered in item.get("covered_by", [])
    )


@pytest.mark.skipif(
    not _reference_assets_present(), reason="Complete RoboTwin assets not present"
)
def test_covered_classes_exist(recipe: dict) -> None:
    """covered_by 引用的物体类存在于 RoboTwin-OD 物体库。"""
    from cloud_robotics_sim.robotwin.object_library import RoboTwinObjectLibrary

    available = set(RoboTwinObjectLibrary(OBJECTS_DIR).list_classes())
    referenced = set()
    for it in recipe["items"]:
        referenced.update(it.get("covered_by", []))
    missing = referenced - available
    assert not missing, f"classes referenced but missing from library: {missing}"


def test_dynamics_batches_reference_real_entrypoints(recipe: dict) -> None:
    """动力学批次的物品引用、生成器和运行入口均真实存在。"""
    item_ids = {item["id"] for item in recipe["items"]}
    batches = {batch["id"]: batch for batch in recipe["dynamics_batches"]}
    cup = batches["cup_fracture_v1"]
    assert cup["target"] in item_ids
    assert cup["fragment_count"] >= 2
    assert cup["impulse_threshold_Ns"] > 0.0
    cloth = batches["household_cloth_v1"]
    assert set(cloth["items"]) <= item_ids
    for batch in batches.values():
        assert (REPO_ROOT / batch["generator"]).is_file()
        assert (REPO_ROOT / batch["runtime"]).is_file()
