"""Static three-way classification tests for ReplicaCADSceneBuilder.

Offline: a minimal fake dataset is created under tmp_path (no network). The
Genesis smoke test additionally requires genesis-world and the real local
dataset; it auto-skips without them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))

from genesis_maniskill.scenes import replica_cad_scene as rcs  # noqa: E402

try:
    from tests.optional_deps import genesis_only, xuvdb_only
except ImportError:  # plugin tests also run standalone (repo root off sys.path)
    genesis_only = pytest.mark.skipif(
        __import__("importlib.util", fromlist=["util"]).find_spec("genesis") is None,
        reason="genesis-world not installed",
    )
    xuvdb_only = pytest.mark.skipif(
        __import__("importlib.util", fromlist=["util"]).find_spec("xuvdb") is None,
        reason="xuvdb not installed",
    )

# ---------------------------------------------------------------------------
# Fake dataset (same convention as test_replicacad.py)
# ---------------------------------------------------------------------------


def _write_fake_dataset(root: Path) -> Path:
    dset = root / "replica_cad_dataset"
    (dset / "configs" / "scenes").mkdir(parents=True)
    (dset / "configs" / "objects").mkdir()
    (dset / "configs" / "stages").mkdir()
    (dset / "stages").mkdir()
    (dset / "objects").mkdir()
    (dset / "configs" / "scenes" / "fake.scene_instance.json").write_text(
        json.dumps(
            {
                "stage_instance": {"template_name": "stages/fake_stage"},
                "object_instances": [
                    {
                        "template_name": "objects/fake_obj",
                        "motion_type": "DYNAMIC",
                        "translation": [0.0, 0.0, 0.0],
                        "rotation": [0.0, 0.0, 0.0, 1.0],
                    },
                    {
                        "template_name": "objects/fake_obj",
                        "motion_type": "STATIC",
                        "translation": [1.0, 0.0, 0.0],
                        "rotation": [0.0, 0.0, 0.0, 1.0],
                    },
                ],
                "articulated_object_instances": [
                    {
                        "template_name": "urdf/fake_art",
                        "translation": [2.0, 0.0, 0.0],
                        "rotation": [0.0, 0.0, 0.0, 1.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (dset / "configs" / "stages" / "fake_stage.stage_config.json").write_text(
        json.dumps({"render_asset": "../../stages/fake_stage.glb"}), encoding="utf-8"
    )
    (dset / "configs" / "objects" / "fake_obj.object_config.json").write_text(
        json.dumps({"render_asset": "../../objects/fake_obj.glb"}), encoding="utf-8"
    )
    # Mesh files are never loaded by the classification path (resolution only).
    (dset / "stages" / "fake_stage.glb").write_bytes(b"")
    (dset / "objects" / "fake_obj.glb").write_bytes(b"")
    return dset


# ---------------------------------------------------------------------------
# collect_static_files
# ---------------------------------------------------------------------------


def test_collect_static_files_stage_plus_static_only(tmp_path) -> None:
    """Stage 与 STATIC 对象进汤，DYNAMIC 排除。"""
    dset = _write_fake_dataset(tmp_path)
    config = rcs.load_scene_config(dset, "fake")
    entries = rcs.collect_static_files(config, dset)
    # stage + 1 STATIC object; the DYNAMIC object is excluded.
    assert len(entries) == 2
    stage_path, stage_pose = entries[0]
    assert stage_path == (dset / "stages" / "fake_stage.glb").resolve()
    # The stage pose is the habitat->Genesis RotX(+90 deg) frame rotation
    # (translation zero) left-multiplied onto the raw habitat-frame mesh —
    # the bake path (habitat_pose_to_bake), NOT the entity conjugation.
    assert np.allclose(stage_pose[:3, 3], [0.0, 0.0, 0.0], atol=1e-9)
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    assert np.allclose(stage_pose[:3, :3], expected, atol=1e-9)
    obj_path, obj_pose = entries[1]
    assert obj_path == (dset / "objects" / "fake_obj.glb").resolve()
    # habitat (1,0,0) -> Genesis (1,0,0) under the RotX(+90) frame conversion.
    assert np.allclose(obj_pose[:3, 3], [1.0, 0.0, 0.0], atol=1e-9)


def test_ground_slab_spec() -> None:
    """地板替代盒顶面与 z=0 齐平且固定。"""
    spec = rcs.ground_slab_spec()
    size = spec["size"]
    pos = spec["pos"]
    assert spec["fixed"] is True
    # Slab top flush with the z=0 apartment floor.
    assert pos[2] + size[2] / 2 == pytest.approx(0.0)
    assert size[0] > 0 and size[1] > 0


def test_static_mode_validation_before_dataset_access() -> None:
    """Unknown static_mode raises before any dataset/scene access."""
    builder = rcs.ReplicaCADSceneBuilder(
        scene=None, scene_name="fake", config={"static_mode": "banana"}
    )
    with pytest.raises(ValueError, match="static_mode"):
        builder.build()


def test_default_config_unchanged() -> None:
    """默认配置保持 mesh 模式现状行为。"""
    builder = rcs.ReplicaCADSceneBuilder(scene=None, config=None)
    assert (
        "static_mode" not in builder.config
        or builder.config.get("static_mode") == "mesh"
    )
    assert builder.static_sdf is None


# ---------------------------------------------------------------------------
# Genesis smoke (auto-skips without genesis / local dataset)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@genesis_only
@xuvdb_only
def test_apt0_static_sdf_smoke(tmp_path) -> None:
    """apt_0 static_mode="sdf" on the CPU backend: bake + query self-check."""
    from genesis_maniskill.datasets.replicacad_assets import ensure_dataset

    try:
        dset = ensure_dataset()
    except Exception as exc:  # noqa: BLE001 - dataset absent offline
        pytest.skip(f"ReplicaCAD dataset unavailable: {exc}")

    import genesis as gs

    from cloud_robotics_sim.utils.genesis_compat import get_genesis_backend

    # Prefer CUDA on GPU boxes (the genesis "gpu" alias auto-falls back to
    # cpu); CI without a GPU runs the same path on cpu.
    gs.init(backend=get_genesis_backend("gpu"), precision="32")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=1),
        show_viewer=False,
    )
    builder = rcs.ReplicaCADSceneBuilder(
        scene=scene,
        scene_name="apt_0",
        dataset_root=dset.parent,
        config={
            "static_mode": "sdf",
            "sdf_voxel_size": 0.05,
            "sdf_cache_dir": str(tmp_path / "cache"),
            "add_lights": False,
        },
    )
    builder.build()
    assert builder.static_sdf is not None
    vol = builder.static_sdf
    report = vol.memory_report()
    assert report["active_voxels"] > 0

    # Sign census over the dense leaf buffers (random *linear* samples would
    # interpolate thin wall interiors away).
    n_neg = sum(int((leaf.values < 0).sum()) for leaf in vol.grid.leaves())
    n_pos = sum(int((leaf.values > 0).sum()) for leaf in vol.grid.leaves())
    assert n_neg > 0, "baked shell has no interior voxels"
    assert n_pos > 0, "baked shell has no exterior voxels"

    # Trilinear sampling over the bake bbox stays finite.
    lo, hi = report["bbox_ijk"]
    rng = np.random.default_rng(0)
    pts = rng.uniform(lo, hi, size=(512, 3)).astype(np.float32) * np.float32(
        report["voxel_size"]
    )
    vals = np.atleast_1d(vol.sample_linear(pts))
    assert np.all(np.isfinite(vals))
