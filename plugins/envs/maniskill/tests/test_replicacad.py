"""Tests for the ReplicaCAD scene dataset integration (no network needed)."""

from __future__ import annotations

import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

# The plugin's datasets subpackage uses absolute imports, so it must be
# imported under its top-level name with ``core/`` on sys.path (same
# convention as the plugin's examples).
_CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))

from genesis_maniskill.datasets import replicacad_assets as assets  # noqa: E402
from genesis_maniskill.scenes import replica_cad_scene as rcs  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_archive(path: Path, entries: dict[str, bytes]) -> Path:
    """Write a fake ``replica_cad_dataset.zip`` without touching the network."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def _fake_scene_json() -> bytes:
    return json.dumps(
        {
            "stage_instance": {"template_name": "stages/frl_apartment_stage"},
            "object_instances": [
                {
                    "template_name": "objects/frl_apartment_basket",
                    "motion_type": "DYNAMIC",
                    "translation": [0.0, 0.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                }
            ],
            "articulated_object_instances": [],
        }
    ).encode()


def _write_fake_dataset(root: Path) -> Path:
    """Create a minimal extracted dataset directory under ``root``."""
    dset = assets.dataset_dir(root)
    (dset / "configs" / "scenes").mkdir(parents=True)
    (dset / "configs" / "objects").mkdir()
    (dset / "stages").mkdir()
    (dset / "objects").mkdir()
    (dset / "configs" / "scenes" / "apt_0.scene_instance.json").write_bytes(
        _fake_scene_json()
    )
    return dset


# ---------------------------------------------------------------------------
# Assets module
# ---------------------------------------------------------------------------


class TestAssets:
    """Download/verify/extract logic of replicacad_assets (no network)."""

    def test_url_points_at_modelscope_dataset(self):
        url = assets.dataset_url()
        assert "modelscope.cn" in url
        assert assets.MODELSCOPE_DATASET in url
        assert url.endswith("replica_cad_dataset.zip")

    def test_dataset_ready_and_list_scenes(self, tmp_path, monkeypatch):
        monkeypatch.setenv(assets.ENV_AUTO_DOWNLOAD, "0")
        _write_fake_dataset(tmp_path)
        assert assets.dataset_ready(tmp_path)
        assert assets.list_scenes(root=tmp_path) == ["apt_0"]

    def test_ensure_downloads_and_extracts(self, tmp_path, monkeypatch):
        archive = _make_archive(
            tmp_path / "src" / assets.ARCHIVE_NAME,
            {
                "replica_cad_dataset/configs/scenes/apt_0.scene_instance.json": _fake_scene_json(),
                "replica_cad_dataset/stages/frl_apartment_stage.glb": b"glb",
                "replica_cad_dataset/objects/frl_apartment_basket.glb": b"glb",
                "replica_cad_dataset/.git/config": b"junk",
                "replica_cad_dataset/.cache/huggingface/x.lock": b"junk",
                "replica_cad_dataset/__MACOSX/._foo": b"junk",
            },
        )

        def _fake_download(url: str, dest: Path) -> None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.read_bytes())

        monkeypatch.setattr(assets, "_download_file", _fake_download)
        dset = assets.ensure_dataset(root=tmp_path)

        assert dset.is_dir()
        assert (dset / "configs" / "scenes" / "apt_0.scene_instance.json").is_file()
        # Packing junk is skipped during extraction.
        assert not (dset / ".git").exists()
        assert not (dset / ".cache").exists()
        assert not (dset / "__MACOSX").exists()

    def test_reuses_existing_archive(self, tmp_path, monkeypatch):
        _make_archive(
            tmp_path / assets.ARCHIVE_NAME,
            {
                "replica_cad_dataset/configs/scenes/apt_0.scene_instance.json": _fake_scene_json(),
                "replica_cad_dataset/stages/s.glb": b"x",
                "replica_cad_dataset/objects/o.glb": b"x",
            },
        )

        def _boom(url: str, dest: Path) -> None:  # pragma: no cover - must not run
            raise AssertionError("network access is not allowed in tests")

        monkeypatch.setattr(assets, "_download_file", _boom)
        assets.ensure_dataset(root=tmp_path)
        assert assets.dataset_ready(tmp_path)

    def test_auto_download_disabled_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv(assets.ENV_AUTO_DOWNLOAD, "0")
        with pytest.raises(FileNotFoundError):
            assets.ensure_dataset(root=tmp_path)

    def test_corrupt_archive_rejected(self, tmp_path):
        root = tmp_path
        root.mkdir(exist_ok=True)
        (root / assets.ARCHIVE_NAME).write_bytes(b"not a zip")
        with pytest.raises(RuntimeError):
            assets.download_dataset(root)

    def test_zip_slip_rejected(self, tmp_path, monkeypatch):
        archive = _make_archive(
            tmp_path / "src" / assets.ARCHIVE_NAME,
            {"../evil.txt": b"pwn"},
        )

        def _fake_download(url: str, dest: Path) -> None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.read_bytes())

        monkeypatch.setattr(assets, "_download_file", _fake_download)
        with pytest.raises(RuntimeError, match="unsafe zip entry"):
            assets.ensure_dataset(root=tmp_path)
        assert not (tmp_path / "evil.txt").exists()


# ---------------------------------------------------------------------------
# Pose conversion
# ---------------------------------------------------------------------------


class TestPoseConversion:
    """habitat (Y-up, xyzw) -> genesis (Z-up, wxyz) conversion."""

    def test_identity(self):
        pos, quat = rcs.habitat_pose_to_genesis([0, 0, 0], [0, 0, 0, 1])
        np.testing.assert_allclose(pos, [0, 0, 0], atol=1e-12)
        s = math.sin(math.radians(45))
        c = math.cos(math.radians(45))
        np.testing.assert_allclose(quat, [c, s, 0, 0], atol=1e-12)

    def test_up_axis_maps_to_z(self):
        # 1 m along habitat's Y-up becomes +Z in genesis.
        pos, _ = rcs.habitat_pose_to_genesis([0, 1, 0], [0, 0, 0, 1])
        np.testing.assert_allclose(pos, [0, 0, 1], atol=1e-12)

    def test_x_translation_unchanged(self):
        pos, _ = rcs.habitat_pose_to_genesis([1, 0, 0], [0, 0, 0, 1])
        np.testing.assert_allclose(pos, [1, 0, 0], atol=1e-12)

    def test_habitat_z_maps_to_negative_y(self):
        pos, _ = rcs.habitat_pose_to_genesis([0, 0, 1], [0, 0, 0, 1])
        np.testing.assert_allclose(pos, [0, -1, 0], atol=1e-12)

    def test_unnormalized_quat_normalized(self):
        _, quat = rcs.habitat_pose_to_genesis([0, 0, 0], [0, 0, 0, 2])
        np.testing.assert_allclose(np.linalg.norm(quat), 1.0, atol=1e-12)

    def test_bad_shapes_raise(self):
        with pytest.raises(ValueError):
            rcs.habitat_pose_to_genesis([0, 0], [0, 0, 0, 1])
        with pytest.raises(ValueError):
            rcs.habitat_pose_to_genesis([0, 0, 0], [0, 0, 1])


# ---------------------------------------------------------------------------
# Scene config parsing (synthetic mini dataset, no Genesis needed)
# ---------------------------------------------------------------------------


class TestConfigParsing:
    """Resolution of scene/object/stage/URDF references."""

    @pytest.fixture()
    def mini_dataset(self, tmp_path: Path) -> Path:
        dset = tmp_path / "replica_cad_dataset"
        (dset / "configs" / "scenes").mkdir(parents=True)
        (dset / "configs" / "objects").mkdir()
        (dset / "configs" / "stages").mkdir()
        (dset / "stages").mkdir()
        (dset / "objects" / "convex").mkdir(parents=True)
        (dset / "urdf" / "fridge").mkdir(parents=True)

        (dset / "configs" / "scenes" / "mini.scene_instance.json").write_text(
            json.dumps(
                {
                    "stage_instance": {"template_name": "stages/stage_x"},
                    "object_instances": [
                        {
                            "template_name": "objects/obj_a",
                            "motion_type": "DYNAMIC",
                            "translation": [1, 2, 3],
                            "rotation": [0, 0, 0, 1],
                        },
                        {
                            "template_name": "objects/obj_b",
                            "motion_type": "STATIC",
                            "translation": [0, 0, 0],
                            "rotation": [0, 0, 0, 1],
                        },
                    ],
                    "articulated_object_instances": [
                        {
                            "template_name": "fridge",
                            "fixed_base": True,
                            "translation": [0, 0, 0],
                            "rotation": [0, 0, 0, 1],
                            "uniform_scale": 1.0,
                        }
                    ],
                }
            )
        )
        (dset / "configs" / "objects" / "obj_a.object_config.json").write_text(
            json.dumps(
                {
                    "render_asset": "../../objects/obj_a.glb",
                    "collision_asset": "../../objects/convex/obj_a_cv_decomp.glb",
                    "mass": 0.5,
                }
            )
        )
        (dset / "configs" / "objects" / "obj_b.object_config.json").write_text(
            json.dumps({"render_asset": "../../objects/obj_b.glb"})
        )
        (dset / "configs" / "stages" / "stage_x.stage_config.json").write_text(
            json.dumps({"render_asset": "../../stages/stage_x.glb"})
        )
        for p in (
            "objects/obj_a.glb",
            "objects/convex/obj_a_cv_decomp.glb",
            "objects/obj_b.glb",
            "stages/stage_x.glb",
            "urdf/fridge/fridge.urdf",
        ):
            (dset / p).write_bytes(b"x")
        return dset

    def test_load_scene_config(self, mini_dataset):
        cfg = rcs.load_scene_config(mini_dataset, "mini")
        assert cfg["stage_instance"]["template_name"] == "stages/stage_x"
        assert len(cfg["object_instances"]) == 2
        assert cfg["object_instances"][0]["motion_type"] == "DYNAMIC"

    def test_load_scene_config_missing(self, mini_dataset):
        with pytest.raises(FileNotFoundError):
            rcs.load_scene_config(mini_dataset, "nope")

    def test_resolve_object_meshes_prefers_collision(self, mini_dataset):
        meshes = rcs.resolve_object_meshes(mini_dataset, "objects/obj_a")
        assert meshes["visual"] == (mini_dataset / "objects" / "obj_a.glb").resolve()
        assert (
            meshes["body"]
            == (mini_dataset / "objects" / "convex" / "obj_a_cv_decomp.glb").resolve()
        )
        assert meshes["mass"] == 0.5

    def test_resolve_object_meshes_falls_back_to_visual(self, mini_dataset):
        meshes = rcs.resolve_object_meshes(mini_dataset, "objects/obj_b")
        assert meshes["body"] == meshes["visual"]
        assert meshes["mass"] is None

    def test_resolve_object_meshes_missing_template(self, mini_dataset):
        with pytest.raises(FileNotFoundError):
            rcs.resolve_object_meshes(mini_dataset, "objects/nope")

    def test_resolve_stage_mesh(self, mini_dataset):
        path = rcs.resolve_stage_mesh(mini_dataset, "stages/stage_x")
        assert path == (mini_dataset / "stages" / "stage_x.glb").resolve()

    def test_resolve_articulated_urdf(self, mini_dataset):
        path = rcs.resolve_articulated_urdf(mini_dataset, "fridge")
        assert path == (mini_dataset / "urdf" / "fridge" / "fridge.urdf").resolve()
        with pytest.raises(FileNotFoundError):
            rcs.resolve_articulated_urdf(mini_dataset, "nope")


# ---------------------------------------------------------------------------
# Genesis CPU smoke test (tiny synthetic scene)
# ---------------------------------------------------------------------------


def _write_tiny_glb(path: Path, size: float) -> None:
    import trimesh

    trimesh.creation.box(extents=[size, size, size]).export(str(path), file_type="glb")


@pytest.mark.slow()
def test_build_tiny_scene_on_cpu(tmp_path):
    """Build a minimal ReplicaCAD scene in Genesis and step it (CPU only)."""
    gs = pytest.importorskip("genesis")

    dset = tmp_path / "replica_cad_dataset"
    (dset / "configs" / "scenes").mkdir(parents=True)
    (dset / "configs" / "objects").mkdir()
    (dset / "configs" / "stages").mkdir()
    (dset / "stages").mkdir()
    (dset / "objects" / "convex").mkdir(parents=True)

    _write_tiny_glb(dset / "stages" / "stage_x.glb", 2.0)
    _write_tiny_glb(dset / "objects" / "obj_a.glb", 0.1)
    _write_tiny_glb(dset / "objects" / "convex" / "obj_a_cv_decomp.glb", 0.1)
    _write_tiny_glb(dset / "objects" / "obj_b.glb", 0.2)

    (dset / "configs" / "scenes" / "mini.scene_instance.json").write_text(
        json.dumps(
            {
                "stage_instance": {"template_name": "stages/stage_x"},
                "object_instances": [
                    {
                        "template_name": "objects/obj_a",
                        "motion_type": "DYNAMIC",
                        "translation": [0.0, 1.5, 0.0],
                        "rotation": [0.0, 0.0, 0.0, 1.0],
                    },
                    {
                        "template_name": "objects/obj_b",
                        "motion_type": "STATIC",
                        "translation": [0.5, 1.15, 0.0],
                        "rotation": [0.0, 0.0, 0.0, 1.0],
                    },
                ],
                "articulated_object_instances": [],
            }
        )
    )
    (dset / "configs" / "objects" / "obj_a.object_config.json").write_text(
        json.dumps(
            {
                "render_asset": "../../objects/obj_a.glb",
                "collision_asset": "../../objects/convex/obj_a_cv_decomp.glb",
                "mass": 0.5,
            }
        )
    )
    (dset / "configs" / "objects" / "obj_b.object_config.json").write_text(
        json.dumps({"render_asset": "../../objects/obj_b.glb"})
    )
    (dset / "configs" / "stages" / "stage_x.stage_config.json").write_text(
        json.dumps({"render_asset": "../../stages/stage_x.glb"})
    )

    if not gs._initialized:
        gs.init(backend=gs.cpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        vis_options=gs.options.VisOptions(ambient_light=(0.3, 0.3, 0.3)),
        show_viewer=False,
    )
    builder = rcs.ReplicaCADSceneBuilder(
        scene, scene_name="mini", dataset_root=tmp_path, config={"add_lights": False}
    )
    builder.build()
    scene.build()
    builder.finalize()

    # Habitat Y-up: (0, 1.5, 0) -> genesis z=1.5, i.e. 0.45 m above the
    # stage top (stage is a 2 m box centered at the origin).
    movable = builder.movable_objects["obj_a-0"]
    z0 = float(movable.get_pos()[2])
    assert z0 == pytest.approx(1.5, abs=1e-5)
    for _ in range(20):
        scene.step()
    pos = np.asarray(movable.get_pos()).reshape(-1)
    assert np.all(np.isfinite(pos))
    # The dynamic box falls towards the stage top.
    assert float(pos[2]) < z0
    assert float(pos[2]) > 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
