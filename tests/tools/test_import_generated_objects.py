"""Tests for the generated-object importer (tools/import_generated_objects.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloud_robotics_sim.robotwin.object_library import (  # noqa: E402
    RoboTwinObjectLibrary,
)
from tools.import_generated_objects import (  # noqa: E402
    collect_input_files,
    import_meshes,
    load_mesh,
    next_instance_index,
    normalize_mesh,
    resolve_class_name,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_OBJECTS_DIR = REPO_ROOT / "assets" / "robotwin" / "objects" / "objects"


def _export_glb(mesh, path: Path) -> Path:
    mesh.export(str(path))
    return path


def test_collect_input_files(tmp_path: Path) -> None:
    """Directories expand to sorted mesh files; junk is skipped."""
    import trimesh

    mesh_dir = tmp_path / "in"
    mesh_dir.mkdir()
    for name in ("b.glb", "a.glb"):
        _export_glb(trimesh.creation.box(), mesh_dir / name)
    (mesh_dir / "notes.txt").write_text("ignore me", encoding="utf-8")
    files = collect_input_files([str(mesh_dir)])
    assert [f.name for f in files] == ["a.glb", "b.glb"]
    with pytest.raises(FileNotFoundError):
        collect_input_files([str(tmp_path / "missing.glb")])
    with pytest.raises(ValueError):
        collect_input_files([str(mesh_dir / "notes.txt")])


def test_resolve_class_name(tmp_path: Path) -> None:
    """Bare names get the next free id; prefixed names pass through."""
    (tmp_path / "001_bottle").mkdir()
    (tmp_path / "120_plant").mkdir()
    assert resolve_class_name("mug", tmp_path) == "121_mug"
    assert resolve_class_name("001_bottle", tmp_path) == "001_bottle"
    assert resolve_class_name("mug", tmp_path / "missing") == "001_mug"


def test_next_instance_index(tmp_path: Path) -> None:
    """Numbering continues after the highest existing model_data<N>.json."""
    assert next_instance_index(tmp_path) == 0
    (tmp_path / "model_data0.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model_data2.json").write_text("{}", encoding="utf-8")
    assert next_instance_index(tmp_path) == 3


def test_enforce_min_thickness_pads_thin_axes() -> None:
    """Thin axes are padded to the floor; thick meshes pass through unchanged."""
    import trimesh

    from tools.import_generated_objects import enforce_min_thickness

    thin = trimesh.creation.box(extents=(0.09, 0.008, 0.038))
    padded = enforce_min_thickness(thin, 0.012)
    assert min(padded.extents) == pytest.approx(0.012)
    assert padded.extents[0] == pytest.approx(0.09)  # other axes untouched

    thick = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    assert enforce_min_thickness(thick, 0.012) is thick
    assert enforce_min_thickness(thin, 0.0) is thin  # disabled


def test_normalize_mesh() -> None:
    """Recentering puts the base at y=0, xz at origin; max edge hits target."""
    import trimesh

    mesh = trimesh.creation.box(extents=(1.0, 2.0, 0.5))
    mesh.apply_translation([5.0, 3.0, -2.0])  # off-center, floating
    out = normalize_mesh(mesh, target_size=0.15)
    bounds = out.bounds
    assert bounds[0][1] == pytest.approx(0.0)  # base on the ground
    center_xz = (bounds[0] + bounds[1]) / 2.0
    assert center_xz[0] == pytest.approx(0.0, abs=1e-7)
    assert center_xz[2] == pytest.approx(0.0, abs=1e-7)
    assert max(out.extents) == pytest.approx(0.15)


def test_load_mesh_z_up(tmp_path: Path) -> None:
    """--z-up rotates Z-up geometry into the Y-up glTF frame."""
    import trimesh

    # tall along Z; after z_up rotation it must be tall along Y
    path = _export_glb(
        trimesh.creation.box(extents=(0.1, 0.1, 0.3)), tmp_path / "z.glb"
    )
    mesh = load_mesh(path, z_up=True)
    assert mesh.extents[1] == pytest.approx(max(mesh.extents), rel=1e-4)


def test_import_meshes_end_to_end(tmp_path: Path) -> None:
    """Two raw meshes -> two instances, loadable by RoboTwinObjectLibrary."""
    import trimesh

    src = tmp_path / "src"
    src.mkdir()
    box = trimesh.creation.box(extents=(1.0, 2.0, 1.0))
    box.apply_translation([10.0, 5.0, 0.0])
    _export_glb(box, src / "style_a.glb")
    _export_glb(
        trimesh.creation.icosphere(subdivisions=2, radius=3.0), src / "style_b.glb"
    )

    objects_dir = tmp_path / "objects"
    results = import_meshes(
        [str(src / "style_a.glb"), str(src / "style_b.glb")],
        class_name="gen-thing",
        objects_dir=objects_dir,
        target_size=0.12,
    )
    assert [r.index for r in results] == [0, 1]
    assert results[0].class_name == "001_gen-thing"

    class_dir = objects_dir / "001_gen-thing"
    for i in (0, 1):
        assert (class_dir / "collision" / f"base{i}.glb").is_file()
        meta = json.loads((class_dir / f"model_data{i}.json").read_text("utf-8"))
        assert meta["scale"] == [1.0, 1.0, 1.0]
        assert max(meta["extents"]) == pytest.approx(0.12, rel=1e-3)
        assert meta["stable"] is True

    # base of the exported mesh sits at y=0 (glTF frame)
    import trimesh as tm

    exported = tm.load(str(class_dir / "collision" / "base0.glb"), force="mesh")
    assert exported.bounds[0][1] == pytest.approx(0.0, abs=1e-5)

    lib = RoboTwinObjectLibrary(objects_dir)
    assert lib.list_classes() == ["001_gen-thing"]
    assert lib.instance_count("001_gen-thing") == 2
    inst = lib.get_instance("001_gen-thing", 1)
    assert inst.kind == "glb"
    assert inst.asset_path.is_file()
    assert max(inst.scaled_extents) == pytest.approx(0.12, rel=1e-3)


def test_import_appends_instances(tmp_path: Path) -> None:
    """Importing into an existing class appends new style instances."""
    import trimesh

    glb = _export_glb(trimesh.creation.box(), tmp_path / "a.glb")
    objects_dir = tmp_path / "objects"
    import_meshes([str(glb)], class_name="005_mug", objects_dir=objects_dir)
    results = import_meshes([str(glb)], class_name="005_mug", objects_dir=objects_dir)
    assert results[0].index == 1
    lib = RoboTwinObjectLibrary(objects_dir)
    assert lib.instance_indices("005_mug") == [0, 1]


def test_import_decimates_dense_mesh(tmp_path: Path) -> None:
    """Meshes above max_faces are simplified for the contact solver."""
    import trimesh

    dense = trimesh.creation.icosphere(subdivisions=5, radius=1.0)  # 20480 faces
    glb = _export_glb(dense, tmp_path / "dense.glb")
    results = import_meshes(
        [str(glb)], class_name="ball", objects_dir=tmp_path / "objects", max_faces=2000
    )
    r = results[0]
    assert r.faces_in == 20480
    assert r.faces_out <= 2000
    assert any("decimated" in w for w in r.warnings)


@pytest.mark.skipif(not REAL_OBJECTS_DIR.is_dir(), reason="real assets not present")
def test_import_real_robotwin_glb_roundtrip(tmp_path: Path) -> None:
    """A shipped RoboTwin GLB survives the import pipeline (self-test path)."""
    objects_dir = tmp_path / "objects"
    results = import_meshes(
        [str(REAL_OBJECTS_DIR / "002_bowl" / "collision" / "base1.glb")],
        class_name="bowl-copy",
        objects_dir=objects_dir,
        target_size=0.15,
    )
    assert results[0].glb_path.is_file()
    lib = RoboTwinObjectLibrary(objects_dir)
    inst = lib.get_instance("001_bowl-copy")
    assert inst.asset_path.is_file()
    scaled = inst.scaled_extents
    assert max(scaled) == pytest.approx(0.15, rel=1e-3)
    assert np.all(np.array(scaled) > 0)


def test_import_records_provenance(tmp_path: Path) -> None:
    """License / source_url / author provenance lands in model_data<N>.json."""
    import trimesh

    src = tmp_path / "src"
    src.mkdir()
    _export_glb(trimesh.creation.box(), src / "mug_a.glb")

    objects_dir = tmp_path / "objects"
    provenance = {
        "license": "CC-BY-4.0",
        "source_url": "https://objaverse.example/uid-123",
        "author": "unit-test",
    }
    results = import_meshes(
        [str(src / "mug_a.glb")],
        class_name="mug-gen",
        objects_dir=objects_dir,
        provenance=provenance,
    )
    meta = json.loads(results[0].metadata_path.read_text("utf-8"))
    assert meta["license"] == "CC-BY-4.0"
    assert meta["source_url"] == "https://objaverse.example/uid-123"
    assert meta["author"] == "unit-test"
    assert meta["generator"] == "tools/import_generated_objects.py"

    # no provenance -> no license key (manifest will flag UNREGISTERED)
    results2 = import_meshes(
        [str(src / "mug_a.glb")], class_name="mug-gen2", objects_dir=objects_dir
    )
    meta2 = json.loads(results2[0].metadata_path.read_text("utf-8"))
    assert "license" not in meta2
