"""Tests for the asset expansion workflow (scripts/expand_object_library.py)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloud_robotics_sim.robotwin.object_library import (  # noqa: E402
    RoboTwinObjectLibrary,
)
from scripts.expand_object_library import (  # noqa: E402
    discover_staging,
    parse_smoke_output,
    run_import_stage,
    run_workflow,
    write_report,
)


def _make_staging(tmp_path: Path) -> Path:
    """Staging dir: one bare-name class (2 meshes) + one loose mesh."""
    import trimesh

    staging = tmp_path / "staging"
    cls = staging / "bottle-gen"
    cls.mkdir(parents=True)
    trimesh.creation.box(extents=(0.5, 1.0, 0.5)).export(str(cls / "a.glb"))
    trimesh.creation.icosphere(subdivisions=1, radius=0.4).export(str(cls / "b.glb"))
    (cls / "ignore.txt").write_text("junk", encoding="utf-8")
    return staging


def test_discover_staging_subdirs(tmp_path: Path) -> None:
    """Each staging subdir becomes a class; non-mesh files are ignored."""
    staging = _make_staging(tmp_path)
    classes = discover_staging(staging)
    assert len(classes) == 1
    assert classes[0].label == "bottle-gen"
    assert [f.name for f in classes[0].files] == ["a.glb", "b.glb"]


def test_discover_staging_loose_requires_class_name(tmp_path: Path) -> None:
    """Loose mesh files need --class-name to be grouped."""
    import trimesh

    staging = _make_staging(tmp_path)
    trimesh.creation.box().export(str(staging / "loose.glb"))
    with pytest.raises(ValueError, match="class-name"):
        discover_staging(staging)
    classes = discover_staging(staging, class_name="extra")
    assert [c.label for c in classes] == ["bottle-gen", "extra"]
    assert classes[1].files[0].name == "loose.glb"


def test_discover_staging_errors(tmp_path: Path) -> None:
    """Missing dir and empty staging raise errors."""
    with pytest.raises(FileNotFoundError):
        discover_staging(tmp_path / "nope")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no mesh files"):
        discover_staging(empty)


def test_parse_smoke_output() -> None:
    """Smoke stdout lines parse into per-instance results."""
    text = (
        "class 121_x: instances [0, 1]\n"
        "instance 0: z=0.0359 finite=True -> OK\n"
        "instance 1: z=nan finite=False -> FAIL\n"
        "SUCCESS\n"
    )
    out = parse_smoke_output(text)
    assert out[0] == {"z": 0.0359, "finite": True, "ok": True}
    assert out[1]["ok"] is False
    assert parse_smoke_output("no matching lines") == {}


def test_run_import_stage(tmp_path: Path) -> None:
    """Import stage converts every staged class and reports per class."""
    staging = discover_staging(_make_staging(tmp_path))
    objects_dir = tmp_path / "objects"
    report = run_import_stage(staging, objects_dir, target_size=0.1)
    info = report["bottle-gen"]
    assert info["status"] == "ok"
    assert info["class_name"] == "001_bottle-gen"
    assert info["instances"] == [0, 1]
    lib = RoboTwinObjectLibrary(objects_dir)
    assert lib.instance_count("001_bottle-gen") == 2
    inst = lib.get_instance("001_bottle-gen", 0)
    assert max(inst.scaled_extents) == pytest.approx(0.1, rel=1e-3)


def test_write_report(tmp_path: Path) -> None:
    """Report JSON + MD are written with per-class stage rows."""
    report = {
        "created_at": "2026-08-06T00:00:00+00:00",
        "staging_dir": "s",
        "objects_dir": "o",
        "classes": [
            {
                "label": "bottle-gen",
                "import": {
                    "status": "ok",
                    "class_name": "121_bottle-gen",
                    "instances": [0, 1],
                },
                "smoke": {"status": "ok"},
                "grasp": {"status": "success"},
            }
        ],
        "summary": {
            "classes": 1,
            "imported_instances": 2,
            "smoke_ok": 1,
            "grasp_success": 1,
        },
    }
    write_report(report, tmp_path)
    payload = json.loads((tmp_path / "report.json").read_text("utf-8"))
    assert payload["summary"]["imported_instances"] == 2
    md = (tmp_path / "report.md").read_text("utf-8")
    assert "121_bottle-gen" in md
    assert "| bottle-gen | 121_bottle-gen | 2 | ok | success |" in md


def _workflow_args(tmp_path: Path, staging: Path, **overrides) -> argparse.Namespace:
    defaults = dict(
        staging_dir=staging,
        objects_dir=tmp_path / "objects",
        out=tmp_path / "out",
        class_name=None,
        target_size=0.12,
        max_faces=5000,
        z_up=False,
        skip_smoke=True,
        smoke_timeout=60.0,
        grasp=False,
        planner="auto",
        record=False,
        episodes_per_class=1,
        jitter_xy=0.0,
        chunk_size=12,
        seed=0,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_run_workflow_import_only(tmp_path: Path) -> None:
    """End-to-end (smoke/grasp skipped): import -> report files."""
    staging = _make_staging(tmp_path)
    args = _workflow_args(tmp_path, staging)
    report = run_workflow(args)
    assert report["summary"]["classes"] == 1
    assert report["summary"]["imported_instances"] == 2
    assert report["classes"][0]["smoke"]["status"] == "skipped"
    assert report["classes"][0]["grasp"]["status"] == "skipped"
    assert (args.out / "report.json").is_file()
    assert (args.out / "report.md").is_file()
    lib = RoboTwinObjectLibrary(args.objects_dir)
    assert lib.list_classes() == ["001_bottle-gen"]
