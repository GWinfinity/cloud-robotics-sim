"""Tests for the asset manifest builder (tools/build_asset_manifest.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.build_asset_manifest import (  # noqa: E402
    UNREGISTERED,
    UPSTREAM_LICENSE,
    build_manifest,
    class_kind,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_OBJECTS_DIR = REPO_ROOT / "assets" / "robotwin" / "objects" / "objects"


def _make_glb_class(objects_dir: Path, name: str, metas: list[dict]) -> Path:
    class_dir = objects_dir / name
    class_dir.mkdir(parents=True)
    for i, meta in enumerate(metas):
        (class_dir / f"model_data{i}.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
    return class_dir


def test_class_kind(tmp_path: Path) -> None:
    """Glb when model_data*.json exists, urdf otherwise."""
    glb = _make_glb_class(tmp_path, "001_bottle", [{"center": [0, 0, 0]}])
    assert class_kind(glb) == "glb"
    urdf = tmp_path / "009_kettle"
    (urdf / "102730").mkdir(parents=True)
    (urdf / "102730" / "mobility.urdf").write_text("<robot/>", encoding="utf-8")
    assert class_kind(urdf) == "urdf"


def test_build_manifest_licenses(tmp_path: Path) -> None:
    """Upstream instances fall back to MIT; unlicensed imports are flagged."""
    _make_glb_class(
        tmp_path,
        "001_bottle",
        [
            {"center": [0, 0, 0]},  # shipped upstream (no generator)
            {
                "generator": "tools/import_generated_objects.py",
                "source": "style_a.glb",
                "license": "CC0-1.0",
            },
            {
                "generator": "tools/import_generated_objects.py",
                "source": "style_b.glb",
            },  # imported, no license -> UNREGISTERED
        ],
    )
    manifest = build_manifest(tmp_path)
    assert manifest["totals"]["classes"] == 1
    assert manifest["totals"]["instances"] == 3
    assert manifest["totals"]["imported_instances"] == 2
    assert manifest["totals"]["unregistered_instances"] == 1

    records = manifest["classes"][0]["instance_records"]
    assert records[0]["license"] == UPSTREAM_LICENSE
    assert records[1]["license"] == "CC0-1.0"
    assert records[2]["license"] == UNREGISTERED
    assert any("001_bottle" in w for w in manifest["warnings"])


def test_build_manifest_disallowed_license_warns(tmp_path: Path) -> None:
    """Imported assets with a license outside the allowlist get a warning."""
    _make_glb_class(
        tmp_path,
        "002_thing",
        [{"generator": "x", "license": "GPL-3.0"}],
    )
    manifest = build_manifest(tmp_path)
    assert manifest["totals"]["unregistered_instances"] == 0
    assert any("allowlist" in w for w in manifest["warnings"])


def test_build_manifest_urdf_class(tmp_path: Path) -> None:
    """PartNet-Mobility classes contribute one upstream record per instance."""
    class_dir = tmp_path / "009_kettle"
    for inst in ("102730", "102738"):
        (class_dir / inst).mkdir(parents=True)
        (class_dir / inst / "mobility.urdf").write_text("<robot/>", "utf-8")
    manifest = build_manifest(tmp_path)
    entry = manifest["classes"][0]
    assert entry["kind"] == "urdf"
    assert entry["instances"] == 2
    assert entry["licenses"] == [UPSTREAM_LICENSE]


def test_main_writes_manifest_and_check_gate(tmp_path: Path) -> None:
    """--out writes JSON; --check fails only on UNREGISTERED imports."""
    _make_glb_class(tmp_path, "001_a", [{"center": [0, 0, 0]}])
    out = tmp_path / "manifest.json"
    assert main(["--objects-dir", str(tmp_path), "--out", str(out), "--check"]) == 0
    assert json.loads(out.read_text("utf-8"))["totals"]["classes"] == 1

    _make_glb_class(tmp_path, "002_b", [{"generator": "x"}])  # no license
    assert main(["--objects-dir", str(tmp_path), "--out", str(out), "--check"]) == 1


@pytest.mark.skipif(not REAL_OBJECTS_DIR.is_dir(), reason="RoboTwin assets not present")
def test_real_library_manifest() -> None:
    """The shipped library scans cleanly: all classes upstream-licensed."""
    manifest = build_manifest(REAL_OBJECTS_DIR)
    assert manifest["totals"]["classes"] >= 100
    assert manifest["totals"]["instances"] >= 100
    # shipped assets carry no generator field -> no UNREGISTERED entries
    assert manifest["totals"]["unregistered_instances"] == 0
    kinds = {c["kind"] for c in manifest["classes"]}
    assert kinds == {"glb", "urdf"}
