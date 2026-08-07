"""Tests for the Objaverse-XL GitHub-subset fetch tool.

The catalog search is tested against a tiny fixture parquet written with
pyarrow; downloads and imports are monkeypatched/offline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.fetch_objaverse_xl_github import (  # noqa: E402
    blob_to_raw_url,
    main,
    map_repo_license,
    search_catalog,
)


@pytest.fixture()
def fixture_parquet(tmp_path: Path) -> Path:
    """A 6-row catalog exercising license mapping / rejection / file types."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = {
        "fileIdentifier": [
            "https://github.com/a/makeup/blob/deadbeef1234/models/makeup_organizer.glb",
            "https://github.com/a/gplstuff/blob/deadbeef1234/models/organizer.glb",
            "https://github.com/a/nolicense/blob/deadbeef1234/models/organizer.glb",
            "https://github.com/a/makeup/blob/deadbeef1234/models/blend_file.blend",
            "https://github.com/b/cc0/blob/cafe1234abcd/props/vanity_box.obj",
            "https://github.com/c/apache/blob/00112233aabb/storage/makeup_case.stl",
        ],
        "license": [
            "MIT License",
            "GNU General Public License v3.0",
            None,
            "MIT License",
            "Creative Commons Zero v1.0 Universal",
            "Apache License 2.0",
        ],
        "fileType": ["glb", "glb", "glb", "blend", "obj", "stl"],
    }
    path = tmp_path / "github.parquet"
    pq.write_table(pa.table(rows), str(path))
    return path


def test_map_repo_license() -> None:
    """SPDX mapping; GPL/NC/None/unmapped all rejected."""
    assert map_repo_license("MIT License") == "MIT"
    assert map_repo_license("Apache License 2.0") == "Apache-2.0"
    assert map_repo_license("Creative Commons Zero v1.0 Universal") == "CC0-1.0"
    assert map_repo_license("GNU General Public License v3.0") is None
    assert map_repo_license("Creative Commons - Attribution - Non-Commercial") is None
    assert map_repo_license(None) is None
    assert map_repo_license("Some Custom License") is None


def test_blob_to_raw_url() -> None:
    """Blob URLs convert to raw.githubusercontent URLs; junk raises."""
    assert (
        blob_to_raw_url("https://github.com/o/r/blob/abc123/a b/model.glb")
        == "https://raw.githubusercontent.com/o/r/abc123/a b/model.glb"
    )
    with pytest.raises(ValueError, match="not a GitHub blob URL"):
        blob_to_raw_url("https://example.com/o/r/blob/abc/model.glb")


def test_search_catalog_filters_license_and_type(fixture_parquet: Path) -> None:
    """Search keeps only importable types + whitelisted licenses."""
    hits = search_catalog(fixture_parquet, "organizer")
    urls = [h.blob_url for h in hits]
    assert any("makeup_organizer.glb" in u for u in urls)
    assert not any("gplstuff" in u or "nolicense" in u for u in urls)
    assert hits[0].license_spdx == "MIT"

    hits = search_catalog(fixture_parquet, "vanity|makeup_case")
    assert {h.license_spdx for h in hits} == {"CC0-1.0", "Apache-2.0"}

    # blend files are not importable types
    assert not search_catalog(fixture_parquet, "blend_file")


def test_search_catalog_full_path_mode(fixture_parquet: Path) -> None:
    """--full-path matches repo segments, name-only mode does not."""
    assert not search_catalog(fixture_parquet, "cc0", name_only=True)
    assert search_catalog(fixture_parquet, "cc0", name_only=False)


def test_fetch_imports_with_provenance(tmp_path: Path, monkeypatch) -> None:
    """End-to-end: mocked raw download -> real import with SPDX provenance."""
    import trimesh

    from tools import fetch_objaverse_xl_github as fxg

    glb_path = tmp_path / "src.glb"
    trimesh.creation.box(extents=(0.2, 0.1, 0.18)).export(str(glb_path))

    def fake_fetch(url: str, timeout: float = 60.0) -> bytes:
        assert url.startswith("https://raw.githubusercontent.com/")
        return glb_path.read_bytes()

    monkeypatch.setattr(fxg, "fetch_url", fake_fetch)
    objects_dir = tmp_path / "objects"
    blob = "https://github.com/a/makeup/blob/deadbeef1234/models/makeup_organizer.glb"
    rc = main(
        [
            "fetch",
            blob,
            "--license",
            "MIT License",
            "--class-name",
            "130_cosmetic_organizer",
            "--objects-dir",
            str(objects_dir),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    assert rc == 0
    meta = json.loads(
        (objects_dir / "130_cosmetic_organizer" / "model_data0.json").read_text()
    )
    assert meta["license"] == "MIT"
    assert meta["source_url"] == blob
    assert meta["source_site"] == "objaverse-xl/github"


def test_fetch_refuses_gpl(tmp_path: Path) -> None:
    """GPL repos abort with rc=1 before any download."""
    rc = main(
        [
            "fetch",
            "https://github.com/a/gpl/blob/deadbeef1234/models/x.glb",
            "--license",
            "GNU General Public License v3.0",
            "--class-name",
            "130_cosmetic_organizer",
            "--objects-dir",
            str(tmp_path / "objects"),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    assert rc == 1
    assert not (tmp_path / "objects").exists()
