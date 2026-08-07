"""Tests for the Poly Pizza fetch/import tool (tools/fetch_polypizza_assets.py).

Parsing is tested offline against trimmed HTML fixtures captured from real
poly.pizza pages; no network access is required. The end-to-end import path
is covered by monkeypatching the downloader and by building a real GLB with
trimesh into a tmp object library.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.fetch_polypizza_assets import (  # noqa: E402
    IMPORTABLE_LICENSES,
    main,
    map_license_url,
    parse_model_page,
    parse_search_results,
)

# --- fixtures: trimmed real-world HTML snippets ----------------------------

SEARCH_HTML = """
<div class="MuiGrid-root">
<a href="/m/B5nWfdzHzO"><div class="MuiCardMedia-root" title="Hair Dryer" loading="lazy"></div></a>
<a href="/m/8yTYklZdPCM"><div class="MuiCardMedia-root" title="Hair iron" loading="lazy"></div></a>
<a href="/m/B5nWfdzHzO"><div class="MuiCardMedia-root" title="Hair Dryer" loading="lazy"></div></a>
</div>
"""

MODEL_PAGE_CC0 = """
<html><head>
<meta property="og:image" content="https://static.poly.pizza/e406c510-e426-48d9-9e59-0f70f61445a1.jpg"/>
</head><body>
<script>self.__next_f.push({"ID":"5a1","PublicID":"B5nWfdzHzO","Likes":2,"Tris":480,
"Title":"Hair Dryer","Creator":{"Username":"CreativeTrio","DPURL":""}});</script>
<link rel="preload" href="https://static.poly.pizza/e406c510-e426-48d9-9e59-0f70f61445a1.glb"/>
<a href="https://creativecommons.org/publicdomain/zero/1.0/" target="blank">CC0</a>
</body></html>
"""

MODEL_PAGE_CC_BY = """
<html><body>
<script>{"PublicID":"abcDEF123","Tris":1200,"Title":"Spatula",
"Creator":{"Username":"Kenney"}}</script>
<img src="https://static.poly.pizza/11111111-2222-3333-4444-555555555555.glb"/>
<a href="https://creativecommons.org/licenses/by/4.0/" target="blank">CC-BY</a>
</body></html>
"""

MODEL_PAGE_NC = MODEL_PAGE_CC_BY.replace("licenses/by/4.0", "licenses/by-nc/4.0")


def test_parse_search_results_dedupes_and_keeps_order() -> None:
    """Cards map to (public_id, title); duplicates collapse, order preserved."""
    assert parse_search_results(SEARCH_HTML) == [
        ("B5nWfdzHzO", "Hair Dryer"),
        ("8yTYklZdPCM", "Hair iron"),
    ]


def test_map_license_url() -> None:
    """creativecommons.org paths map to SPDX ids; unknown paths raise."""
    assert map_license_url("publicdomain/zero/1.0") == "CC0-1.0"
    assert map_license_url("licenses/by/4.0") == "CC-BY-4.0"
    assert map_license_url("licenses/by/3.0") == "CC-BY-3.0"
    assert "CC-BY-3.0" in IMPORTABLE_LICENSES  # Google Poly archive models
    assert map_license_url("licenses/by-nc/4.0") == "CC-BY-NC-4.0"
    with pytest.raises(ValueError, match="unrecognized"):
        map_license_url("licenses/gpl/2.0")


def test_parse_model_page_cc0() -> None:
    """CC0 model pages yield full metadata incl. GLB download URL."""
    info = parse_model_page(MODEL_PAGE_CC0, "https://poly.pizza/m/B5nWfdzHzO")
    assert info.public_id == "B5nWfdzHzO"
    assert info.title == "Hair Dryer"
    assert info.author == "CreativeTrio"
    assert info.tris == 480
    assert info.license_spdx == "CC0-1.0"
    assert info.license_spdx in IMPORTABLE_LICENSES
    assert info.glb_url == (
        "https://static.poly.pizza/e406c510-e426-48d9-9e59-0f70f61445a1.glb"
    )


def test_parse_model_page_cc_by() -> None:
    """CC-BY models keep the author for attribution and stay importable."""
    info = parse_model_page(MODEL_PAGE_CC_BY, "https://poly.pizza/m/abcDEF123")
    assert info.license_spdx == "CC-BY-4.0"
    assert info.author == "Kenney"
    assert info.license_spdx in IMPORTABLE_LICENSES


def test_parse_model_page_nc_not_importable() -> None:
    """Non-commercial licenses parse but are excluded from the whitelist."""
    info = parse_model_page(MODEL_PAGE_NC, "https://poly.pizza/m/abcDEF123")
    assert info.license_spdx == "CC-BY-NC-4.0"
    assert info.license_spdx not in IMPORTABLE_LICENSES


def test_parse_model_page_rejects_foreign_url() -> None:
    """Non poly.pizza URLs fail before any scraping happens."""
    with pytest.raises(ValueError, match="not a Poly Pizza"):
        parse_model_page(MODEL_PAGE_CC0, "https://example.com/m/B5nWfdzHzO")


def test_parse_model_page_missing_blob() -> None:
    """A page without the model's JSON blob raises a clear error."""
    with pytest.raises(ValueError, match="not found"):
        parse_model_page("<html>no blob</html>", "https://poly.pizza/m/B5nWfdzHzO")


def test_fetch_imports_with_provenance(tmp_path: Path, monkeypatch) -> None:
    """End-to-end: mocked network -> real GLB import with license provenance."""
    import trimesh

    from tools import fetch_polypizza_assets as fpa

    glb_path = tmp_path / "src.glb"
    trimesh.creation.box(extents=(1.0, 2.0, 3.0)).export(str(glb_path))
    glb_bytes = glb_path.read_bytes()

    def fake_fetch(url: str, timeout: float = 30.0) -> bytes:
        if "poly.pizza/m/" in url:
            return MODEL_PAGE_CC0.encode()
        assert url.endswith(".glb")
        return glb_bytes

    monkeypatch.setattr(fpa, "fetch_url", fake_fetch)
    objects_dir = tmp_path / "objects"
    rc = main(
        [
            "fetch",
            "https://poly.pizza/m/B5nWfdzHzO",
            "--class-name",
            "148_hairdryer_set",
            "--objects-dir",
            str(objects_dir),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    assert rc == 0
    meta = json.loads(
        (objects_dir / "148_hairdryer_set" / "model_data0.json").read_text()
    )
    assert meta["license"] == "CC0-1.0"
    assert meta["source_url"] == "https://poly.pizza/m/B5nWfdzHzO"
    assert meta["author"] == "CreativeTrio"
    assert meta["source_site"] == "poly.pizza"


def test_fetch_refuses_nc_license(tmp_path: Path, monkeypatch) -> None:
    """NC-licensed models abort with rc=1 and import nothing."""
    from tools import fetch_polypizza_assets as fpa

    monkeypatch.setattr(
        fpa, "fetch_url", lambda url, timeout=30.0: MODEL_PAGE_NC.encode()
    )
    objects_dir = tmp_path / "objects"
    rc = main(
        [
            "fetch",
            "abcDEF123",
            "--class-name",
            "149_kitchen_utensil_set",
            "--objects-dir",
            str(objects_dir),
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    assert rc == 1
    assert not (objects_dir / "149_kitchen_utensil_set").exists()
