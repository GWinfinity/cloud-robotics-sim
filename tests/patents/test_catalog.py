"""Tests for the patents catalog YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

CATALOG_PATH = (
    Path(__file__).parent.parent.parent
    / "src"
    / "cloud_robotics_sim"
    / "patents"
    / "data"
    / "patents_catalog.yaml"
)

REQUIRED_KEYS = {
    "id",
    "number",
    "title",
    "inventors",
    "grant_date",
    "breakthrough",
    "module",
    "has_simulation",
}


def test_catalog_exists_and_loads() -> None:
    """The catalog file exists and is valid YAML."""
    assert CATALOG_PATH.exists(), f"Catalog not found at {CATALOG_PATH}"
    with CATALOG_PATH.open("r", encoding="utf-8") as f:
        catalog = yaml.safe_load(f)
    assert "patents" in catalog
    assert isinstance(catalog["patents"], list)


def test_catalog_entries_have_required_keys() -> None:
    """Every catalog entry contains the expected fields."""
    with CATALOG_PATH.open("r", encoding="utf-8") as f:
        catalog = yaml.safe_load(f)

    seen_ids: set[str] = set()
    for entry in catalog["patents"]:
        missing = REQUIRED_KEYS - set(entry.keys())
        assert not missing, f"Missing keys in {entry.get('id')}: {missing}"
        assert isinstance(entry["id"], str)
        assert isinstance(entry["inventors"], list)
        assert isinstance(entry["has_simulation"], bool)
        assert entry["id"] not in seen_ids, f"Duplicate patent id {entry['id']}"
        seen_ids.add(entry["id"])


def test_catalog_count_matches_registry() -> None:
    """The number of catalog entries matches the global registry."""
    from cloud_robotics_sim.patents import list_patents

    with CATALOG_PATH.open("r", encoding="utf-8") as f:
        catalog = yaml.safe_load(f)

    catalog_ids = {entry["id"] for entry in catalog["patents"]}
    registry_ids = set(list_patents())
    assert catalog_ids == registry_ids


def test_pilot_marked_as_simulated() -> None:
    """The Wright Flyer is marked as having a simulation."""
    with CATALOG_PATH.open("r", encoding="utf-8") as f:
        catalog = yaml.safe_load(f)

    wright = next((p for p in catalog["patents"] if p["id"] == "US821393"), None)
    assert wright is not None
    assert wright["has_simulation"] is True
