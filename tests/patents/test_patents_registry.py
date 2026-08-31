"""Tests for the patent simulation registry."""

from __future__ import annotations

import pytest

from cloud_robotics_sim.patents import (
    PatentSimConfig,
    create_simulation,
    default_registry,
    list_patents,
)
from cloud_robotics_sim.patents.base import StubPatentSimulation

EXPECTED_PATENTS = [
    "US1647",
    "US3633",
    "US4750",
    "US6469",
    "US174465",
    "US223898",
    "US381968",
    "US586193",
    "US593138",
    "US821393",
    "US1155986",
    "US1781541",
    "US1773980",
    "US2292387",
    "US2495429",
    "US2708656",
    "US2524035",
    "US2981877",
    "US3541541",
    "US3671542",
    "US3923554",
    "US4136359",
]


def test_all_patents_registered() -> None:
    """Every patent in the catalog must be registered."""
    registered = list_patents()
    for patent_id in EXPECTED_PATENTS:
        assert patent_id in registered, f"Missing patent {patent_id}"
    assert len(registered) == len(EXPECTED_PATENTS)


def test_create_stub_simulation() -> None:
    """Creating a stub simulation returns a StubPatentSimulation instance."""
    config = PatentSimConfig(patent_id="US2981877", headless=True)
    sim = create_simulation("US2981877", config=config)
    assert isinstance(sim, StubPatentSimulation)
    assert sim.patent_id == "US2981877"


def test_metadata_available() -> None:
    """Each registered patent has metadata with a title."""
    registry = default_registry()
    for patent_id in EXPECTED_PATENTS:
        metadata = registry.get_metadata(patent_id)
        assert "title" in metadata
        assert isinstance(metadata["title"], str)


def test_unknown_patent_raises() -> None:
    """Requesting an unregistered patent raises KeyError."""
    with pytest.raises(KeyError):
        create_simulation("US000000")
