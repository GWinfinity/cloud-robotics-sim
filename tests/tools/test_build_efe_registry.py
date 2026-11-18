"""Tests for the EFE migration registry toolchain (tools/build_efe_registry.py).

All tests are offline: they consume the committed ``data/efe`` records or
synthetic copies under ``tmp_path``.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.build_efe_registry import (  # noqa: E402
    PAPER,
    build_registry,
    main,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "efe"


def copy_data(tmp_path: Path) -> Path:
    """Copy the committed data/efe records into tmp_path for tampering."""
    dest = tmp_path / "efe"
    shutil.copytree(DATA_DIR / "suites", dest / "suites")
    shutil.copytree(DATA_DIR / "tasks", dest / "tasks")
    shutil.copy(DATA_DIR / "robots.yaml", dest / "robots.yaml")
    return dest


def test_committed_registry_is_clean() -> None:
    """The committed registry reproduces the paper totals."""
    registry, errors = build_registry(DATA_DIR)
    assert errors == []
    assert registry["totals"]["suites"] == PAPER["suites"] + 1  # + humanclaw
    assert registry["totals"]["in_paper_suites"] == PAPER["suites"]
    assert registry["totals"]["tasks"] == PAPER["tasks"] + 55
    assert registry["totals"]["tests"] >= PAPER["tests"]  # upstream drift allowed
    in_paper = [s for s in registry["suites"] if s["efe"]["in_paper"]]
    robots = {r for s in in_paper for r in s["efe"]["robots"]}
    assert len(robots) == PAPER["robots"]


def test_check_main_passes_on_committed_data(capsys) -> None:
    """The main --check entry exits 0 on the committed data."""
    assert main(["--data-dir", str(DATA_DIR), "--check"]) == 0
    out = capsys.readouterr().out
    assert f"in-paper {PAPER['suites']}" in out


def test_task_count_drift_is_an_error(tmp_path: Path) -> None:
    """A desynced task count fails the gate."""
    data_dir = copy_data(tmp_path)
    suite_path = data_dir / "suites" / "robowits.yaml"
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    suite["efe"]["tasks"] += 1  # desync from the paper
    suite_path.write_text(yaml.safe_dump(suite, allow_unicode=True), encoding="utf-8")
    _, errors = build_registry(data_dir)
    assert any("robowits" in e and "tasks" in e for e in errors)
    assert main(["--data-dir", str(data_dir), "--check"]) == 1


def test_unknown_tier_is_an_error(tmp_path: Path) -> None:
    """An unknown tier fails structural validation."""
    data_dir = copy_data(tmp_path)
    suite_path = data_dir / "suites" / "kinder.yaml"
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    suite["project"]["tier"] = "T9"
    suite_path.write_text(yaml.safe_dump(suite, allow_unicode=True), encoding="utf-8")
    _, errors = build_registry(data_dir)
    assert any("kinder" in e and "tier" in e for e in errors)


def test_robot_without_provenance_is_an_error(tmp_path: Path) -> None:
    """A robot missing from robots.yaml fails validation."""
    data_dir = copy_data(tmp_path)
    suite_path = data_dir / "suites" / "kinder.yaml"
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    suite["efe"]["robots"] = ["mystery-bot"]
    suite_path.write_text(yaml.safe_dump(suite, allow_unicode=True), encoding="utf-8")
    _, errors = build_registry(data_dir)
    assert any("mystery-bot" in e for e in errors)


def test_robodojo_upstream_drift_is_warning_not_error() -> None:
    """Upstream-only test-dir drift stays a warning."""
    registry, errors = build_registry(DATA_DIR)
    assert errors == []
    assert any("robodojo" in w for w in registry["warnings"])


def test_registry_json_is_in_sync_with_data(tmp_path: Path) -> None:
    """A fresh build equals the committed registry.json."""
    data_dir = copy_data(tmp_path)
    assert main(["--data-dir", str(data_dir)]) == 0
    registry = json.loads((data_dir / "registry.json").read_text(encoding="utf-8"))
    assert registry["errors"] == []
    committed = json.loads((DATA_DIR / "registry.json").read_text(encoding="utf-8"))
    # Ignore timestamps; the curated payload must match the committed build.
    registry.pop("generated_at")
    committed.pop("generated_at")
    assert registry == committed
