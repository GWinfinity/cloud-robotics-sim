"""Tests for the permanent EFE exclusion boundary.

MyoSuite (musculoskeletal) stays excluded: its tasks never enter executable
artifacts. Playground's quadruped tasks were excluded under the strict
five-kind gate; the user override of 2026-11-20 (embodiment_gate v2 lists
``quadruped`` as an allowed kind) migrated all 13 Playground tasks, so only
MyoSuite remains on the exclusion manifest.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.check_efe_execution_scope import (  # noqa: E402
    EXCLUSIONS,
    GATE_REPORT,
    audit_scope,
    load_exclusions,
    main,
)


def _inputs() -> tuple[dict, dict]:
    manifest = load_exclusions(EXCLUSIONS)
    gate = json.loads(GATE_REPORT.read_text(encoding="utf-8"))
    return manifest, gate


def test_committed_execution_scope_is_clean(capsys) -> None:
    """The manifest matches the gate and no excluded task is executable."""
    manifest, gate = _inputs()
    errors, summary = audit_scope(manifest, gate)
    assert errors == []
    assert summary["excluded_suites"] == ["myosuite"]
    assert summary["excluded_tasks"] == 22
    assert main(["--check"]) == 0
    assert "tasks=22 errors=0" in capsys.readouterr().out


def test_exclusion_boundary_is_exact() -> None:
    """MyoSuite is the only remaining (whole-suite) exclusion."""
    manifest, _ = _inputs()
    suites = {record["suite"]: record for record in manifest["suites"]}
    assert sorted(suites) == ["myosuite"]
    assert suites["myosuite"]["scope"] == "suite"
    assert len(suites["myosuite"]["tasks"]) == 22
    assert all(name.startswith("myosuite-") for name in suites["myosuite"]["tasks"])


def test_gate_relaxation_requires_documented_override() -> None:
    """A relaxed gate is accepted only with the override on record."""
    manifest, gate = _inputs()
    assert manifest["policy"]["gate_relaxed"] is True
    errors, _ = audit_scope(copy.deepcopy(manifest), gate)
    assert errors == []

    tampered = copy.deepcopy(manifest)
    tampered["policy"]["note"] = ""
    errors, _ = audit_scope(tampered, gate)
    assert any("gate_relaxed" in error for error in errors)

    tampered = copy.deepcopy(manifest)
    tampered["policy"]["decision"] = "relax"
    errors, _ = audit_scope(tampered, gate)
    assert any("decision" in error for error in errors)


def test_exclusion_task_drift_is_rejected() -> None:
    """The exclusion list must stay byte-for-byte aligned with the gate."""
    manifest, gate = _inputs()
    tampered = copy.deepcopy(manifest)
    tampered["suites"][0]["tasks"].pop()
    errors, _ = audit_scope(tampered, gate)
    assert any("task drift" in error for error in errors)


def test_excluded_task_cannot_leak_into_executable_artifacts(
    tmp_path: Path,
) -> None:
    """A config, aggregate or pending entry fails the execution-scope gate."""
    manifest, gate = _inputs()
    configs = tmp_path / "configs"
    suites = tmp_path / "suites"
    compile_root = tmp_path / "compile"

    suite = "myosuite"
    excluded = "myosuite-arm-reach-i00"
    config = configs / suite / f"{excluded}.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("version: 1\n", encoding="utf-8")

    aggregate = suites / f"efe_{suite}.yaml"
    aggregate.parent.mkdir(parents=True)
    aggregate.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "name": f"efe-{suite}",
                "tasks": [f"../../configs/{suite}/{excluded}.yaml"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    kb = compile_root / f"{suite}.yaml"
    kb.parent.mkdir(parents=True)
    kb.write_text(
        yaml.safe_dump(
            {
                "suite": suite,
                "tasks": {},
                "pending_tasks": [excluded],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    errors, _ = audit_scope(
        manifest,
        gate,
        configs_root=configs,
        suites_root=suites,
        compile_root=compile_root,
    )
    assert any("excluded task config present" in error for error in errors)
    assert any("aggregate references excluded task" in error for error in errors)
    assert any("executable KB tasks/pending_tasks" in error for error in errors)


def test_excluded_metadata_copied_into_kb_is_rejected(tmp_path: Path) -> None:
    """Excluded task names must stay out of the compile knowledge base."""
    manifest, gate = _inputs()
    compile_root = tmp_path / "compile"
    kb = compile_root / "myosuite.yaml"
    kb.parent.mkdir(parents=True)
    kb.write_text(
        yaml.safe_dump(
            {
                "suite": "myosuite",
                "tasks": {},
                "excluded_tasks": ["myosuite-arm-reach-i00"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    errors, _ = audit_scope(
        manifest,
        gate,
        configs_root=tmp_path / "empty-configs",
        suites_root=tmp_path / "empty-suites",
        compile_root=compile_root,
    )
    assert any(
        "must stay only in data/efe/exclusions.yaml" in error for error in errors
    )
