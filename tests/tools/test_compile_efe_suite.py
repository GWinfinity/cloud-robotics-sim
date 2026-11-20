"""Tests for the EFE suite compiler (tools/compile_efe_suite.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloud_robotics_sim.core.task_loader import load_task_spec  # noqa: E402
from tools.compile_efe_suite import (  # noqa: E402
    build_outputs,
    check_against_inventory,
    check_outputs,
    main,
    parse_mjcf_objects,
    parse_mjcf_sites,
    render_suite_yaml,
    render_task_yaml,
    resolve_metaworldplus,
    write_outputs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestMjcfParsing:
    """The MetaWorld resolver primitives parse env MJCFs."""

    def test_free_bodies_are_objects(self, tmp_path: Path) -> None:
        xml = tmp_path / "env.xml"
        xml.write_text(
            """
<mujoco>
  <worldbody>
    <body name="obj" pos="0 0.6 0.02">
      <joint name="objjoint" type="free"/>
      <geom name="g" type="cylinder" size="0.02 0.02" rgba="1 0 0 1"/>
    </body>
    <body name="fixed_fixture" pos="1 0 0">
      <geom name="f" type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
""",
            encoding="utf-8",
        )
        objects = parse_mjcf_objects(xml)
        assert list(objects) == ["obj"]
        assert objects["obj"]["geom_type"] == "cylinder"
        assert objects["obj"]["pos"] == [0.0, 0.6, 0.02]

    def test_sites_become_targets(self, tmp_path: Path) -> None:
        xml = tmp_path / "env.xml"
        xml.write_text(
            """
<mujoco><worldbody>
  <body name="goal"><site name="goal" pos="0.1 0.8 0.1" size="0.02"/></body>
</worldbody></mujoco>
""",
            encoding="utf-8",
        )
        assert parse_mjcf_sites(xml)["goal"] == [0.1, 0.8, 0.1]


RECORD = {
    "family": "metaworld/push",
    "goal": "Push the puck to the target",
    "robot": "sawyer",
    "scene": "efe_tabletop",
    "task_type": "efe_static",
    "assets_task_relevant": [
        {
            "name": "obj",
            "shape_type": "cylinder",
            "size": [0.02, 0.02, 0.02],
            "position": [0.0, 0.6, 0.02],
            "color": [1.0, 0.0, 0.0, 1.0],
            "mass": 0.75,
            "static": False,
            "friction": 1.0,
        }
    ],
    "success": {
        "type": "distance_threshold",
        "params": {
            "object": "obj",
            "target_position": [0.1, 0.8, 0.02],
            "threshold": 0.05,
        },
    },
    "max_episode_steps": 500,
    "efe": {"fidelity": "spec"},
}


class TestRendering:
    """Compiled YAMLs parse as schema v1 task specs."""

    def test_rendered_yaml_loads(self, tmp_path: Path) -> None:
        text = render_task_yaml("metaworldplus", "metaworldplus-push-i00", RECORD)
        path = tmp_path / "task.yaml"
        path.write_text(text, encoding="utf-8")
        spec = load_task_spec(path)
        assert spec.name == "metaworldplus-push-i00"
        assert spec.robot_name == "sawyer"
        assert spec.scene_name == "efe_tabletop"
        assert spec.task_name == "efe_static"

    def test_rendered_yaml_uses_new_condition_types(self, tmp_path: Path) -> None:
        record = dict(RECORD)
        record["success"] = {
            "type": "in_region",
            "params": {"object": "obj", "region": [0, 0.2, 0.4, 0.6, 0.0, 0.1]},
        }
        path = tmp_path / "task.yaml"
        path.write_text(
            render_task_yaml("metaworldplus", "t", record), encoding="utf-8"
        )
        spec = load_task_spec(path)
        assert (
            spec.success_condition().evaluate(_scene_with_obj((0.1, 0.5, 0.05)), None)
            is True
        )

    def test_suite_yaml_references_all_tasks(self) -> None:
        text = render_suite_yaml("metaworldplus", ["a", "b"])
        data = yaml.safe_load(text)
        assert data["name"] == "efe-metaworldplus"
        assert len(data["tasks"]) == 2


def _scene_with_obj(pos):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import numpy as np

    entity = MagicMock()
    entity.get_pos.return_value = np.asarray(pos, dtype=float)
    return SimpleNamespace(entities={"obj": entity})


def _write_kb(root: Path, suite: str, names: list[str]) -> Path:
    kb = {
        "suite": suite,
        "robot": "sawyer",
        "scene": "efe_tabletop",
        "tasks": {name: dict(RECORD, family=f"metaworld/{name}") for name in names},
    }
    path = root / "data" / "efe" / "compile" / f"{suite}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(kb, sort_keys=False), encoding="utf-8")
    return path


def _write_inventory(root: Path, suite: str, names: list[str]) -> Path:
    inv = {
        "suite": suite,
        "n_tasks": len(names),
        "n_task_dirs": 2 * len(names),
        "tasks": [
            {
                "name": name,
                "family": f"metaworld/{name}",
                "robot": "sawyer",
                "simulator": "mujoco",
                "modes": {"privileged": {}, "standard": {}},
            }
            for name in names
        ],
    }
    path = root / "data" / "efe" / "tasks" / f"{suite}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inv), encoding="utf-8")
    return path


def _write_exclusions(root: Path, suite: str, names: list[str]) -> Path:
    manifest = {
        "version": 1,
        "policy": {"decision": "exclude", "gate_relaxed": False},
        "suites": [
            {
                "suite": suite,
                "scope": "tasks",
                "kinds": ["quadruped"],
                "robots": ["unitree-go1"],
                "tasks": names,
                "reason": "test exclusion",
            }
        ],
    }
    path = root / "data" / "efe" / "exclusions.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


class TestResolver:
    """The metaworld+ resolver turns predicate table + MJCF into a KB."""

    ENV_XML = """
<mujoco>
  <worldbody>
    <body name="nut" pos="0.1 0.6 0.02">
      <joint name="nutjoint" type="free"/>
      <geom name="g" type="cylinder" size="0.015 0.015 0.01" rgba="1 0 0 1"/>
    </body>
    <body name="goal_marker">
      <site name="goal" pos="0.12 0.62 0.11" size="0.01"/>
    </body>
  </worldbody>
</mujoco>
"""

    def test_resolve_pending_and_from_site(self, tmp_path, monkeypatch) -> None:
        xml_dir = tmp_path / "sawyer_xyz"
        xml_dir.mkdir(parents=True)
        (xml_dir / "sawyer_assembly.xml").write_text(self.ENV_XML, encoding="utf-8")
        monkeypatch.setattr("tools.compile_efe_suite._mw_xml_dir", lambda: tmp_path)
        predicates = {
            "envs": {
                "assembly": {
                    "xml": "sawyer_assembly.xml",
                    "object": "nut",
                    "goal_site": "goal",
                    "goal": "Place the nut onto the peg",
                    "success": {
                        "type": "axis_window",
                        "params": {
                            "object": "nut",
                            "target_position": "from_site",
                            "windows": {
                                "x": [-0.02, 0.02],
                                "y": [-0.02, 0.02],
                                "z": [None, 0.0],
                            },
                        },
                    },
                },
                "dial-turn": {"pending": True},
            }
        }
        kb = resolve_metaworldplus(predicates)
        assert kb["pending_tasks"] == ["metaworldplus-dial-turn-i00"]
        record = kb["tasks"]["metaworldplus-assembly-i00"]
        params = record["success"]["params"]
        assert params["target_position"] == [0.12, 0.62, 0.11]
        assert record["assets_task_relevant"][0]["name"] == "nut"
        assert record["assets_task_relevant"][0]["position"] == [0.1, 0.6, 0.02]


class TestInventoryGate:
    """KB and inventory must cover each other exactly."""

    def test_missing_kb_entry_is_an_error(self, tmp_path: Path, monkeypatch) -> None:
        suite = "metaworldplus"
        _write_inventory(tmp_path, suite, ["metaworldplus-a-i00"])
        kb_path = _write_kb(tmp_path, suite, [])
        monkeypatch.setattr(
            "tools.compile_efe_suite.DATA_EFE", tmp_path / "data" / "efe"
        )
        kb = yaml.safe_load(kb_path.read_text(encoding="utf-8"))
        errors = check_against_inventory(suite, kb)
        assert any("metaworldplus-a-i00" in e for e in errors)

    def test_roundtrip_check_passes_then_fails(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        suite = "metaworldplus"
        names = ["metaworldplus-a-i00", "metaworldplus-b-i00"]
        _write_inventory(tmp_path, suite, names)
        _write_kb(tmp_path, suite, names)
        monkeypatch.setattr(
            "tools.compile_efe_suite.COMPILE_DIR",
            tmp_path / "data" / "efe" / "compile",
        )
        monkeypatch.setattr(
            "tools.compile_efe_suite.DATA_EFE", tmp_path / "data" / "efe"
        )
        monkeypatch.setattr(
            "tools.compile_efe_suite.CONFIGS_OUT",
            tmp_path / "configs" / "tasks" / "efe",
        )
        monkeypatch.setattr(
            "tools.compile_efe_suite.SUITES_OUT", tmp_path / "data" / "suites"
        )
        outputs = build_outputs(suite)
        write_outputs(outputs)
        assert check_outputs(outputs) == []
        assert main(["--suite", suite, "--check"]) == 0
        # Tamper: a hand edit is caught by the gate.
        task_file = tmp_path / "configs" / "tasks" / "efe" / suite / f"{names[0]}.yaml"
        task_file.write_text(
            task_file.read_text(encoding="utf-8") + "# edit\n", encoding="utf-8"
        )
        assert main(["--suite", suite, "--check"]) == 1

    def test_excluded_tasks_are_inventory_explained_but_not_executable(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        suite = "mujoco-playground"
        names = ["arm-a", "arm-pending", "quadruped-excluded"]
        _write_inventory(tmp_path, suite, names)
        _write_exclusions(tmp_path, suite, ["quadruped-excluded"])
        kb_path = _write_kb(tmp_path, suite, ["arm-a"])
        kb = yaml.safe_load(kb_path.read_text(encoding="utf-8"))
        kb["pending_tasks"] = ["arm-pending"]
        kb_path.write_text(yaml.safe_dump(kb, sort_keys=False), encoding="utf-8")
        monkeypatch.setattr(
            "tools.compile_efe_suite.DATA_EFE", tmp_path / "data" / "efe"
        )
        monkeypatch.setattr(
            "tools.compile_efe_suite.COMPILE_DIR",
            tmp_path / "data" / "efe" / "compile",
        )
        monkeypatch.setattr(
            "tools.compile_efe_suite.CONFIGS_OUT",
            tmp_path / "configs" / "tasks" / "efe",
        )
        monkeypatch.setattr(
            "tools.compile_efe_suite.SUITES_OUT", tmp_path / "data" / "suites"
        )
        assert check_against_inventory(suite, kb) == []
        outputs = build_outputs(suite)
        rendered = "\n".join(outputs.values())
        assert "arm-a" in rendered
        assert "arm-pending" not in rendered
        assert "quadruped-excluded" not in rendered
