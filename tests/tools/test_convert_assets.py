"""Tests for the RoboTwin asset conversion toolchain (tools/convert_assets.py)."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.convert_assets import (  # noqa: E402
    check_fixed_base,
    convert_assets,
    expand_mimic_joints,
    find_mesh_references,
    find_missing_inertial,
    find_root_link,
    main,
    parse_urdf,
    smoke_test_meshes,
)

ARM_WITH_MIMIC = """<?xml version="1.0"?>
<robot name="gripper_bot">
  <link name="base_link"/>
  <link name="finger_left">
    <visual><geometry><box size="0.01 0.01 0.05"/></geometry></visual>
    <inertial><mass value="0.05"/></inertial>
  </link>
  <link name="finger_right">
    <visual><geometry><box size="0.01 0.01 0.05"/></geometry></visual>
  </link>
  <joint name="finger_left_joint" type="prismatic">
    <parent link="base_link"/><child link="finger_left"/>
    <axis xyz="1 0 0"/><limit lower="0" upper="0.04" effort="10" velocity="1"/>
  </joint>
  <joint name="finger_right_joint" type="prismatic">
    <parent link="base_link"/><child link="finger_right"/>
    <axis xyz="1 0 0"/><limit lower="0" upper="0.04" effort="10" velocity="1"/>
    <mimic joint="finger_left_joint" multiplier="-1" offset="0.04"/>
  </joint>
</robot>
"""

FLOATING_ROBOT = """<?xml version="1.0"?>
<robot name="floater">
  <link name="base_link"/>
  <link name="body"><inertial><mass value="1"/></inertial></link>
  <joint name="root" type="floating">
    <parent link="base_link"/><child link="body"/>
  </joint>
</robot>
"""


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture()
def asset_tree(tmp_path: Path) -> Path:
    """Synthetic RoboTwin-style asset tree."""
    src = tmp_path / "assets"
    _write(src / "embodiments" / "bot" / "robot.urdf", ARM_WITH_MIMIC)
    _write(src / "embodiments" / "floater" / "robot.urdf", FLOATING_ROBOT)
    _write(src / "robotwin_od" / "002_apple" / "annotations" / "keypoints.json", "{}")
    return src


class TestUrdfChecks:
    """Individual URDF checklist functions (doc section 4.1)."""

    def test_expand_mimic_joints(self, tmp_path: Path) -> None:
        tree = parse_urdf(_write(tmp_path / "robot.urdf", ARM_WITH_MIMIC))
        mapping = expand_mimic_joints(tree)

        assert mapping == {
            "finger_right_joint": {
                "master": "finger_left_joint",
                "multiplier": -1.0,
                "offset": 0.04,
            }
        }
        # The mimic element must be gone from the rewritten tree.
        assert not list(tree.getroot().iter("mimic"))
        # The slave joint itself survives as a plain joint.
        names = [j.get("name") for j in tree.getroot().iter("joint")]
        assert "finger_right_joint" in names

    def test_expand_mimic_defaults(self, tmp_path: Path) -> None:
        urdf = ARM_WITH_MIMIC.replace(' multiplier="-1" offset="0.04"', "")
        tree = parse_urdf(_write(tmp_path / "robot.urdf", urdf))
        mapping = expand_mimic_joints(tree)
        assert mapping["finger_right_joint"]["multiplier"] == 1.0
        assert mapping["finger_right_joint"]["offset"] == 0.0

    def test_check_fixed_base(self, tmp_path: Path) -> None:
        fixed = parse_urdf(_write(tmp_path / "fixed.urdf", ARM_WITH_MIMIC))
        floating = parse_urdf(_write(tmp_path / "float.urdf", FLOATING_ROBOT))
        assert check_fixed_base(fixed) is True
        assert check_fixed_base(floating) is False

    def test_find_root_link(self, tmp_path: Path) -> None:
        tree = parse_urdf(_write(tmp_path / "robot.urdf", ARM_WITH_MIMIC))
        assert find_root_link(tree) == "base_link"

    def test_find_missing_inertial(self, tmp_path: Path) -> None:
        tree = parse_urdf(_write(tmp_path / "robot.urdf", ARM_WITH_MIMIC))
        # finger_right has a visual but no <inertial>; base_link is the root.
        assert find_missing_inertial(tree) == ["finger_right"]

    def test_find_mesh_references(self, tmp_path: Path) -> None:
        urdf = """<robot name="m"><link name="l">
        <visual><geometry><mesh filename="meshes/a.stl"/></geometry></visual>
        <collision><geometry><mesh filename="package://pkg/meshes/b.stl"/></geometry></collision>
        </link></robot>"""
        tree = parse_urdf(_write(tmp_path / "robot.urdf", urdf))
        refs = find_mesh_references(tree)
        assert refs == ["meshes/a.stl", "package://pkg/meshes/b.stl"]


class TestConvertAssets:
    """End-to-end asset tree conversion."""

    def test_end_to_end(self, asset_tree: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "assets_genesis"
        report = convert_assets(asset_tree, out_dir)

        assert report["n_assets"] == 2
        report_path = out_dir / "conversion_report.json"
        assert report_path.exists()
        persisted = json.loads(report_path.read_text(encoding="utf-8"))
        assert persisted["n_assets"] == 2

        # Converted URDFs contain no <mimic> anymore.
        bot_urdf = out_dir / "embodiments" / "bot" / "robot.urdf"
        assert "<mimic" not in bot_urdf.read_text(encoding="utf-8")

        # Report entries capture the checklist facts.
        bot = next(a for a in report["assets"] if "bot" in a["urdf"])
        floater = next(a for a in report["assets"] if "floater" in a["urdf"])
        assert bot["fixed_base"] is True
        assert bot["mimic_map"]["finger_right_joint"]["master"] == ("finger_left_joint")
        assert bot["missing_inertial"] == ["finger_right"]
        assert floater["fixed_base"] is False
        assert bot["load_ok"] is None  # no --smoke-test

        # Annotations are copied verbatim (section 4.2).
        assert (
            out_dir / "robotwin_od" / "002_apple" / "annotations" / "keypoints.json"
        ).exists()

    def test_missing_mesh_reported(self, tmp_path: Path) -> None:
        src = tmp_path / "assets"
        _write(
            src / "obj" / "robot.urdf",
            """<robot name="m"><link name="l">
            <visual><geometry><mesh filename="meshes/missing.stl"/></geometry></visual>
            <inertial><mass value="1"/></inertial></link></robot>""",
        )
        report = convert_assets(src, tmp_path / "out")
        assert report["assets"][0]["missing_meshes"] == ["meshes/missing.stl"]

    def test_existing_mesh_resolves(self, tmp_path: Path) -> None:
        src = tmp_path / "assets"
        _write(
            src / "obj" / "robot.urdf",
            """<robot name="m"><link name="l">
            <visual><geometry><mesh filename="meshes/ok.stl"/></geometry></visual>
            <inertial><mass value="1"/></inertial></link></robot>""",
        )
        _write(src / "obj" / "meshes" / "ok.stl", "solid ok\nendsolid ok\n")
        report = convert_assets(src, tmp_path / "out")
        assert report["assets"][0]["missing_meshes"] == []

    def test_package_uri_rewritten(self, tmp_path: Path) -> None:
        """ROS package:// URIs resolve via path-tail match (RoboTwin piper)."""
        src = tmp_path / "assets"
        _write(
            src / "piper" / "urdf" / "robot.urdf",
            """<robot name="m"><link name="l">
            <visual><geometry>
            <mesh filename="package://piper_description/meshes/base.stl"/>
            </geometry></visual>
            <collision><geometry>
            <mesh filename="package://piper_description/meshes/gone.stl"/>
            </geometry></collision>
            <inertial><mass value="1"/></inertial></link></robot>""",
        )
        _write(src / "piper" / "meshes" / "base.stl", "solid ok\nendsolid ok\n")
        report = convert_assets(src, tmp_path / "out")
        entry = report["assets"][0]

        assert entry["package_uri_rewrites"] == {
            "package://piper_description/meshes/base.stl": "../meshes/base.stl"
        }
        # The rewritten reference resolves; only the truly missing one is left.
        assert entry["missing_meshes"] == [
            "package://piper_description/meshes/gone.stl"
        ]
        out_text = (tmp_path / "out" / "piper" / "urdf" / "robot.urdf").read_text(
            encoding="utf-8"
        )
        assert "../meshes/base.stl" in out_text
        assert "package://piper_description/meshes/base.stl" not in out_text

    def test_src_dir_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            convert_assets(tmp_path / "nope", tmp_path / "out")

    def test_cli_main(
        self, asset_tree: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main([str(asset_tree), str(tmp_path / "out")])
        assert rc == 0
        assert "Converted 2 URDF asset(s)" in capsys.readouterr().out


def test_report_is_json_serializable(asset_tree: Path, tmp_path: Path) -> None:
    """The conversion report serializes and converted URDFs stay valid XML."""
    report = convert_assets(asset_tree, tmp_path / "out")
    json.dumps(report)  # must not raise
    ET.parse(str(tmp_path / "out" / "embodiments" / "bot" / "robot.urdf"))


class TestMeshSmoke:
    """GLB object-library smoke test sampling (RoboTwin-OD layout)."""

    def test_samples_one_mesh_per_group(self, tmp_path: Path, monkeypatch) -> None:
        src = tmp_path / "objects"
        for cls in ("001_bottle", "002_bowl"):
            for group in ("collision", "visual"):
                for i in range(3):
                    _write(src / cls / group / f"base{i}.glb", "glb")

        loaded: list[str] = []

        def fake_batched(paths, batch_size=16):
            loaded.extend(p.name for p in paths)
            return [{"mesh": str(p), "load_ok": True, "error": None} for p in paths]

        monkeypatch.setattr(
            "tools.convert_assets._smoke_test_meshes_batched", fake_batched
        )
        report = smoke_test_meshes(src, tmp_path / "out")

        # 2 classes x 2 groups, first mesh per group only.
        assert report["n_assets"] == 4
        assert report["n_classes"] == 2
        assert loaded == ["base0.glb"] * 4
        assert not (tmp_path / "out" / "001_bottle").exists()  # no tree copy
        assert (tmp_path / "out" / "conversion_report.json").exists()

    def test_flat_layout_fallback(self, tmp_path: Path, monkeypatch) -> None:
        src = tmp_path / "objects"
        _write(src / "003_apple" / "base0.glb", "glb")
        monkeypatch.setattr(
            "tools.convert_assets._smoke_test_meshes_batched",
            lambda paths, batch_size=16: [
                {"mesh": str(p), "load_ok": True, "error": None} for p in paths
            ],
        )
        report = smoke_test_meshes(src, tmp_path / "out")
        assert report["n_assets"] == 1
