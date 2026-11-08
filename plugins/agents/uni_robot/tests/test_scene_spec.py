"""SceneSpec 校验与序列化测试（无重依赖）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_CORE_DIR = _TESTS_DIR.parent / "core"
for _dir in (str(_TESTS_DIR), str(_CORE_DIR)):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

from uni_robot_genesis.scene_spec import (  # noqa: E402
    ArmSpec,
    CameraSpec,
    JudgeSpec,
    PoseGoal,
    SceneSpec,
)


class TestValidation:
    """SceneSpec 校验规则。"""

    def test_defaults(self):
        spec = SceneSpec(
            instruction="do something",
            arms=(ArmSpec(),),
            cameras=(CameraSpec(name="c", position=(1, 0, 1), lookat=(0, 0, 0)),),
        )
        assert spec.control.frequency_hz == 20.0
        assert spec.ground is True
        assert spec.judge.enabled is False

    def test_duplicate_camera_names_rejected(self):
        cam = CameraSpec(name="c", position=(1, 0, 1), lookat=(0, 0, 0))
        with pytest.raises(Exception, match="duplicate"):
            SceneSpec(
                instruction="x",
                arms=(ArmSpec(),),
                cameras=(cam, cam),
            )

    def test_object_fixture_name_clash_rejected(self):
        box = dict(size=(0.1, 0.1, 0.1), position=(0, 0, 0))
        from uni_robot_genesis.scene_spec import BoxSpec

        with pytest.raises(Exception, match="duplicate"):
            SceneSpec(
                instruction="x",
                arms=(ArmSpec(),),
                cameras=(CameraSpec(name="c", position=(1, 0, 1), lookat=(0, 0, 0)),),
                fixtures=(BoxSpec(name="same", fixed=True, **box),),
                objects=(BoxSpec(name="same", fixed=False, **box),),
            )

    def test_zero_quaternion_rejected(self):
        with pytest.raises(Exception, match="quaternion"):
            ArmSpec(base_quaternion=(0.0, 0.0, 0.0, 0.0))

    def test_quaternion_normalized(self):
        arm = ArmSpec(base_quaternion=(2.0, 0.0, 0.0, 0.0))
        assert abs(arm.base_quaternion[0] - 1.0) < 1e-12


class TestYaml:
    """YAML 序列化往返与示例文件加载。"""

    def test_round_trip(self, tmp_path):
        spec = SceneSpec(
            instruction="pick and place",
            arms=(
                ArmSpec(name="left"),
                ArmSpec(name="right", base_position=(0.0, 0.6, 0.0)),
            ),
            cameras=(
                CameraSpec(name="head", position=(1, 0, 1.2), lookat=(0, 0, 0.4)),
            ),
            judge=JudgeSpec(
                goals=(PoseGoal(object_name="cube", position=(0.5, 0.0, 0.4)),)
            ),
        )
        path = tmp_path / "spec.yaml"
        spec.to_yaml(str(path))
        loaded = SceneSpec.from_yaml(str(path))
        assert loaded == spec
        assert loaded.judge.enabled is True

    def test_example_file_loads(self):
        example = Path(__file__).resolve().parents[1] / "examples" / "franka_table.yaml"
        spec = SceneSpec.from_yaml(str(example))
        assert spec.arms[0].preset == "franka_panda"
        assert spec.judge.goals[0].object_name == "red_cube"
