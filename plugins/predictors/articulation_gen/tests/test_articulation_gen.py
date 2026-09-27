"""articulation_gen 离线测试（numpy-only，无网络、无 GPU、无 genesis）。

构造合成家具顶点云（基座柜体 + 抽屉 + 柜门），覆盖：
- 基座识别（最大体积）
- 抽屉 → prismatic（轴向 = 外露面法向 +y，行程 = 深度×0.95）
- 柜门 → revolute（竖直铰链、限位 = 碰撞扫描自由角）
- 置信度/复核队列（嵌套部件 → review）
- URDF 导出与结构校验
- batch_annotate 落盘
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from plugins.predictors.articulation_gen import (
    ArticulatedPart,
    JointType,
    batch_annotate,
    estimate_joints,
    result_to_urdf,
    validate_urdf,
)
from plugins.predictors.articulation_gen.core.segment import (
    DEFAULT_CLASSES,
    Detection,
    YoloPartDetector,
)


def box_vertices(
    cx: float, cy: float, cz: float, sx: float, sy: float, sz: float, n: int = 6
) -> np.ndarray:
    """以 (cx,cy,cz) 为中心、边长 (sx,sy,sz) 的盒顶点云（含面上采样点）。"""
    rng = np.random.default_rng(0)
    corners = np.array(
        [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]
    ) * [sx, sy, sz]
    faces = corners + np.array([cx, cy, cz])
    jitter = faces + rng.normal(0, 0.001, faces.shape)
    return np.vstack([jitter, faces])


@pytest.fixture()
def cabinet_parts() -> list[ArticulatedPart]:
    """柜体 + 两个抽屉 + 一个柜门（前面朝 +y）。"""
    base = ArticulatedPart(
        "cabinet_body", box_vertices(0, 0, 0.5, 0.8, 0.5, 1.0), kind="base"
    )
    drawer1 = ArticulatedPart(
        "drawer_top",
        box_vertices(0.0, 0.26, 0.78, 0.36, 0.06, 0.18),
        kind="drawer",
    )
    drawer2 = ArticulatedPart(
        "drawer_mid",
        box_vertices(0.0, 0.26, 0.56, 0.36, 0.06, 0.18),
        kind="drawer",
    )
    # 柜门：宽 0.30 高 0.40（高过宽），x ∈ [-0.15, 0.15]，贴在 +y 面外
    door = ArticulatedPart(
        "door_left", box_vertices(0.0, 0.27, 0.22, 0.30, 0.04, 0.40), kind="door"
    )
    return [base, drawer1, drawer2, door]


class TestEstimate:
    """几何关节估计（合成柜体：2 抽屉 + 1 柜门）。"""

    def test_base_is_largest(self, cabinet_parts):
        result = estimate_joints(cabinet_parts)
        assert result.base == "cabinet_body"

    def test_drawers_are_prismatic(self, cabinet_parts):
        result = estimate_joints(cabinet_parts)
        drawers = [j for j in result.joints if j.child.startswith("drawer")]
        assert len(drawers) == 2
        for j in drawers:
            assert j.joint_type is JointType.PRISMATIC
            # 轴向为外露面法向（+y 或 -y，取决于碰撞扫描方向选择）
            assert abs(abs(j.axis[1]) - 1.0) < 1e-3
            # 行程 = 深度(0.06) × 0.95（含顶点抖动容差）
            assert 0.02 <= j.upper <= 0.06 * 0.95 * 1.1
            assert j.lower == 0.0

    def test_door_is_revolute_with_vertical_axis(self, cabinet_parts):
        result = estimate_joints(cabinet_parts)
        door = next(j for j in result.joints if j.child == "door_left")
        assert door.joint_type is JointType.REVOLUTE
        assert abs(abs(door.axis[2]) - 1.0) < 1e-3  # 竖直铰链
        # 限位 = 碰撞扫描自由角：柜门应至少能开到 10°
        assert door.upper >= np.deg2rad(10)
        assert door.upper <= np.deg2rad(110) + 1e-9
        # 铰链在门的竖直棱上（x = ±0.15）
        assert abs(abs(door.origin[0]) - 0.15) < 0.02

    def test_no_review_for_clean_cabinet(self, cabinet_parts):
        result = estimate_joints(cabinet_parts)
        assert result.review == []
        assert result.acceptance_rate == 1.0

    def test_kind_guessing_without_hints(self, cabinet_parts):
        for p in cabinet_parts:
            p.kind = None
        result = estimate_joints(cabinet_parts)
        types = {j.child: j.joint_type for j in result.joints}
        assert types["drawer_top"] is JointType.PRISMATIC
        assert types["door_left"] is JointType.REVOLUTE

    def test_nested_part_goes_to_review(self):
        base = ArticulatedPart(
            "body", box_vertices(0, 0, 0.5, 0.8, 0.5, 1.0), kind="base"
        )
        # 完全嵌套在基座内部、与基座同心的"部件"——找不到朝外的面
        inner = ArticulatedPart(
            "weird", box_vertices(0, 0, 0.5, 0.2, 0.2, 0.2), kind="drawer"
        )
        result = estimate_joints([base, inner])
        assert result.joints == []
        assert len(result.review) == 1
        assert result.review[0].part == "weird"

    def test_low_confidence_goes_to_review(self, cabinet_parts):
        # 阈值提到 1.1（不可能达到）→ 全部进复核队列
        result = estimate_joints(cabinet_parts, confidence_threshold=1.1)
        assert len(result.joints) == 0
        assert len(result.review) == 3

    def test_requires_two_parts(self):
        with pytest.raises(ValueError):
            estimate_joints([ArticulatedPart("only", box_vertices(0, 0, 0, 1, 1, 1))])


class TestUrdfExport:
    """URDF 导出与结构校验。"""

    def test_export_and_validate(self, cabinet_parts):
        result = estimate_joints(cabinet_parts)
        urdf = result_to_urdf(result, cabinet_parts, object_name="cabinet")
        root = validate_urdf(urdf)  # 结构合法
        assert root.get("name") == "cabinet"
        links = {el.get("name") for el in root.findall("link")}
        assert links == {"cabinet_body", "drawer_top", "drawer_mid", "door_left"}
        joints = root.findall("joint")
        assert len(joints) == 3
        types = {j.get("name"): j.get("type") for j in joints}
        assert types["cabinet_body_to_drawer_top"] == "prismatic"
        assert types["cabinet_body_to_door_left"] == "revolute"
        # revolute 限位单位为弧度
        door = next(j for j in joints if j.get("name") == "cabinet_body_to_door_left")
        limit = door.find("limit")
        assert float(limit.get("upper")) <= np.deg2rad(110) + 1e-9

    def test_export_is_xml_parseable(self, cabinet_parts):
        result = estimate_joints(cabinet_parts)
        urdf = result_to_urdf(result, cabinet_parts)
        ET.fromstring(urdf)  # 不抛异常即可


class TestBatch:
    """批量标注落盘（results.json + review.jsonl）。"""

    def test_batch_annotate_writes_review_queue(self, cabinet_parts, tmp_path: Path):
        summary = batch_annotate(
            [("cabinet_a", cabinet_parts)], tmp_path, confidence_threshold=1.1
        )
        assert summary["assets"] == 1
        assert summary["joints_auto_accepted"] == 0
        assert summary["items_need_review"] == 3
        assert summary["auto_acceptance_rate"] == 0.0
        results = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
        assert "cabinet_a" in results
        review_lines = (
            (tmp_path / "review.jsonl").read_text(encoding="utf-8").strip().splitlines()
        )
        assert len(review_lines) == 3
        first = json.loads(review_lines[0])
        assert first["asset"] == "cabinet_a"
        assert "part" in first and "reason" in first

    def test_batch_with_verifier_moves_failures_to_review(
        self, cabinet_parts, tmp_path: Path
    ):
        """闭环验证接入：验证器判否的关节应降级进 review.jsonl，
        且 reason 带仿真读数；验证器异常不中断批次。
        """

        class FakeVerifier:
            def __init__(self, fail_names=(), raise_exc=False):
                self._fail = set(fail_names)
                self._raise = raise_exc

            def verify(self, result, parts, object_name="obj"):
                if self._raise:
                    raise RuntimeError("sim backend down")
                from plugins.predictors.articulation_gen.core.verify import (
                    JointCheck,
                    VerificationReport,
                )

                return VerificationReport(
                    checks=[
                        JointCheck(
                            joint=j.name,
                            child=j.child,
                            target=j.upper,
                            achieved=j.upper * 0.5 if j.name in self._fail else j.upper,
                            tracking_error=(
                                0.0 if j.name not in self._fail else j.upper * 0.5
                            ),
                            passed=j.name not in self._fail,
                            reason=(
                                "开合到位"
                                if j.name not in self._fail
                                else "关节未到达限位"
                            ),
                        )
                        for j in result.joints
                    ]
                )

        # 柜体 fixture 有 3 个关节；让 door 判否
        door_name = "cabinet_body_to_door_left"
        summary = batch_annotate(
            [("cab_a", cabinet_parts)],
            tmp_path,
            verifier=FakeVerifier(fail_names={door_name}),
        )
        assert summary["joints_auto_accepted"] == 2
        assert summary["items_need_review"] == 1
        assert summary["sim_rejected"] == 1
        review = json.loads(
            (tmp_path / "review.jsonl")
            .read_text(encoding="utf-8")
            .strip()
            .splitlines()[0]
        )
        assert review["asset"] == "cab_a"
        assert "未到达限位" in review["reason"]
        assert "实际到达" in review["reason"]

        # 验证器抛异常 → 全部进复核队列（reason 含异常信息），批次不中断，
        # 且不计入 sim_rejected（仿真未实际运行）
        summary = batch_annotate(
            [("cab_b", cabinet_parts)],
            tmp_path / "b",
            verifier=FakeVerifier(raise_exc=True),
        )
        assert summary["joints_auto_accepted"] == 0
        assert summary["items_need_review"] == 3
        assert summary["sim_rejected"] == 0
        b_review = json.loads(
            (tmp_path / "b" / "review.jsonl")
            .read_text(encoding="utf-8")
            .strip()
            .splitlines()[0]
        )
        assert "仿真验证异常" in b_review["reason"]


class TestLabeledMeshPath:
    """路径 1：命名节点 GLB → 部件 → 关节（trimesh 往返）。"""

    def test_glb_roundtrip(self, tmp_path: Path):
        trimesh = pytest.importorskip("trimesh")
        scene = trimesh.Scene()
        base = trimesh.creation.box(extents=(0.8, 0.5, 1.0))
        base.apply_translation((0, 0, 0.5))
        drawer = trimesh.creation.box(extents=(0.36, 0.4, 0.18))
        drawer.apply_translation((0, 0.1, 0.78))
        scene.add_geometry(base, geom_name="cabinet_body", node_name="cabinet_body")
        scene.add_geometry(drawer, geom_name="drawer_top", node_name="drawer_top")
        path = tmp_path / "cabinet.glb"
        scene.export(str(path))

        from plugins.predictors.articulation_gen import annotate_labeled_mesh

        result, parts = annotate_labeled_mesh(path)
        assert {p.name for p in parts} == {"cabinet_body", "drawer_top"}
        assert result.base == "cabinet_body"
        (drawer_joint,) = result.joints
        assert drawer_joint.joint_type is JointType.PRISMATIC
        # 真实深抽屉：行程 = 深度 0.4 × 0.95
        assert abs(drawer_joint.upper - 0.38) < 1e-6
        assert result.review == []


class TestSegmentGuards:
    """可选依赖 guard 与数据模型默认值。"""

    def test_detector_requires_ultralytics(self):
        pytest.importorskip("sys")
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            with pytest.raises(ImportError, match="ultralytics"):
                YoloPartDetector("nope.pt")

    def test_detection_dataclass_defaults(self):
        d = Detection(box=(0, 0, 10, 10), label="drawer", score=0.9)
        assert d.mask is None

    def test_default_classes_cover_joint_parts(self):
        assert "drawer" in DEFAULT_CLASSES and "door" in DEFAULT_CLASSES
