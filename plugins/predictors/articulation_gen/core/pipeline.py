"""标注流水线：把"原始输入"（mesh / 图像）变成 ArticulationResult。

三条输入路径（按需选用）：

1. ``annotate_labeled_mesh(path)`` — GLB/OBJ/USD 带命名节点的 mesh，
   节点名即部件名（asset_gen 产物、人工 rig 的模型走这条，完全离线）；
2. ``annotate_from_images(mesh, images, cameras, detector, refiner)`` —
   YOLO 检测 + SAM mask，把 2D mask 反投影回 mesh 顶点得到部件分割，
   再走同一几何估计器（需要本地推理依赖与微调权重）；
3. ``annotate_parts(parts)`` — 已有部件顶点（如 PartNet-Mobility 标签）。

``batch_annotate`` 把一批资产跑完并落盘 ``results.json`` + 复核队列
``review.jsonl``——这是把"全自动接受率"与"总吞吐"解耦的关键：
低置信度的不静默放过，而是进队列人工确认。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np
import trimesh

from .estimate import estimate_joints
from .joint_spec import ArticulatedPart, ArticulationResult
from .segment import Detection, SamMaskRefiner, YoloPartDetector

#: 节点名 → 语义 kind 的启发式映射（子串匹配，小写）
_KIND_HINTS = {
    "drawer": "drawer",
    "door": "door",
    "lid": "lid",
    "knob": "knob",
    "handle": "handle",
    "base": "base",
    "cabinet": "base",
    "body": "base",
    "frame": "base",
}


class _Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]: ...


class _Refiner(Protocol):
    def refine(
        self, image: np.ndarray, detections: list[Detection]
    ) -> list[Detection]: ...


@dataclass
class Camera:
    """针孔相机（世界 → 像素）。"""

    K: np.ndarray  # (3, 3) 内参
    pose: np.ndarray  # (4, 4) 世界 → 相机

    def project(self, points_world: np.ndarray) -> np.ndarray:
        """(N, 3) 世界点 → (N, 2) 像素坐标（z<=0 的点返回 NaN）。"""
        pts = np.asarray(points_world, dtype=np.float64)
        cam = (self.pose @ np.column_stack([pts, np.ones(len(pts))]).T).T[:, :3]
        uvw = (self.K @ cam.T).T
        with np.errstate(invalid="ignore", divide="ignore"):
            uv = uvw[:, :2] / uvw[:, 2:3]
        uv[cam[:, 2] <= 0] = np.nan
        return uv


# ---------------------------------------------------------------------------
# 路径 1：命名节点 mesh
# ---------------------------------------------------------------------------


def parts_from_trimesh_scene(path: str | Path) -> list[ArticulatedPart]:
    """加载 GLB/OBJ/USD，每个命名几何体成为一个部件。

    节点名按 :data:`_KIND_HINTS` 映射语义 kind（如 ``drawer_front`` →
    drawer）；无匹配的 kind 为 None（由几何猜测）。
    """
    scene = trimesh.load(path, force="scene")
    if not isinstance(scene, trimesh.Scene):
        scene = trimesh.Scene(scene)
    parts: list[ArticulatedPart] = []
    for node_name in scene.graph.nodes_geometry:
        geometry_name = scene.graph[node_name][1]
        geom = scene.geometry[geometry_name]
        verts = np.asarray(geom.vertices, dtype=np.float64)
        if len(verts) < 4:
            continue
        # 应用节点自身的世界变换
        matrix = scene.graph.get(node_name)[0]
        verts = trimesh.transformations.transform_points(verts, matrix)
        lowered = geometry_name.lower()
        kind = next((v for k, v in _KIND_HINTS.items() if k in lowered), None)
        parts.append(ArticulatedPart(name=geometry_name, vertices=verts, kind=kind))
    if not parts:
        raise ValueError(f"no geometry nodes found in {path}")
    return parts


def annotate_labeled_mesh(
    path: str | Path, *, object_name: str | None = None, **estimate_kw: Any
) -> tuple[ArticulationResult, list[ArticulatedPart]]:
    """路径 1 入口：命名节点 mesh → (结果, 部件)。"""
    parts = parts_from_trimesh_scene(path)
    return annotate_parts(parts, **estimate_kw), parts


# ---------------------------------------------------------------------------
# 路径 2：图像 + mesh（YOLO + SAM 反投影）
# ---------------------------------------------------------------------------


def lift_masks_to_labels(
    mesh: trimesh.Trimesh,
    images: list[np.ndarray],
    cameras: list[Camera],
    detections_per_view: list[list[Detection]],
) -> np.ndarray:
    """把多视角 2D mask 投票成顶点标签 (-1 = 未标注)。

    对每个视角，把 mesh 顶点投影到图像；落在某 mask 内的顶点获得该
    mask 类别。多视角冲突时采用"被投票次数最多"的类别。
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    votes: list[dict[str, int]] = [dict() for _ in range(len(verts))]
    for image, cam, dets in zip(images, cameras, detections_per_view):
        h, w = image.shape[:2]
        uv = cam.project(verts)
        for det in dets:
            if det.mask is None:
                continue
            inside = (
                (uv[:, 0] >= 0)
                & (uv[:, 0] < w)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < h)
                & det.mask[uv[:, 1].astype(int) % h, uv[:, 0].astype(int) % w]
            )
            for idx in np.nonzero(inside)[0]:
                votes[idx][det.label] = votes[idx].get(det.label, 0) + 1
    labels = np.full(len(verts), -1, dtype=int)
    class_ids = sorted({lbl for v in votes for lbl in v})
    class_to_id = {c: i for i, c in enumerate(class_ids)}
    for i, v in enumerate(votes):
        if v:
            labels[i] = class_to_id[max(v, key=v.get)]
    return labels


def split_mesh_by_labels(
    mesh: trimesh.Trimesh, labels: np.ndarray, class_names: list[str]
) -> list[ArticulatedPart]:
    """按顶点标签切分 mesh（面内全部顶点同标签才归属该部件）。"""
    faces = np.asarray(mesh.faces)
    face_labels = np.full(len(faces), -1, dtype=int)
    for i, f in enumerate(faces):
        fl = labels[f]
        if (fl >= 0).all() and (fl == fl[0]).all():
            face_labels[i] = fl[0]
    parts = []
    for cid, cname in enumerate(class_names):
        if cname in ("handle", "knob"):
            continue  # 把手类小件不参与关节估计
        face_idx = np.nonzero(face_labels == cid)[0]
        if len(face_idx) < 2:
            continue
        sub = mesh.submesh([face_idx], append=True)
        verts = np.asarray(sub.vertices, dtype=np.float64)
        if len(verts) >= 4:
            parts.append(
                ArticulatedPart(name=f"part_{cname}", vertices=verts, kind=cname)
            )
    if not parts:
        raise ValueError("no labeled parts survived mesh splitting")
    return parts


def annotate_from_images(
    mesh: trimesh.Trimesh,
    images: list[np.ndarray],
    cameras: list[Camera],
    detector: _Detector,
    refiner: _Refiner | None = None,
    **estimate_kw: Any,
) -> tuple[ArticulationResult, list[ArticulatedPart]]:
    """路径 2 入口：图像 + mesh → (结果, 部件)。

    Args:
        mesh: 与图像对应的完整 mesh（世界坐标）。
        images: 多视角 (H, W, 3) uint8 图。
        cameras: 与 images 对齐的相机参数。
        detector: YOLO 检测器（如 :class:`YoloPartDetector`）。
        refiner: 可选 SAM 精修器；None 时以检测框为 mask。
        **estimate_kw: 透传给 :func:`estimate_joints`（如 confidence_threshold）。
    """
    detections_per_view: list[list[Detection]] = []
    for image in images:
        dets = detector.detect(image)
        if refiner is not None:
            dets = refiner.refine(image, dets)
        else:
            for d in dets:  # 退化：以框为 mask
                x1, y1, x2, y2 = (int(round(v)) for v in d.box)
                m = np.zeros(image.shape[:2], dtype=bool)
                m[max(y1, 0) : y2, max(x1, 0) : x2] = True
                d.mask = m
        detections_per_view.append(dets)

    all_labels = sorted({d.label for dets in detections_per_view for d in dets})
    labels = lift_masks_to_labels(mesh, images, cameras, detections_per_view)
    parts = split_mesh_by_labels(mesh, labels, all_labels)
    return annotate_parts(parts, **estimate_kw), parts


# ---------------------------------------------------------------------------
# 路径 3 / 公共入口 / 批量
# ---------------------------------------------------------------------------


def annotate_parts(
    parts: list[ArticulatedPart], **estimate_kw: Any
) -> ArticulationResult:
    """公共入口：已分割部件 → 关节估计。"""
    return estimate_joints(parts, **estimate_kw)


def batch_annotate(
    items: Iterable[tuple[str, list[ArticulatedPart]]],
    out_dir: str | Path,
    verifier: Any | None = None,
    **estimate_kw: Any,
) -> dict[str, Any]:
    """批量标注并落盘。

    Args:
        items: ``(资产名, 部件列表)`` 迭代器。
        out_dir: 输出目录，写 ``results.json`` 与复核队列 ``review.jsonl``。
        verifier: 可选闭环验证器（如 ``GenesisVerifier``）。提供时，每个
            关节先经仿真开合验证，未通过的从 joints 降级到复核队列
            （reason 含仿真读数）。验证器抛异常时不中断批次，该资产的
            全部关节进复核队列。
        **estimate_kw: 透传给 :func:`estimate_joints`（如 confidence_threshold）。

    Returns:
        汇总 dict，含全自动接受率（自动化接受 / 需要人工复核）。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    n_auto = n_review = n_sim_failed = 0
    with (out / "review.jsonl").open("w", encoding="utf-8") as review_f:
        for name, parts in items:
            result = annotate_parts(parts, **estimate_kw)
            if verifier is not None and result.joints:
                result, n_failed = _apply_verification(result, parts, verifier, name)
                n_sim_failed += n_failed
            results[name] = result.to_dict()
            for item in result.review:
                review_f.write(
                    json.dumps({"asset": name, **item.to_dict()}, ensure_ascii=False)
                    + "\n"
                )
            n_auto += len(result.joints)
            n_review += len(result.review)
    (out / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = n_auto + n_review
    return {
        "assets": len(results),
        "joints_auto_accepted": n_auto,
        "items_need_review": n_review,
        "sim_rejected": n_sim_failed,
        "auto_acceptance_rate": n_auto / total if total else 1.0,
        "results": str(out / "results.json"),
        "review_queue": str(out / "review.jsonl"),
    }


def _apply_verification(
    result: ArticulationResult,
    parts: list[ArticulatedPart],
    verifier: Any,
    name: str,
) -> tuple[ArticulationResult, int]:
    """仿真验证：未通过的关节降级到复核队列。

    Returns:
        (新结果, 被仿真判否的关节数)。不改动入参。
    """
    from .joint_spec import ReviewItem

    try:
        report = verifier.verify(result, parts, object_name=name)
    except Exception as exc:  # noqa: BLE001 - 验证器故障不中断批次
        # 仿真未实际运行：全部关节进复核队列，但不计入 sim_rejected
        failed = {j.name: f"仿真验证异常: {exc}" for j in result.joints}
        achieved: dict[str, float] = {}
        sim_rejected = 0
    else:
        failed = {c.joint: c.reason for c in report.failed}
        achieved = {c.joint: c.achieved for c in report.failed}
        sim_rejected = len(failed)
    if not failed:
        return result, 0
    kept = []
    review = list(result.review)
    for joint in result.joints:
        if joint.name in failed:
            reason = failed[joint.name]
            if joint.name in achieved:
                reason += (
                    f"（实际到达 {achieved[joint.name]:.4f} / 目标 {joint.upper:.4f}）"
                )
            review.append(
                ReviewItem(part=joint.child, reason=reason, confidence=joint.confidence)
            )
        else:
            kept.append(joint)
    return (
        ArticulationResult(base=result.base, joints=kept, review=review),
        sim_rejected,
    )


__all__ = [
    "Camera",
    "annotate_from_images",
    "annotate_labeled_mesh",
    "annotate_parts",
    "batch_annotate",
    "lift_masks_to_labels",
    "parts_from_trimesh_scene",
    "split_mesh_by_labels",
    "SamMaskRefiner",
    "YoloPartDetector",
]
