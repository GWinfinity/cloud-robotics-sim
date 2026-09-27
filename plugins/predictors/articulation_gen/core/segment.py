"""2D 部件分割适配层：YOLO 检测 + SAM mask 精修。

设计原则（与项目其它模块一致）：

- **可选依赖全部 guard**：``ultralytics``（YOLO 家族任意 checkpoint，
  包括微调后的 YOLO26 检测/分割模型）和 ``sam2``（或 ultralytics 自带
  FastSAM）任一缺失时抛带安装提示的 ImportError，核心几何估计不受影响。
- **检测器与分割器解耦**：检测器给出带语义的框（drawer/door/handle），
  分割器把框精修成 mask；分割器缺失时可退化为"以框为 mask"（置信度
  惩罚由调用方决定）。
- 不在本模块内下载权重：模型路径由调用方提供（微调产物或官方 checkpoint）。

典型用法::

    detector = YoloPartDetector("runs/detect/train/weights/best.pt")
    refiner = SamMaskRefiner()          # 需要 sam2 或 ultralytics FastSAM
    dets = detector.detect(image)
    masks = refiner.refine(image, dets)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Detection:
    """单张图上的一个 2D 检测结果。"""

    box: tuple[float, float, float, float]  # xyxy
    label: str
    score: float
    mask: np.ndarray | None = field(
        default=None, repr=False
    )  # (H, W) bool，refine 后填充


#: 关节标注关心的默认类别（训练 YOLO 时建议覆盖）
DEFAULT_CLASSES = ("drawer", "door", "handle", "knob")


def _missing(package: str, extra: str) -> ImportError:
    return ImportError(
        f"articulation_gen 的可选依赖 {package!r} 未安装。"
        f"请运行: uv sync --extra dev && uv pip install {extra}"
    )


class YoloPartDetector:
    """YOLO 家族检测器适配（ultralytics）。

    接受任意 ultralytics 支持的 checkpoint：官方 YOLO26/25/v11 检测权重，
    或在你自己标注的家具部件数据集上微调的权重（推荐——公开预训练权重
    不覆盖 drawer/door/handle 类别）。

    Args:
        model_path: 本地权重路径。
        conf: 置信度阈值。
        classes: 保留的类别名（其余过滤）。
    """

    def __init__(
        self,
        model_path: str,
        conf: float = 0.5,
        classes: tuple[str, ...] = DEFAULT_CLASSES,
    ) -> None:
        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - 依赖缺失路径
            raise _missing("ultralytics", "ultralytics") from exc
        self._model = YOLO(model_path)
        self._conf = conf
        self._classes = set(classes)

    def detect(self, image: np.ndarray) -> list[Detection]:
        """对单张 (H, W, 3) uint8 图像检测，返回过滤后的 Detection 列表。"""
        results: Any = self._model.predict(image, conf=self._conf, verbose=False)
        dets: list[Detection] = []
        for r in results:
            names: dict[int, str] = r.names
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls = names[int(b.cls)]
                if cls not in self._classes:
                    continue
                dets.append(
                    Detection(
                        box=tuple(float(v) for v in b.xyxy[0].tolist()),
                        label=cls,
                        score=float(b.conf),
                    )
                )
        return dets


class SamMaskRefiner:
    """SAM mask 精修（sam2 优先，退化 ultralytics FastSAM）。

    Args:
        checkpoint: sam2 权重路径；None 时尝试 FastSAM（ultralytics 自动管理）。
        device: 推理设备。
    """

    def __init__(
        self, checkpoint: str | None = None, device: str | None = None
    ) -> None:
        self._backend = self._load_sam2(checkpoint, device) or self._load_fastsam(
            device
        )
        if self._backend is None:
            raise _missing("sam2 / ultralytics[fastsam]", "sam2")  # pragma: no cover

    @staticmethod
    def _load_sam2(checkpoint: str | None, device: str | None) -> Any | None:
        try:
            from sam2.sam2_image_predictor import (
                SAM2ImagePredictor,  # type: ignore[import-not-found]
            )
        except ImportError:
            return None
        predictor = SAM2ImagePredictor.from_pretrained(
            checkpoint or "facebook/sam2.1-hiera-large"
        )
        if device:
            predictor.to(device)
        return ("sam2", predictor)

    @staticmethod
    def _load_fastsam(device: str | None) -> Any | None:
        try:
            from ultralytics import FastSAM  # type: ignore[import-not-found]
        except ImportError:
            return None
        model = FastSAM("FastSAM-s.pt")
        if device:
            model.to(device)
        return ("fastsam", model)

    def refine(self, image: np.ndarray, detections: list[Detection]) -> list[Detection]:
        """对每个 Detection 填充 mask（失败则保留 None）。返回同一列表。"""
        if self._backend is None:  # pragma: no cover
            return detections
        kind, model = self._backend
        if kind == "sam2":
            model.set_image(image)
            if not detections:
                return detections
            boxes = np.array([d.box for d in detections], dtype=np.float32)
            masks, _, _ = model.predict(box=boxes, multimask_output=False)
            for det, m in zip(detections, masks):
                det.mask = np.asarray(m, dtype=bool).squeeze()
            return detections
        # FastSAM：逐框推理
        for det in detections:
            x1, y1, x2, y2 = det.box
            results = model(image, bboxes=[x1, y1, x2, y2], verbose=False)
            if results and results[0].masks is not None:
                det.mask = results[0].masks.data[0].cpu().numpy().astype(bool)
        return detections


__all__ = ["DEFAULT_CLASSES", "Detection", "SamMaskRefiner", "YoloPartDetector"]
