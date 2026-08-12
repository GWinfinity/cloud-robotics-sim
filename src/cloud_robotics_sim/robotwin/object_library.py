"""RoboTwin-OD object library loader.

The RoboTwin object database (``assets/robotwin/objects/objects/``) stores
~129 object classes as ``NNN_name/`` directories. Each class contains one
collision mesh + one metadata JSON **per scale instance**::

    001_bottle/
        collision/base0.glb      # instance 0 collision mesh (coacd convex set)
        collision/base1.glb      # instance 1
        model_data0.json         # instance 0 metadata (scale/center/extents)
        model_data1.json
        visual/                  # optional visual meshes

This module enumerates classes, picks one instance per class (default: the
first), and spawns it into a Genesis scene as a free rigid body.

Nine classes (e.g. ``009_kettle``) use a different, SAPIEN PartNet-Mobility
layout — numbered sub-directories containing an articulated ``mobility.urdf``::

    009_kettle/
        102730/
            mobility.urdf
            model_data.json
        102738/

Both layouts are supported; :meth:`RoboTwinObjectLibrary.get_instance` returns
an :class:`ObjectInstance` whose ``kind`` field is ``"glb"`` or ``"urdf"``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cloud_robotics_sim.robotwin.assets import ensure_for_path

logger = logging.getLogger(__name__)

__all__ = ["ObjectInstance", "RoboTwinObjectLibrary", "normalize_scale"]


def normalize_scale(
    raw_height: float,
    scale: tuple[float, float, float],
    max_height: float = 0.35,
    target_height: float = 0.20,
) -> tuple[float, float, float]:
    """Shrink ``scale`` when the spawned object would be implausibly tall.

    Some RoboTwin classes ship no ``scale`` metadata (e.g. raw meshes ~2 m
    tall). If ``raw_height * scale`` exceeds ``max_height``, uniformly
    rescale so the object becomes ``target_height`` tall.

    ``raw_height`` is the mesh height in its own frame (glTF: Y extent).
    """
    height = raw_height * scale[1]
    if raw_height <= 0 or height <= max_height:
        return scale
    factor = target_height / raw_height
    return (scale[0] * factor, scale[1] * factor, scale[2] * factor)


_CLASS_DIR_RE = re.compile(r"^\d{3}_")


@dataclass
class ObjectInstance:
    """One instance of a RoboTwin object class."""

    class_name: str
    index: int
    kind: str  # "glb" (rigid collision mesh) or "urdf" (PartNet-Mobility)
    asset_path: Path  # baseN.glb or mobility.urdf
    scale: tuple[float, float, float]
    center: tuple[float, float, float]
    extents: tuple[float, float, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def scaled_extents(self) -> tuple[float, float, float]:
        """Axis-aligned bounding-box size after applying ``scale``."""
        return tuple(e * s for e, s in zip(self.extents, self.scale))  # type: ignore[return-value]


class RoboTwinObjectLibrary:
    """Enumerates RoboTwin object classes and resolves instance assets."""

    def __init__(self, objects_dir: str | Path) -> None:
        self.objects_dir = Path(objects_dir)
        if not self.objects_dir.is_dir() and ensure_for_path(
            "objects", self.objects_dir
        ):
            logger.info("RoboTwin objects auto-downloaded to %s", self.objects_dir)
        if not self.objects_dir.is_dir():
            raise FileNotFoundError(f"objects dir not found: {self.objects_dir}")

    def list_classes(self) -> list[str]:
        """Return sorted class directory names (``001_bottle``, ...)."""
        return sorted(
            d.name
            for d in self.objects_dir.iterdir()
            if d.is_dir() and _CLASS_DIR_RE.match(d.name)
        )

    def instance_indices(self, class_name: str) -> list[int]:
        """Available instance indices (``model_data<N>.json`` numbers)."""
        class_dir = self.objects_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"class dir not found: {class_dir}")
        indices = []
        for path in class_dir.glob("model_data*.json"):
            match = re.search(r"model_data(\d+)\.json$", path.name)
            if match:
                indices.append(int(match.group(1)))
        return sorted(indices)

    def class_kind(self, class_name: str) -> str:
        """``"glb"`` for mesh instances, ``"urdf"`` for PartNet-Mobility dirs."""
        if self.instance_indices(class_name):
            return "glb"
        class_dir = self.objects_dir / class_name
        for sub in sorted(class_dir.iterdir()):
            if sub.is_dir() and (sub / "mobility.urdf").is_file():
                return "urdf"
        raise FileNotFoundError(
            f"{class_name}: neither model_data*.json nor mobility.urdf found"
        )

    def instance_count(self, class_name: str) -> int:
        """Number of scale instances (``model_data*.json`` files) of a class."""
        return len(self.instance_indices(class_name))

    def get_instance(self, class_name: str, index: int | None = None) -> ObjectInstance:
        """Resolve one instance of ``class_name``.

        Args:
            class_name: Class directory name.
            index: Instance number (the ``N`` in ``baseN.glb`` /
                ``model_dataN.json``). ``None`` selects the smallest available
                index — note some classes start at 1 rather than 0.

        Raises:
            FileNotFoundError: class dir / metadata / collision mesh missing.
            IndexError: instance index out of range.
        """
        indices = self.instance_indices(class_name)
        if not indices:
            return self._get_urdf_instance(class_name, index)
        if index is None:
            index = indices[0]
        elif index not in indices:
            raise IndexError(
                f"{class_name}: instance {index} not available (have {indices})"
            )

        class_dir = self.objects_dir / class_name
        meta = json.loads(
            (class_dir / f"model_data{index}.json").read_text(encoding="utf-8")
        )
        glb_path = class_dir / "collision" / f"base{index}.glb"
        if not glb_path.is_file():
            raise FileNotFoundError(f"collision mesh not found: {glb_path}")

        return ObjectInstance(
            class_name=class_name,
            index=index,
            kind="glb",
            asset_path=glb_path,
            scale=tuple(float(x) for x in meta.get("scale", [1.0, 1.0, 1.0])),  # type: ignore[arg-type]
            center=tuple(float(x) for x in meta.get("center", [0.0, 0.0, 0.0])),  # type: ignore[arg-type]
            extents=tuple(float(x) for x in meta.get("extents", [0.0, 0.0, 0.0])),  # type: ignore[arg-type]
            metadata=meta,
        )

    def _get_urdf_instance(self, class_name: str, index: int | None) -> ObjectInstance:
        """Resolve a PartNet-Mobility (``mobility.urdf``) instance."""
        class_dir = self.objects_dir / class_name
        subs = sorted(
            s
            for s in class_dir.iterdir()
            if s.is_dir() and (s / "mobility.urdf").is_file()
        )
        if not subs:
            raise FileNotFoundError(
                f"{class_name}: no collision/baseN.glb and no mobility.urdf"
            )
        sub_idx = 0 if index is None else index
        if not 0 <= sub_idx < len(subs):
            raise IndexError(
                f"{class_name}: urdf instance {sub_idx} out of range ({len(subs)})"
            )
        sub = subs[sub_idx]
        meta_path = sub / "model_data.json"
        meta: dict[str, Any] = (
            json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.is_file()
            else {}
        )
        scale_raw = meta.get("scale", [1.0])
        if isinstance(scale_raw, (int, float)):
            scale_raw = [float(scale_raw)] * 3
        if len(scale_raw) == 1:
            scale_raw = list(scale_raw) * 3
        return ObjectInstance(
            class_name=class_name,
            index=int(sub.name) if sub.name.isdigit() else sub_idx,
            kind="urdf",
            asset_path=sub / "mobility.urdf",
            scale=tuple(float(x) for x in scale_raw[:3]),  # type: ignore[arg-type]
            center=(0.0, 0.0, 0.0),
            extents=(0.0, 0.0, 0.0),
            metadata=meta,
        )

    def spawn_in_scene(
        self,
        scene: Any,
        class_name: str,
        pos: tuple[float, float, float],
        index: int | None = None,
        quat: tuple[float, float, float, float] | None = None,
        friction: float = 0.8,
    ) -> Any:
        """Spawn the instance into a Genesis scene as a free rigid body.

        Args:
            scene: A ``genesis.Scene`` (entities are added before ``build``).
            class_name: Class directory name.
            pos: Spawn position (place slightly above the support surface and
                let the object settle under gravity).
            index: Instance number within the class (``None`` = first available).
            quat: Optional spawn quaternion ``(w, x, y, z)``.
            friction: Contact friction coefficient.

        Returns:
            The created Genesis entity.
        """
        import genesis as gs  # local import: keeps module importable without GPU

        inst = self.get_instance(class_name, index)
        if inst.kind == "urdf":
            urdf_scale = float(inst.scale[0])
            bbox_path = inst.asset_path.parent / "bounding_box.json"
            if bbox_path.is_file():
                try:
                    bb = json.loads(bbox_path.read_text(encoding="utf-8"))
                    dims = [
                        float(hi) - float(lo) for lo, hi in zip(bb["min"], bb["max"])
                    ]
                    urdf_scale = normalize_scale(
                        max(dims), (urdf_scale, urdf_scale, urdf_scale)
                    )[0]
                except Exception:  # noqa: BLE001 - keep original scale
                    pass
            entity = scene.add_entity(
                gs.morphs.URDF(
                    file=str(inst.asset_path),
                    pos=pos,
                    scale=urdf_scale,
                    fixed=False,
                ),
                surface=gs.surfaces.Default(roughness=0.6),
            )
            return entity
        scale = inst.scale
        if not inst.metadata.get("keep_scale"):
            try:
                import trimesh

                # conservative proxy: shrink if ANY dimension is implausible
                raw_height = float(max(trimesh.load(str(inst.asset_path)).extents))
                scale = normalize_scale(raw_height, inst.scale)
            except Exception:  # noqa: BLE001 - keep original scale on any failure
                pass
        kwargs: dict[str, Any] = {
            "file": str(inst.asset_path),
            "pos": pos,
            "scale": scale,
            "fixed": False,
            "convexify": True,
        }
        if quat is not None:
            kwargs["quat"] = quat
        entity = scene.add_entity(
            gs.morphs.Mesh(**kwargs),
            surface=gs.surfaces.Default(roughness=0.6),
        )
        return entity
