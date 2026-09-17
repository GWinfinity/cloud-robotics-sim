"""ReplicaCAD scene builder for Genesis.

Ports the semantics of ManiSkill's ``ReplicaCADSceneBuilder``
(``mani_skill/utils/scene_builder/replicacad/scene_builder.py``) to Genesis so
the ReplicaCAD scene dataset mirrored on ModelScope
(``jessy888/ManiSkill_replica_cad_dataset``) can be used with the
``ReplicaCAD_SceneManipulation-v1``-style scenes from XLerobot's workflow.

Dataset layout (extracted under the dataset root)::

    replica_cad_dataset/
        configs/scenes/<name>.scene_instance.json
        configs/objects/<name>.object_config.json
        configs/stages/<name>.stage_config.json
        stages/<name>.glb
        objects/<name>.glb, objects/convex/<name>_cv_decomp.glb
        urdf/<name>/<name>.urdf

Conventions (matching ManiSkill's port of the habitat dataset):

- All ReplicaCAD assets are authored Y-up; every instance pose is left-multiplied
  by a +90-degree rotation about X to obtain Genesis's Z-up world frame. Dataset
  quaternions are (x, y, z, w); Genesis expects (w, x, y, z).
- DYNAMIC objects are simulated as rigid bodies; STATIC objects are fixed.
  For DYNAMIC objects the pre-decomposed convex collision GLB is used as the
  entity mesh (Genesis's own convex decomposition is then trivial). The rendered
  surface is the convex decomposition, i.e. a slightly faceted version of the
  original object -- a deliberate trade-off to avoid expensive per-object
  convex decomposition at scene build time.
- Articulated furniture is loaded from ``urdf/<name>/<name>.urdf``; door
  templates are skipped by default, matching ManiSkill.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

# NOTE: replicacad_assets is imported lazily inside the methods that use it:
# the datasets subpackage uses absolute imports and is only importable under
# the top-level ``genesis_maniskill`` name (see the plugin examples).

logger = logging.getLogger(__name__)

# RotX(+90 deg) as a habitat-style (x, y, z, w) quaternion, used to convert
# the Y-up ReplicaCAD/habitat frame into Genesis's Z-up world frame (identical
# to ManiSkill's SAPIEN port).
_SIN45 = math.sin(math.radians(45.0))
_FRAME_QUAT_XYZW = np.array([_SIN45, 0.0, 0.0, _SIN45])


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without Genesis)
# ---------------------------------------------------------------------------


def habitat_pose_to_genesis(
    translation: Any,
    rotation_xyzw: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a habitat pose into Genesis ``(pos, quat_wxyz)``.

    Args:
        translation: (3,) habitat-frame position (meters).
        rotation_xyzw: (4,) habitat quaternion in x, y, z, w order.

    Returns:
        ``(pos, quat)`` with ``pos`` a (3,) float array in the Z-up world
        frame and ``quat`` a (4,) float array in Genesis's w, x, y, z order.
    """
    t = np.asarray(translation, dtype=np.float64)
    q = np.asarray(rotation_xyzw, dtype=np.float64)
    if t.shape != (3,):
        raise ValueError(f"expected (3,) translation, got {t.shape}")
    if q.shape != (4,):
        raise ValueError(f"expected (4,) rotation, got {q.shape}")
    q = q / np.linalg.norm(q)

    frame = _FRAME_QUAT_XYZW
    # Rotate the translation by the frame rotation.
    pos = _rotate_vec_by_quat_xyzw(frame, t)
    # Left-multiply orientations by the frame rotation (Hamilton product).
    rot = _quat_mul_xyzw(frame, q)
    quat_wxyz = np.array([rot[3], rot[0], rot[1], rot[2]])
    return pos, quat_wxyz


def _rotate_vec_by_quat_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by quaternion ``q`` (x, y, z, w) via q*v*q^-1."""
    x, y, z, w = q
    u = np.array([x, y, z])
    uv = np.cross(u, v)
    uuv = np.cross(u, uv)
    return v + 2.0 * (w * uv + uuv)


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two (x, y, z, w) quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def load_scene_config(dataset_path: str | Path, scene_name: str) -> dict:
    """Load ``<scene_name>.scene_instance.json`` from the dataset."""
    path = (
        Path(dataset_path) / "configs" / "scenes" / f"{scene_name}.scene_instance.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"ReplicaCAD scene config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_json_config(config_path: str | Path) -> dict:
    """Load a ReplicaCAD ``*.object_config.json`` / ``*.stage_config.json``."""
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def resolve_object_meshes(
    dataset_path: str | Path,
    template_name: str,
) -> dict:
    """Resolve an object template into mesh files and physical properties.

    Args:
        dataset_path: Extracted ``replica_cad_dataset`` directory.
        template_name: Habitat object template, e.g. ``objects/frl_apartment_basket``.

    Returns:
        Dict with keys ``visual`` (Path), ``body`` (Path, the mesh used for
        the physics body), ``mass`` (float or None) and ``is_dynamic``
        suitability is left to the caller.
    """
    name = Path(template_name).name
    config_path = (
        Path(dataset_path) / "configs" / "objects" / f"{name}.object_config.json"
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"ReplicaCAD object config not found: {config_path}")
    config = load_json_config(config_path)

    def _resolve(rel: str) -> Path:
        # Asset paths are relative to configs/objects/.
        return (config_path.parent / rel).resolve()

    visual = _resolve(config["render_asset"])
    body = visual
    collision_rel = config.get("collision_asset")
    if collision_rel:
        # Pre-decomposed convex collision mesh: reuse it as the body mesh so
        # Genesis's convexification is trivial (see module docstring).
        candidate = _resolve(collision_rel)
        if candidate.is_file():
            body = candidate
        else:
            logger.debug(
                "collision asset missing, falling back to visual: %s", candidate
            )
    mass = config.get("mass")
    return {
        "visual": visual,
        "body": body,
        "mass": float(mass) if mass is not None else None,
        "use_bounding_box_for_collision": bool(
            config.get("use_bounding_box_for_collision", False)
        ),
    }


def resolve_stage_mesh(dataset_path: str | Path, template_name: str) -> Path:
    """Resolve a stage template (e.g. ``stages/frl_apartment_stage``) to its GLB."""
    name = Path(template_name).name
    config_path = (
        Path(dataset_path) / "configs" / "stages" / f"{name}.stage_config.json"
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"ReplicaCAD stage config not found: {config_path}")
    config = load_json_config(config_path)
    return (config_path.parent / config["render_asset"]).resolve()


def resolve_articulated_urdf(dataset_path: str | Path, template_name: str) -> Path:
    """Resolve an articulated-object template to ``urdf/<name>/<name>.urdf``."""
    name = Path(template_name).name
    path = Path(dataset_path) / "urdf" / name / f"{name}.urdf"
    if not path.is_file():
        raise FileNotFoundError(f"ReplicaCAD articulated URDF not found: {path}")
    return path


# ---------------------------------------------------------------------------
# Genesis scene builder
# ---------------------------------------------------------------------------


class ReplicaCADSceneBuilder:
    """Build a ReplicaCAD scene (from the ModelScope dataset) in Genesis.

    Args:
        scene: A ``gs.Scene`` instance (entities are added before ``build()``).
        scene_name: Scene configuration name, e.g. ``"apt_0"`` or
            ``"v3_sc1_staging_00"``. See :func:`list_available_scenes`.
        dataset_root: Dataset root containing ``replica_cad_dataset/``
            (default: :func:`~genesis_maniskill.datasets.replicacad_assets.default_dataset_root`).
        num_envs: Number of parallel environments (currently only 1).
        config: Optional dict with keys:

            - ``include_doors`` (bool, default False): also spawn ``door*``
              articulated-object templates (skipped by default, matching
              ManiSkill).
            - ``add_lights`` (bool, default True): best-effort addition of the
              default ReplicaCAD point lights (requires a renderer that
              supports ``scene.add_light``, e.g. the batch renderer;
              silently skipped otherwise). Ambient lighting is controlled by
              the caller through ``gs.options.VisOptions(ambient_light=...)``.

    Call :meth:`finalize` after ``scene.build()`` to apply object masses.
    """

    def __init__(
        self,
        scene: Any,
        scene_name: str = "apt_0",
        dataset_root: str | Path | None = None,
        num_envs: int = 1,
        config: dict | None = None,
    ):
        self.scene = scene
        self.scene_name = scene_name
        self.dataset_root = Path(dataset_root) if dataset_root is not None else None
        self.num_envs = num_envs
        self.config = config or {}

        self.scene_objects: dict[str, Any] = {}
        self.movable_objects: dict[str, Any] = {}
        self.articulations: dict[str, Any] = {}
        self.background: Any = None
        self._default_object_poses: list[tuple[Any, tuple]] = []
        self._movable_defaults: list[tuple[Any, tuple]] = []
        self._pending_masses: list[tuple[Any, float]] = []

    # -- introspection ------------------------------------------------------

    @staticmethod
    def list_available_scenes(root: str | Path | None = None) -> list[str]:
        """Return the sorted ReplicaCAD scene configuration names."""
        from ..datasets.replicacad_assets import list_scenes

        return list_scenes(root=root)

    # -- building -----------------------------------------------------------

    def build(self) -> None:
        """Add all ReplicaCAD entities to the Genesis scene.

        Must be called before ``scene.build()``; call :meth:`finalize` after.
        """
        import genesis as gs

        if self.num_envs != 1:
            raise NotImplementedError(
                "ReplicaCADSceneBuilder currently supports num_envs=1 only"
            )
        dset = self._dataset_path()
        scene_config = load_scene_config(dset, self.scene_name)
        include_doors = bool(self.config.get("include_doors", False))

        # -- stage (static apartment background) -----------------------------
        stage_template = scene_config["stage_instance"]["template_name"]
        stage_mesh = resolve_stage_mesh(dset, stage_template)
        stage_pos, stage_quat = habitat_pose_to_genesis(
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
        )
        self.background = self.scene.add_entity(
            gs.morphs.Mesh(
                file=str(stage_mesh),
                pos=tuple(stage_pos),
                quat=tuple(stage_quat),
                fixed=True,
                convexify=False,
            )
        )

        # -- rigid objects ----------------------------------------------------
        for obj_num, obj_meta in enumerate(scene_config.get("object_instances", [])):
            actor_name = f"{Path(obj_meta['template_name']).name}-{obj_num}"
            pos, quat = habitat_pose_to_genesis(
                obj_meta["translation"], obj_meta["rotation"]
            )
            meshes = resolve_object_meshes(dset, obj_meta["template_name"])
            common = {
                "file": str(meshes["body"]),
                "pos": tuple(pos),
                "quat": tuple(quat),
            }
            if obj_meta.get("motion_type", "DYNAMIC") == "DYNAMIC":
                entity = self.scene.add_entity(
                    gs.morphs.Mesh(fixed=False, convexify=True, **common)
                )
                self.movable_objects[actor_name] = entity
                self._default_object_poses.append((entity, (tuple(pos), tuple(quat))))
                self._movable_defaults.append((entity, (tuple(pos), tuple(quat))))
                if meshes["mass"] is not None:
                    self._pending_masses.append((entity, meshes["mass"]))
            else:
                entity = self.scene.add_entity(
                    gs.morphs.Mesh(fixed=True, convexify=False, **common)
                )
            self.scene_objects[actor_name] = entity

        # -- articulated furniture --------------------------------------------
        articulation_counts: dict[str, int] = {}
        for art_meta in scene_config.get("articulated_object_instances", []):
            template_name = Path(art_meta["template_name"]).name
            if not include_doors and "door" in template_name.lower():
                continue
            urdf_path = resolve_articulated_urdf(dset, template_name)
            pos, quat = habitat_pose_to_genesis(
                art_meta["translation"], art_meta["rotation"]
            )
            scale = float(art_meta.get("uniform_scale", 1.0))
            entity = self.scene.add_entity(
                gs.morphs.URDF(
                    file=str(urdf_path),
                    pos=tuple(pos),
                    quat=tuple(quat),
                    scale=scale,
                    fixed=bool(art_meta.get("fixed_base", True)),
                )
            )
            index = articulation_counts.get(template_name, 0)
            articulation_counts[template_name] = index + 1
            name = f"{template_name}-{index}"
            self.articulations[name] = entity
            self.scene_objects[name] = entity
            self._default_object_poses.append((entity, (tuple(pos), tuple(quat))))

        logger.info(
            "ReplicaCAD scene %s: %d objects (%d movable), %d articulations",
            self.scene_name,
            len(self.scene_objects),
            len(self.movable_objects),
            len(self.articulations),
        )

        if self.config.get("add_lights", True):
            self._add_default_lighting()

    def finalize(self) -> None:
        """Apply deferred per-object masses. Call after ``scene.build()``."""
        for entity, mass in self._pending_masses:
            entity.set_mass(mass)

    def _add_default_lighting(self) -> None:
        """Best-effort ReplicaCAD interior lights (ManiSkill's positions).

        genesis-world 1.4 only supports ``scene.add_light`` on the batch
        renderer; on other renderers this is a no-op and lighting falls back
        to ``VisOptions.ambient_light`` (set by the environment).
        """
        color = (1.0, 0.8, 0.5)
        intensity = 2.0  # np.array([1.0, 0.8, 0.5]) * 2
        # fmt: off
        for pos in [
            (-1.1,  2.775, 2.3),  # entrance
            (-0.5, -1.44,  2.3),  # dining area
            ( 2.4, -1.6,   2.3),  # dining back
            ( 2.5, -6.1,   2.3),  # living room
            ( 3.14, 3.24,  3.0),  # stair
        ]:
            # fmt: on
            try:
                self.scene.add_light(
                    pos=pos,
                    dir=(0.0, 0.0, -1.0),
                    color=color,
                    intensity=intensity,
                    cutoff=45.0,
                )
            except Exception as exc:  # noqa: BLE001 - renderer without add_light
                logger.debug("add_light unsupported, skipping: %s", exc)
                break

    # -- reset / randomization ----------------------------------------------

    def randomize_object_placement(self, noise: float = 0.01) -> None:
        """Jitter movable object positions in the horizontal plane.

        Mirrors ``KitchenSceneBuilder.randomize_object_placement``; called
        after ``scene.reset()`` so entities are in their default poses.
        """
        movable = dict.fromkeys(id(e) for e, _ in self._movable_defaults)
        for entity, (pos, _quat) in self._default_object_poses:
            if id(entity) in movable:
                new_pos = (
                    pos[0] + float(np.random.uniform(-noise, noise)),
                    pos[1] + float(np.random.uniform(-noise, noise)),
                    pos[2],
                )
                entity.set_pos(new_pos)

    # -- internals ------------------------------------------------------------

    def _dataset_path(self) -> Path:
        from ..datasets.replicacad_assets import default_dataset_root, ensure_dataset

        root = self.dataset_root
        if root is None:
            root = default_dataset_root()
        return ensure_dataset(root=root)


__all__ = [
    "ReplicaCADSceneBuilder",
    "habitat_pose_to_genesis",
    "load_json_config",
    "load_scene_config",
    "resolve_articulated_urdf",
    "resolve_object_meshes",
    "resolve_stage_mesh",
]
