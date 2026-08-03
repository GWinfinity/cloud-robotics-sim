# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Genesis Compatibility Utilities

This module provides compatibility utilities for Genesis-based simulations,
adapted from ManiSkill's genesis_utils.py to work with genesis-cloud-sim's
architecture.

Provides:
- Object querying utilities (get_obj_by_name, etc.)
- URDF configuration helpers
- State extraction functions
- Contact processing utilities

References:
    - Original: ManiSkill-main/mani_skill/utils/genesis_utils.py
    - Genesis: https://github.com/Genesis-Embodied-AI/Genesis
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Optional dependencies with graceful fallback
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]

# Genesis imports (optional - will fail gracefully if not installed)
try:
    import genesis as gs

    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False
    gs = None


# =============================================================================
# Genesis Backend Compatibility (genesis-world >= 1.0)
# =============================================================================
# genesis-world 1.0+ exposes backends as gs.cpu / gs.cuda / gs.gpu.
# Older internal names (gs._gs_backend) are avoided when public names exist.


def get_genesis_backend(name: str = "cuda") -> Any:
    """Get a genesis backend by name.

    Args:
        name: Backend name ('cuda', 'cpu', 'gpu'). 'musa' is not a native
            Genesis backend and will return None; use ``device.py`` for MUSA
            torch-device selection.

    Returns:
        The backend object, or None if genesis is not installed or the name
        is not supported.
    """
    if not HAS_GENESIS or gs is None:
        return None
    name_lower = name.lower()
    # MUSA is not a Genesis backend (Genesis uses Taichi/CUDA).
    if name_lower == "musa":
        return None
    # Prefer public top-level backend names (genesis-world >= 1.0).
    backend = getattr(gs, name_lower, None)
    if backend is not None:
        return backend
    # Fallback to the internal enum namespace if the public name is missing.
    if hasattr(gs, "_gs_backend"):
        return getattr(gs._gs_backend, name_lower, None)
    return None


def genesis_init(
    headless: bool = True,
    use_cuda: bool = True,
    device: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Initialize Genesis with version-compatible backend selection.

    Args:
        headless: Run without viewer (kept for API compatibility; Genesis
            itself does not accept this parameter).
        use_cuda: Use GPU if available, else CPU. If CUDA is requested but
            unavailable, automatically falls back to CPU.
        device: Optional device preference (``cpu``/``cuda``/``musa``). When
            ``musa`` is requested, Genesis itself still runs on the CPU backend
            because Genesis does not yet support MUSA; PyTorch tensors can be
            placed on MUSA via ``device.set_default_device``.
        **kwargs: Passed to gs.init().
    """
    del headless  # Not used; gs.init does not accept this parameter.
    if not HAS_GENESIS or gs is None:
        raise RuntimeError("genesis-world is not installed")

    if getattr(gs, "_initialized", False):
        logger.debug("Genesis already initialized, skipping")
        return

    # Compatibility workaround: some genesis-world Linux wheels import a stale
    # or incomplete genesis module object into genesis.utils.misc, so the
    # get_device() helper cannot access gs.gpu/gs.cuda/etc. Point it at the
    # canonical module from sys.modules instead and make sure the canonical
    # module exposes the backend attributes that get_device() and gs.init()
    # compare against.
    import sys

    def _inject_backend_names(_mod: Any) -> None:
        """Expose backend enum members as module attributes and in gs.init globals."""
        _enum = getattr(_mod, "_gs_backend", None)
        if _enum is None:
            # Some genesis-world installs do not re-export the backend enum on
            # the top-level module; import it directly from genesis.constants.
            try:
                from genesis.constants import backend as _imported_enum

                _enum = _imported_enum
            except Exception:
                return
        _members = getattr(_enum, "__members__", {})
        if not _members:
            return
        for _name, _member in _members.items():
            if not hasattr(_mod, _name):
                try:
                    setattr(_mod, _name, _member)
                except Exception:
                    pass
        _init_fn = getattr(_mod, "init", None)
        if callable(_init_fn):
            _globals = getattr(_init_fn, "__globals__", None)
            if not _globals:
                return
            for _name, _member in _members.items():
                _globals[_name] = _member
            # gs.init() references backend names via ``gs.cpu`` etc., where the
            # ``gs`` global may point at a stale genesis module object (e.g.
            # left over from a failed first import) that never received the
            # backend attributes. Inject the names there as well.
            _stale = _globals.get("gs")
            if _stale is not None and _stale is not _mod:
                for _name, _member in _members.items():
                    try:
                        setattr(_stale, _name, _member)
                    except Exception:
                        pass

    is_real_genesis = gs is sys.modules.get("genesis")
    if is_real_genesis:
        try:
            canonical_gs = sys.modules["genesis"]
            _inject_backend_names(canonical_gs)
            try:
                import genesis.utils.misc as _misc

                if _misc.gs is not canonical_gs:
                    logger.debug(
                        "Aligning genesis.utils.misc.gs with canonical genesis module"
                    )
                    _misc.gs = canonical_gs
                _inject_backend_names(_misc.gs)
            except Exception:
                pass
        except Exception:
            pass
    else:
        # If our local gs reference differs from sys.modules (e.g. after a
        # reload or wrapper), still try to make it backend-aware.
        _inject_backend_names(gs)

    # In GitHub Actions / CI runners there is no GPU, so force the CPU backend
    # explicitly as requested (gs.init(backend=gs.cpu)).
    if os.environ.get("CI", "").lower() == "true":
        logger.debug("CI environment detected; forcing Genesis CPU backend")
        use_cuda = False

    # Resolve device preference for Genesis backend selection. MUSA is not a
    # native Genesis backend, so we fall back to CPU for the physics engine.
    if device is not None:
        from .device import get_device

        resolved = get_device(device)
        if resolved == "musa":
            logger.debug(
                "MUSA requested for Genesis; physics engine will use CPU backend"
            )
            use_cuda = False
        else:
            use_cuda = resolved == "cuda"

    backend = get_genesis_backend("cuda" if use_cuda else "cpu")
    if backend is None:
        gs.init(**kwargs)
        return

    try:
        gs.init(backend=backend, **kwargs)
    except AttributeError as exc:
        # Some genesis-world installs crash inside gs.init() because the ``gs``
        # global it references lacks the backend attributes (cpu/gpu/...).
        # Re-inject the names and retry with auto-detection.
        if any(
            name in str(exc).lower()
            for name in ("cpu", "gpu", "cuda", "metal", "amdgpu")
        ):
            logger.debug(
                "gs.init backend attribute missing; re-injecting and auto-detecting"
            )
            _inject_backend_names(sys.modules.get("genesis") or gs)
            gs.init(**kwargs)
            return
        raise
    except gs.GenesisException:
        if use_cuda:
            logger.debug("CUDA backend failed, falling back to CPU")
            backend = get_genesis_backend("cpu")
            if backend is not None:
                gs.init(backend=backend, **kwargs)
                return
        raise


def ensure_genesis_initialized(**kwargs: Any) -> None:
    """Initialize Genesis if not already initialized (idempotent)."""
    if not HAS_GENESIS or gs is None:
        raise RuntimeError("genesis-world is not installed")
    try:
        genesis_init(**kwargs)
    except gs.GenesisException:
        logger.debug("Genesis already initialized")


def get_genesis_lights() -> Any:
    """Return the Genesis lights module if available.

    genesis-world 1.2+ does not expose ``gs.lights``; lighting is configured
    through ``VisOptions`` or the default scene lighting. This helper lets
    callers gracefully handle either case.

    Returns:
        The ``gs.lights`` module, or None if it is not available.
    """
    if not HAS_GENESIS or gs is None:
        return None
    return getattr(gs, "lights", None)


def is_genesis_scene(scene: Any) -> bool:
    """Return True if *scene* is a native Genesis gs.Scene instance."""
    if not HAS_GENESIS or gs is None:
        return False
    try:
        return isinstance(scene, gs.Scene)
    except Exception:
        return False


# =============================================================================
# Type Aliases
# =============================================================================

ArrayLike = Any  # Union[np.ndarray, torch.Tensor] when available


# =============================================================================
# Object Query Utilities
# =============================================================================


def get_obj_by_name(objs: list, name: str, is_unique: bool = True) -> Any:
    """Get an object given the name.

    Args:
        objs: Objects to query. Expect these objects to have a get_name function.
              These may be genesis.Entity, genesis.ArticulationLink etc.
        name: Name for query.
        is_unique: Whether the name should be unique. Defaults to True.

    Raises:
        RuntimeError: The name is not unique when @is_unique is True.

    Returns:
        The matched object or list of objects. None if no matches.
    """
    matched_objects = [
        x for x in objs if hasattr(x, "get_name") and x.get_name() == name
    ]
    if len(matched_objects) > 1:
        if not is_unique:
            return matched_objects
        else:
            raise RuntimeError(f"Multiple objects with the same name {name}.")
    elif len(matched_objects) == 1:
        return matched_objects[0]
    else:
        return None


def get_objs_by_names(objs: list, names: list[str]) -> list:
    """Get a list of objects given a list of names from a larger list of objects.

    The returned list is in the order of the names given.

    Args:
        objs: Objects to query. Expect these objects to have a get_name function.
        names: Names to query.

    Returns:
        List of matched objects in the order of names. None for no matches.
    """
    if not isinstance(objs, (list, tuple)):
        raise TypeError(f"Expected list or tuple, got {type(objs).__name__}")
    ret = [None for _ in names]

    for obj in objs:
        if hasattr(obj, "get_name"):
            name = obj.get_name()
            if name in names:
                ret[names.index(name)] = obj
    return ret


def get_obj_by_type(objs: list, target_type: type, is_unique: bool = True) -> Any:
    """Get an object by its type.

    Args:
        objs: List of objects to search.
        target_type: The type to match.
        is_unique: Whether to expect a unique match.

    Returns:
        The matched object(s) or None.
    """
    matched_objects = [x for x in objs if isinstance(x, target_type)]
    if len(matched_objects) > 1:
        if not is_unique:
            return matched_objects
        else:
            raise RuntimeError(f"Multiple objects with the same type {target_type}.")
    elif len(matched_objects) == 1:
        return matched_objects[0]
    else:
        return None


# =============================================================================
# URDF Configuration
# =============================================================================


def check_urdf_config(urdf_config: dict):
    """Check whether the urdf config is valid for Genesis.

    Args:
        urdf_config: Dict passed to Genesis URDF loader.

    Raises:
        KeyError: If invalid keys are present.
    """
    allowed_keys = ["material", "density", "link"]
    for k in urdf_config.keys():
        if k not in allowed_keys:
            raise KeyError(
                f"Not allowed key ({k}) for Genesis URDF loader. Allowed keys are {allowed_keys}"
            )

    allowed_link_keys = ["material", "density", "patch_radius", "min_patch_radius"]
    for k, v in urdf_config.get("link", {}).items():
        for kk in v.keys():
            if kk not in allowed_link_keys:
                raise KeyError(
                    f"Not allowed key ({kk}) for Genesis URDF loader. Allowed keys are {allowed_link_keys}"
                )


def parse_urdf_config(config_dict: dict) -> dict:
    """Parse config from dict for Genesis URDF loader.

    Args:
        config_dict: A dict containing link physical properties.

    Returns:
        URDF config passed to Genesis URDF loader.
    """
    urdf_config = {}

    # Create the global physical material for all links
    if "material" in config_dict:
        if gs is not None:
            urdf_config["material"] = gs.materials.Rigid(**config_dict["material"])
        else:
            urdf_config["material"] = config_dict["material"]

    # Create link-specific physical materials
    materials = {}
    if "_materials" in config_dict:
        for k, v in config_dict["_materials"].items():
            if gs is not None:
                materials[k] = gs.materials.Rigid(**v)
            else:
                materials[k] = v

    # Specify properties for links
    if "link" in config_dict:
        urdf_config["link"] = {}
        for k, link_config in config_dict["link"].items():
            urdf_config["link"][k] = link_config.copy()
            # substitute with actual material
            if "material" in link_config:
                urdf_config["link"][k]["material"] = materials[link_config["material"]]
    return urdf_config


def apply_urdf_config(loader, urdf_config: dict):
    """Apply URDF config to a Genesis URDF loader.

    Args:
        loader: Genesis URDF loader.
        urdf_config: Dictionary containing URDF configuration.
    """
    if "link" in urdf_config:
        for name, link_config in urdf_config["link"].items():
            if "material" in link_config:
                mat = link_config["material"]
                if hasattr(loader, "set_link_material"):
                    loader.set_link_material(
                        name,
                        getattr(mat, "static_friction", 0.5),
                        getattr(mat, "dynamic_friction", 0.5),
                        getattr(mat, "restitution", 0.0),
                    )
            if "patch_radius" in link_config:
                if hasattr(loader, "set_link_patch_radius"):
                    loader.set_link_patch_radius(name, link_config["patch_radius"])
            if "min_patch_radius" in link_config:
                if hasattr(loader, "set_link_min_patch_radius"):
                    loader.set_link_min_patch_radius(
                        name, link_config["min_patch_radius"]
                    )
            if "density" in link_config:
                if hasattr(loader, "set_link_density"):
                    loader.set_link_density(name, link_config["density"])

    if "material" in urdf_config:
        mat = urdf_config["material"]
        if hasattr(loader, "set_material"):
            loader.set_material(
                getattr(mat, "static_friction", 0.5),
                getattr(mat, "dynamic_friction", 0.5),
                getattr(mat, "restitution", 0.0),
            )
    if "patch_radius" in urdf_config:
        if hasattr(loader, "set_patch_radius"):
            loader.set_patch_radius(urdf_config["patch_radius"])
    if "min_patch_radius" in urdf_config:
        if hasattr(loader, "set_min_patch_radius"):
            loader.set_min_patch_radius(urdf_config["min_patch_radius"])
    if "density" in urdf_config:
        if hasattr(loader, "set_density"):
            loader.set_density(urdf_config["density"])


# =============================================================================
# State Extraction
# =============================================================================


def get_actor_state(actor) -> Optional[ArrayLike]:
    """Get the state of a Genesis actor.

    Args:
        actor: Genesis actor object.

    Returns:
        Actor state including pose, velocity, and angular velocity.
        Shape: (13,) [pos(3), quat(4), vel(3), ang_vel(3)]
    """
    if not HAS_NUMPY:
        return None

    try:
        pose = actor.get_pose()

        # Get position and quaternion from pose
        pos = np.array(pose.p, dtype=np.float32)
        quat = np.array(pose.q, dtype=np.float32)

        # Get velocity and angular velocity
        if hasattr(actor, "get_linear_velocity") and hasattr(
            actor, "get_angular_velocity"
        ):
            vel = actor.get_linear_velocity()
            ang_vel = actor.get_angular_velocity()
        else:
            vel = np.zeros(3, dtype=np.float32)
            ang_vel = np.zeros(3, dtype=np.float32)

        return np.hstack([pos, quat, vel, ang_vel])
    except Exception as e:
        logger.debug("Failed to get actor state: %s", e)
        return None


def get_articulation_state(articulation) -> Optional[ArrayLike]:
    """Get the state of a Genesis articulation.

    Args:
        articulation: Genesis articulation object.

    Returns:
        Articulation state including root pose, velocity, joint positions, and joint velocities.
        Shape: (13 + 2*n_dof,)
    """
    if not HAS_NUMPY:
        return None

    try:
        links = articulation.get_links()
        if not links:
            return None

        root_link = links[0]
        pose = root_link.get_pose()

        # Get root position and quaternion
        pos = np.array(pose.p, dtype=np.float32)
        quat = np.array(pose.q, dtype=np.float32)

        # Get root velocity and angular velocity
        vel = root_link.get_linear_velocity()
        ang_vel = root_link.get_angular_velocity()

        # Get joint positions and velocities
        qpos = articulation.get_qpos()
        qvel = articulation.get_qvel()

        return np.hstack([pos, quat, vel, ang_vel, qpos, qvel])
    except Exception as e:
        logger.debug("Failed to get articulation state: %s", e)
        return None


def get_articulation_padded_state(articulation, max_dof: int) -> Optional[ArrayLike]:
    """Get the padded state of a Genesis articulation.

    Args:
        articulation: Genesis articulation object.
        max_dof: Maximum degrees of freedom to pad to.

    Returns:
        Padded articulation state. Shape: (13 + 2*max_dof,)
    """
    if not HAS_NUMPY:
        return None

    state = get_articulation_state(articulation)
    if state is None:
        return None

    # Split into root (13) + joint states
    root_state = state[:13]
    joint_states = state[13:]

    nq = len(joint_states) // 2  # qpos and qvel
    if max_dof < nq:
        raise ValueError(f"max_dof ({max_dof}) must be >= actual DOFs ({nq})")

    padded_state = np.zeros(13 + 2 * max_dof, dtype=np.float32)
    padded_state[:13] = root_state
    padded_state[13 : 13 + nq] = joint_states[:nq]
    padded_state[13 + max_dof : 13 + max_dof + nq] = joint_states[nq:]
    return padded_state


# =============================================================================
# Contact Processing
# =============================================================================


def get_pairwise_contacts(contacts: list, actor0, actor1) -> list:
    """Get pairwise contacts between two actors.

    Args:
        contacts: List of contact objects.
        actor0: First actor.
        actor1: Second actor.

    Returns:
        List of tuples containing (contact, is_first) where is_first indicates
        if actor0 is the first body in the contact.
    """
    pairwise_contacts = []
    for contact in contacts:
        # Check if contact is between actor0 and actor1
        if hasattr(contact, "bodies") and len(contact.bodies) >= 2:
            if (
                contact.bodies[0].entity == actor0
                and contact.bodies[1].entity == actor1
            ):
                pairwise_contacts.append((contact, True))
            elif (
                contact.bodies[0].entity == actor1
                and contact.bodies[1].entity == actor0
            ):
                pairwise_contacts.append((contact, False))
    return pairwise_contacts


def get_multiple_pairwise_contacts(contacts: list, actor0, actor1_list: list) -> dict:
    """Get pairwise contacts between one actor and multiple actors.

    Args:
        contacts: List of contact objects.
        actor0: First actor.
        actor1_list: List of other actors.

    Returns:
        Dictionary mapping each actor to a list of contacts.
    """
    pairwise_contacts: dict[Any, list] = {actor: [] for actor in actor1_list}
    for contact in contacts:
        if hasattr(contact, "bodies") and len(contact.bodies) >= 2:
            if (
                contact.bodies[0].entity == actor0
                and contact.bodies[1].entity in actor1_list
            ):
                pairwise_contacts[contact.bodies[1].entity].append((contact, True))
            elif (
                contact.bodies[0].entity in actor1_list
                and contact.bodies[1].entity == actor0
            ):
                pairwise_contacts[contact.bodies[0].entity].append((contact, False))
    return pairwise_contacts


def compute_total_impulse(contact_infos: list) -> ArrayLike:
    """Compute total impulse from contacts.

    Args:
        contact_infos: List of tuples containing (contact, is_first).

    Returns:
        Total impulse as a numpy array.
    """
    if not HAS_NUMPY:
        return None

    total_impulse = np.zeros(3)
    for contact, is_first in contact_infos:
        if hasattr(contact, "points"):
            contact_impulse = np.sum(
                [point.impulse for point in contact.points], axis=0
            )
            # Impulse is applied on the first component
            total_impulse += contact_impulse * (1 if is_first else -1)
    return total_impulse


def get_pairwise_contact_impulse(contacts: list, actor0, actor1) -> ArrayLike:
    """Get pairwise contact impulse between two actors.

    Args:
        contacts: List of contact objects.
        actor0: First actor.
        actor1: Second actor.

    Returns:
        Contact impulse as a numpy array.
    """
    pairwise_contacts = get_pairwise_contacts(contacts, actor0, actor1)
    return compute_total_impulse(pairwise_contacts)


def get_cpu_actor_contacts(contacts: list, actor) -> list:
    """Get all contacts for a specific actor.

    Args:
        contacts: List of contact objects.
        actor: The actor to get contacts for.

    Returns:
        List of tuples containing (contact, is_first).
    """
    entity_contacts = []
    for contact in contacts:
        if hasattr(contact, "bodies") and len(contact.bodies) >= 2:
            if contact.bodies[0].entity == actor:
                entity_contacts.append((contact, True))
            elif contact.bodies[1].entity == actor:
                entity_contacts.append((contact, False))
    return entity_contacts


def get_cpu_actors_contacts(contacts: list, actors: list) -> dict:
    """Get contacts for multiple actors.

    Args:
        contacts: List of contact objects.
        actors: List of actors.

    Returns:
        Dictionary mapping each actor to a list of contacts.
    """
    entity_contacts: dict[Any, list] = {actor: [] for actor in actors}
    for contact in contacts:
        if hasattr(contact, "bodies") and len(contact.bodies) >= 2:
            if contact.bodies[0].entity in actors:
                entity_contacts[contact.bodies[0].entity].append((contact, True))
            elif contact.bodies[1].entity in actors:
                entity_contacts[contact.bodies[1].entity].append((contact, False))
    return entity_contacts


# =============================================================================
# Joint and Actor Utilities
# =============================================================================


def check_joint_stuck(
    articulation,
    active_joint_idx: int,
    pos_diff_threshold: float = 1e-3,
    vel_threshold: float = 1e-4,
) -> bool:
    """Check if a joint is stuck.

    Args:
        articulation: Articulation object.
        active_joint_idx: Index of the active joint.
        pos_diff_threshold: Position difference threshold.
        vel_threshold: Velocity threshold.

    Returns:
        True if the joint is stuck, False otherwise.
    """
    try:
        if (
            hasattr(articulation, "get_qpos")
            and hasattr(articulation, "get_drive_target")
            and hasattr(articulation, "get_qvel")
        ):
            actual_pos = articulation.get_qpos()[active_joint_idx]
            target_pos = articulation.get_drive_target()[active_joint_idx]
            actual_vel = articulation.get_qvel()[active_joint_idx]

            return bool(
                abs(actual_pos - target_pos) > pos_diff_threshold
                and abs(actual_vel) < vel_threshold
            )
    except Exception as e:
        logger.debug("Failed to check joint stuck: %s", e)
    return False


def check_actor_static(
    actor, lin_thresh: float = 1e-3, ang_thresh: float = 1e-2
) -> bool:
    """Check if an actor is static.

    Args:
        actor: Actor object.
        lin_thresh: Linear velocity threshold.
        ang_thresh: Angular velocity threshold.

    Returns:
        True if the actor is static, False otherwise.
    """
    try:
        if hasattr(actor, "linear_velocity") and hasattr(actor, "angular_velocity"):
            lin_vel = actor.linear_velocity
            ang_vel = actor.angular_velocity
        elif hasattr(actor, "get_linear_velocity") and hasattr(
            actor, "get_angular_velocity"
        ):
            lin_vel = actor.get_linear_velocity()
            ang_vel = actor.get_angular_velocity()
        else:
            return True

        if HAS_TORCH and isinstance(lin_vel, torch.Tensor):
            result = torch.logical_and(
                torch.linalg.norm(lin_vel, axis=1) <= lin_thresh,
                torch.linalg.norm(ang_vel, axis=1) <= ang_thresh,
            )
            return bool(result.all())
        elif HAS_NUMPY:
            return bool(
                np.linalg.norm(lin_vel) <= lin_thresh
                and np.linalg.norm(ang_vel) <= ang_thresh
            )
        else:
            # Fallback without numpy
            lin_norm = sum(x * x for x in lin_vel) ** 0.5
            ang_norm = sum(x * x for x in ang_vel) ** 0.5
            return bool(lin_norm <= lin_thresh and ang_norm <= ang_thresh)
    except Exception as e:
        logger.debug("Failed to check actor static: %s", e)
        return True


def is_state_dict_consistent(state_dict: dict) -> bool:
    """Check if state dictionary has consistent batch dimensions.

    Args:
        state_dict: Dictionary generated via env.get_state_dict().

    Returns:
        True if all actors/articulations have the same batch dimension.
    """
    batch_size = None
    for name in ["actors", "articulations"]:
        if name in state_dict:
            for k, v in state_dict[name].items():
                if hasattr(v, "shape"):
                    if batch_size is None:
                        batch_size = v.shape[0]
                    else:
                        if v.shape[0] != batch_size:
                            return False
    return True


# =============================================================================
# Legacy / Placeholder Exports
# =============================================================================

# These are placeholders for compatibility but will be properly implemented
# in rendering.py and camera.py
PhysxMaterial = Any
Entity = Any
Actor = Any
PhysxContact = Any
PhysxArticulation = Any
render = Any
Viewer = Any
CameraConfig = Any
GENESIS_RENDER_SYSTEM = "1.0"


# Import rotation conversion utilities (if available)
try:
    from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion
except ImportError:

    def matrix_to_quaternion(matrix):
        """Fallback: convert rotation matrix to quaternion (w, x, y, z).

        Uses Shepperd's method for numerical stability. Accepts a 3x3 or
        batched (N, 3, 3) rotation matrix as a numpy array or torch tensor.
        """
        if HAS_TORCH and isinstance(matrix, torch.Tensor):
            m = matrix.reshape(-1, 3, 3).float()
            trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
            quat = torch.zeros(m.shape[0], 4, device=m.device, dtype=m.dtype)

            # Case 1: trace > 0
            s = torch.sqrt(trace + 1.0) * 2  # 4w
            w = 0.25 * s
            x = (m[:, 2, 1] - m[:, 1, 2]) / s
            y = (m[:, 0, 2] - m[:, 2, 0]) / s
            z = (m[:, 1, 0] - m[:, 0, 1]) / s
            mask = trace > 0
            quat[mask] = torch.stack([w, x, y, z], dim=-1)[mask]

            # Case 2-4: largest diagonal element
            for i, (j, k) in enumerate([(0, (1, 2)), (1, (0, 2)), (2, (0, 1))]):
                j1, j2 = k
                cond = (
                    (~mask) & (m[:, i, i] > m[:, j1, j1]) & (m[:, i, i] > m[:, j2, j2])
                )
                if cond.any():
                    s2 = (
                        torch.sqrt(
                            1.0 + m[cond, i, i] - m[cond, j1, j1] - m[cond, j2, j2]
                        )
                        * 2
                    )
                    q = torch.zeros(int(cond.sum()), 4, device=m.device, dtype=m.dtype)
                    q[:, i + 1] = 0.25 * s2
                    q[:, 0] = (m[cond, j2, j1] - m[cond, j1, j2]) / s2
                    q[:, j1 + 1] = (m[cond, j1, i] + m[cond, i, j1]) / s2
                    q[:, j2 + 1] = (m[cond, j2, i] + m[cond, i, j2]) / s2
                    quat[cond] = q

            # Normalize
            quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            return quat.squeeze(0) if matrix.ndim == 2 else quat

        # Numpy fallback
        m_np = np.asarray(matrix, dtype=np.float64).reshape(-1, 3, 3)
        quat_np = np.zeros((m_np.shape[0], 4), dtype=np.float64)
        trace_np = m_np[:, 0, 0] + m_np[:, 1, 1] + m_np[:, 2, 2]

        pos = trace_np > 0
        if pos.any():
            s_np = np.sqrt(trace_np[pos] + 1.0) * 2
            quat_np[pos, 0] = 0.25 * s_np
            quat_np[pos, 1] = (m_np[pos, 2, 1] - m_np[pos, 1, 2]) / s_np
            quat_np[pos, 2] = (m_np[pos, 0, 2] - m_np[pos, 2, 0]) / s_np
            quat_np[pos, 3] = (m_np[pos, 1, 0] - m_np[pos, 0, 1]) / s_np

        for i, (j1, j2) in enumerate([(1, 2), (0, 2), (0, 1)]):
            neg = (
                (~pos)
                & (m_np[:, i, i] > m_np[:, j1, j1])
                & (m_np[:, i, i] > m_np[:, j2, j2])
            )
            if neg.any():
                s2_np = (
                    np.sqrt(
                        1.0 + m_np[neg, i, i] - m_np[neg, j1, j1] - m_np[neg, j2, j2]
                    )
                    * 2
                )
                quat_np[neg, 0] = (m_np[neg, j2, j1] - m_np[neg, j1, j2]) / s2_np
                quat_np[neg, i + 1] = 0.25 * s2_np
                quat_np[neg, j1 + 1] = (m_np[neg, j1, i] + m_np[neg, i, j1]) / s2_np
                quat_np[neg, j2 + 1] = (m_np[neg, j2, i] + m_np[neg, i, j2]) / s2_np

        norms = np.linalg.norm(quat_np, axis=-1, keepdims=True)
        quat_np = quat_np / np.maximum(norms, 1e-8)
        result = quat_np[0] if matrix.ndim == 2 else quat_np
        return result.tolist() if not HAS_NUMPY else result


try:
    from mani_skill.utils.structs.pose import Pose
except ImportError:

    class Pose:  # type: ignore[no-redef]
        """Placeholder Pose class."""

        def __init__(self, raw_pose):
            self.raw_pose = raw_pose

        @classmethod
        def create_from_pq(cls, p=None, q=None, device=None):
            if p is None:
                if HAS_TORCH:
                    p = torch.zeros((1, 3), device=device)
                else:
                    p = [[0.0, 0.0, 0.0]]
            if q is None:
                if HAS_TORCH:
                    q = torch.zeros((1, 4), device=device)
                    q[:, 0] = 1
                else:
                    q = [[1.0, 0.0, 0.0, 0.0]]
            if not isinstance(p, (torch.Tensor if HAS_TORCH else object)):
                if HAS_TORCH:
                    p = torch.tensor(p, dtype=torch.float32, device=device)
            if not isinstance(q, (torch.Tensor if HAS_TORCH else object)):
                if HAS_TORCH:
                    q = torch.tensor(q, dtype=torch.float32, device=device)
            if HAS_TORCH:
                if p.ndim == 1:
                    p = p.unsqueeze(0)
                if q.ndim == 1:
                    q = q.unsqueeze(0)
                raw_pose = torch.cat([p, q], dim=-1)
            else:
                raw_pose = list(p) + list(q)  # type: ignore[assignment]
            return cls(raw_pose)


__all__ = [
    # Object query
    "get_obj_by_name",
    "get_objs_by_names",
    "get_obj_by_type",
    # URDF config
    "check_urdf_config",
    "parse_urdf_config",
    "apply_urdf_config",
    # State extraction
    "get_actor_state",
    "get_articulation_state",
    "get_articulation_padded_state",
    # Contact processing
    "get_pairwise_contacts",
    "get_multiple_pairwise_contacts",
    "compute_total_impulse",
    "get_pairwise_contact_impulse",
    "get_cpu_actor_contacts",
    "get_cpu_actors_contacts",
    # Utilities
    "check_joint_stuck",
    "check_actor_static",
    "is_state_dict_consistent",
    "is_genesis_scene",
    # Compatibility
    "Pose",
    "matrix_to_quaternion",
    "GENESIS_RENDER_SYSTEM",
]
