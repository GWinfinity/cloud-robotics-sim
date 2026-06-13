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

from typing import Any, List, Optional

# Optional dependencies with graceful fallback
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

# Genesis imports (optional - will fail gracefully if not installed)
try:
    import genesis as gs
    from genesis import utils as gu
    from genesis.utils import geom as gug

    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False
    gs = None
    gu = None
    gug = None


# =============================================================================
# Genesis Backend Compatibility (v0.x → v1.0 migration)
# =============================================================================
# genesis-world 1.0.0 changed: gs.backends.CUDA → gs._gs_backend.cuda
# This helper normalizes the API so code works across versions.

def get_genesis_backend(name: str = "cuda"):
    """Get a genesis backend by name, compatible with both old and new API.

    Args:
        name: Backend name ('cuda', 'cpu', 'GPU', 'CPU')

    Returns:
        The backend object, or None if genesis is not installed.
    """
    if not HAS_GENESIS:
        return None
    name_lower = name.lower()
    # Try new API first (genesis-world >= 1.0)
    if hasattr(gs, '_gs_backend'):
        backend = getattr(gs._gs_backend, name_lower, None)
        if backend is not None:
            return backend
    # Fall back to old API (genesis-world < 1.0)
    if hasattr(gs, 'backends'):
        backend = getattr(gs.backends, name, None) or getattr(gs.backends, name_lower, None)
        if backend is not None:
            return backend
    return None


def genesis_init(headless: bool = True, use_cuda: bool = True, **kwargs):
    """Initialize Genesis with version-compatible backend selection.

    Args:
        headless: Run without viewer.
        use_cuda: Use GPU if available, else CPU.
        **kwargs: Passed to gs.init().
    """
    if not HAS_GENESIS:
        raise RuntimeError("genesis-world is not installed")
    backend = get_genesis_backend("cuda" if use_cuda else "cpu")
    if backend is None:
        # Last resort: let genesis pick
        gs.init(**kwargs)
    else:
        gs.init(backend=backend, **kwargs)


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


def get_objs_by_names(objs: list, names: List[str]) -> list:
    """Get a list of objects given a list of names from a larger list of objects.

    The returned list is in the order of the names given.

    Args:
        objs: Objects to query. Expect these objects to have a get_name function.
        names: Names to query.

    Returns:
        List of matched objects in the order of names. None for no matches.
    """
    assert isinstance(objs, (list, tuple)), type(objs)
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
    matched_objects = [x for x in objs if type(x) == target_type]
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
    urdf_config = dict()

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
        urdf_config["link"] = dict()
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
    except Exception:
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
    except Exception:
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
    assert max_dof >= nq, (max_dof, nq)

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
    pairwise_contacts = {actor: [] for actor in actor1_list}
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
    entity_contacts = {actor: [] for actor in actors}
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

            return (
                abs(actual_pos - target_pos) > pos_diff_threshold
                and abs(actual_vel) < vel_threshold
            )
    except Exception:
        pass
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
            return torch.logical_and(
                torch.linalg.norm(lin_vel, axis=1) <= lin_thresh,
                torch.linalg.norm(ang_vel, axis=1) <= ang_thresh,
            )
        elif HAS_NUMPY:
            return (
                np.linalg.norm(lin_vel) <= lin_thresh
                and np.linalg.norm(ang_vel) <= ang_thresh
            )
        else:
            # Fallback without numpy
            lin_norm = sum(x * x for x in lin_vel) ** 0.5
            ang_norm = sum(x * x for x in ang_vel) ** 0.5
            return lin_norm <= lin_thresh and ang_norm <= ang_thresh
    except Exception:
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
        """Fallback implementation."""
        if HAS_TORCH:
            return torch.tensor([1.0, 0.0, 0.0, 0.0])
        return [1.0, 0.0, 0.0, 0.0]


try:
    from mani_skill.utils.structs.pose import Pose
except ImportError:

    class Pose:
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
                raw_pose = list(p) + list(q)
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
    # Compatibility
    "Pose",
    "matrix_to_quaternion",
    "GENESIS_RENDER_SYSTEM",
]
