# Cloud Robotics Sim - Utilities
#
# This module provides utility functions for Genesis-based simulations,
# adapted from ManiSkill's genesis_utils.py.

__version__ = "0.1.0"

# Import from genesis_compat (core utilities)
# Import from camera module
from .camera import (
    compute_fovy,
    create_viewer,
    genesis_pose_to_opencv_extrinsic,
    get_camera_rays,
    hex2rgba,
    look_at,
    rgba2hex,
    spherical_to_cartesian,
)
from .genesis_compat import (
    GENESIS_RENDER_SYSTEM,
    # Types
    Pose,
    apply_urdf_config,
    check_actor_static,
    # Utilities
    check_joint_stuck,
    # URDF config
    check_urdf_config,
    compute_total_impulse,
    # State extraction
    get_actor_state,
    get_articulation_padded_state,
    get_articulation_state,
    get_cpu_actor_contacts,
    get_cpu_actors_contacts,
    get_multiple_pairwise_contacts,
    # Object query
    get_obj_by_name,
    get_obj_by_type,
    get_objs_by_names,
    get_pairwise_contact_impulse,
    # Contact processing
    get_pairwise_contacts,
    is_state_dict_consistent,
    matrix_to_quaternion,
    parse_urdf_config,
)

# Import from rendering module
from .rendering import (
    ShaderConfig,
    configure_rendering,
    create_checkerboard_texture,
    load_texture,
    save_screenshot,
    set_articulation_render_material,
    set_entity_color,
    set_render_material,
    start_recording,
    stop_recording,
)

# Combined exports
__all__ = [
    # Genesis compatibility
    "get_obj_by_name",
    "get_objs_by_names",
    "get_obj_by_type",
    "check_urdf_config",
    "parse_urdf_config",
    "apply_urdf_config",
    "get_actor_state",
    "get_articulation_state",
    "get_articulation_padded_state",
    "get_pairwise_contacts",
    "get_multiple_pairwise_contacts",
    "compute_total_impulse",
    "get_pairwise_contact_impulse",
    "get_cpu_actor_contacts",
    "get_cpu_actors_contacts",
    "check_joint_stuck",
    "check_actor_static",
    "is_state_dict_consistent",
    "Pose",
    "matrix_to_quaternion",
    "GENESIS_RENDER_SYSTEM",
    # Camera
    "genesis_pose_to_opencv_extrinsic",
    "look_at",
    "spherical_to_cartesian",
    "compute_fovy",
    "get_camera_rays",
    "create_viewer",
    "hex2rgba",
    "rgba2hex",
    # Rendering
    "set_render_material",
    "set_articulation_render_material",
    "set_entity_color",
    "ShaderConfig",
    "configure_rendering",
    "load_texture",
    "create_checkerboard_texture",
    "save_screenshot",
    "start_recording",
    "stop_recording",
]


# Backward compatibility alias for ManiSkill users
def get_entity_by_name(objs, name, is_unique=True):
    """Alias for get_obj_by_name for ManiSkill compatibility."""
    return get_obj_by_name(objs, name, is_unique)


__all__.append("get_entity_by_name")
