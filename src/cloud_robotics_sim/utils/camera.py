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

"""Camera utilities for Genesis simulations.

This module provides camera-related utilities adapted from ManiSkill's
genesis_utils.py to work with genesis-cloud-sim's architecture.

Includes:
- Pose conversion utilities
- Camera look-at calculations
- Color conversion utilities
- Viewer creation helpers

References:
    - Original: ManiSkill-main/mani_skill/utils/genesis_utils.py
    - Genesis: https://github.com/Genesis-Embodied-AI/Genesis
"""

from __future__ import annotations

from typing import Any, Optional, Union

# Optional dependencies
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

# Genesis imports
try:
    import genesis as gs

    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False
    gs = None

# Import Pose from genesis_compat
try:
    from .genesis_compat import GENESIS_RENDER_SYSTEM, Pose, matrix_to_quaternion
except ImportError:
    # Fallback if not available
    class Pose:  # type: ignore[no-redef]
        """Fallback pose container."""

        def __init__(self, raw_pose):
            self.raw_pose = raw_pose

        @classmethod
        def create_from_pq(cls, p=None, q=None, device=None):
            if HAS_TORCH:
                if p is None:
                    p = torch.zeros((1, 3), device=device)
                if q is None:
                    q = torch.zeros((1, 4), device=device)
                    q[:, 0] = 1
                if not isinstance(p, torch.Tensor):
                    p = torch.tensor(p, dtype=torch.float32, device=device)
                if not isinstance(q, torch.Tensor):
                    q = torch.tensor(q, dtype=torch.float32, device=device)
                if p.ndim == 1:
                    p = p.unsqueeze(0)
                if q.ndim == 1:
                    q = q.unsqueeze(0)
                raw_pose = torch.cat([p, q], dim=-1)
            else:
                if p is None:
                    p = [[0.0, 0.0, 0.0]]
                if q is None:
                    q = [[1.0, 0.0, 0.0, 0.0]]
                raw_pose = list(p) + list(q)  # type: ignore[assignment]
            return cls(raw_pose)

    def matrix_to_quaternion(matrix):
        if HAS_TORCH:
            return torch.tensor([1.0, 0.0, 0.0, 0.0])
        return [1.0, 0.0, 0.0, 0.0]

    GENESIS_RENDER_SYSTEM = "1.0"


ArrayLike = Any  # Union[np.ndarray, torch.Tensor]


def genesis_pose_to_opencv_extrinsic(
    genesis_pose_matrix: ArrayLike,
) -> Optional[ArrayLike]:
    """Convert Genesis pose matrix to OpenCV extrinsic matrix.

    Args:
        genesis_pose_matrix: Genesis pose matrix (4x4).

    Returns:
        OpenCV extrinsic matrix (4x4).
    """
    if not HAS_NUMPY:
        return None

    genesis2opencv = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    ex = genesis2opencv @ np.linalg.inv(genesis_pose_matrix)  # world -> camera
    return ex


def look_at(
    eye: Union[list, ArrayLike],
    target: Union[list, ArrayLike],
    up: Union[list, ArrayLike] = (0, 0, 1),
    device: Optional[str] = None,
) -> Pose:
    """Get the camera pose in Genesis by the Look-At method.

    Note:
        https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
        The Genesis camera follows the convention: (forward, right, up) = (x, -y, z)
        while the OpenGL camera follows (forward, right, up) = (-z, x, y)
        Note that the camera coordinate system (OpenGL) is left-hand.

    Args:
        eye: Camera location [x, y, z].
        target: Looking-at location [x, y, z].
        up: A general direction of "up" from the camera (default: [0, 0, 1]).
        device: Device to put the pose on (for torch tensors).

    Returns:
        Pose: Camera pose.
    """
    if not HAS_TORCH:
        # Fallback without torch - return simple pose
        return Pose.create_from_pq(p=eye, q=[1, 0, 0, 0], device=device)

    # Convert inputs to tensors
    if not isinstance(eye, torch.Tensor):
        eye = torch.tensor(eye, dtype=torch.float32, device=device)
        assert eye.ndim == 1, eye.ndim
        assert len(eye) == 3, len(eye)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target, dtype=torch.float32, device=device)
        assert target.ndim == 1, target.ndim
        assert len(target) == 3, len(target)
    if not isinstance(up, torch.Tensor):
        up = torch.tensor(up, dtype=torch.float32, device=device)
        assert up.ndim == 1, up.ndim
        assert len(up) == 3, len(up)

    def normalize_tensor(x, eps=1e-6):
        x = x.view(-1, 3)
        norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        zero_vectors = norm < eps
        x = torch.where(zero_vectors, torch.zeros_like(x), x / (norm + eps))
        return x

    forward = normalize_tensor(target - eye)
    up = normalize_tensor(up)
    left = torch.cross(up, forward, dim=-1)
    left = normalize_tensor(left)
    up = torch.cross(forward, left, dim=-1)

    # Create rotation matrix
    rotation = torch.stack([forward, left, up], dim=-1)

    return Pose.create_from_pq(p=eye, q=matrix_to_quaternion(rotation))


def hex2rgba(h: str, correction: bool = True) -> Optional[ArrayLike]:
    """Convert hex color to RGBA.

    Args:
        h: Hex color string (e.g., "#FF0000").
        correction: Whether to apply gamma correction.

    Returns:
        RGBA color as numpy array [r, g, b, a] in range [0, 1].
    """
    if not HAS_NUMPY:
        return None

    # https://stackoverflow.com/a/29643643
    h = h.lstrip("#")
    r, g, b = tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
    rgba = np.array([r, g, b, 1.0])
    if correction:  # reverse gamma correction in genesis
        rgba = rgba**2.2
    return rgba


def rgba2hex(rgba: ArrayLike) -> str:
    """Convert RGBA to hex color.

    Args:
        rgba: RGBA values in range [0, 1] or [0, 255].

    Returns:
        Hex color string (e.g., "#FF0000").
    """
    if HAS_NUMPY:
        rgba = np.array(rgba)
        # Check if values are in [0, 1]
        if rgba.max() <= 1.0:
            rgba = (rgba * 255).astype(int)
        return "#{:02x}{:02x}{:02x}".format(int(rgba[0]), int(rgba[1]), int(rgba[2]))
    else:
        # Manual conversion
        r, g, b = rgba[:3]
        if max(r, g, b) <= 1.0:
            r, g, b = int(r * 255), int(g * 255), int(b * 255)
        return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))


def spherical_to_cartesian(
    radius: float,
    azimuth: float,
    elevation: float,
    target: Union[list, ArrayLike] = (0, 0, 0),
) -> ArrayLike:
    """Convert spherical coordinates to cartesian position.

    Useful for positioning cameras around an object.

    Args:
        radius: Distance from target.
        azimuth: Azimuth angle in radians (0 = positive x-axis).
        elevation: Elevation angle in radians (0 = horizontal).
        target: Target position to look at.

    Returns:
        Camera position [x, y, z].
    """
    if HAS_NUMPY:
        x = radius * np.cos(elevation) * np.cos(azimuth)
        y = radius * np.cos(elevation) * np.sin(azimuth)
        z = radius * np.sin(elevation)
        return np.array([x, y, z]) + np.array(target)
    else:
        import math

        x = radius * math.cos(elevation) * math.cos(azimuth)
        y = radius * math.cos(elevation) * math.sin(azimuth)
        z = radius * math.sin(elevation)
        return [x + target[0], y + target[1], z + target[2]]


def compute_fovy(focal_length: float, sensor_height: float) -> float:
    """Compute vertical field of view from focal length.

    Args:
        focal_length: Focal length in mm.
        sensor_height: Sensor height in mm.

    Returns:
        Vertical FOV in degrees.
    """
    if HAS_NUMPY:
        return float(2 * np.arctan(sensor_height / (2 * focal_length)) * 180 / np.pi)
    else:
        import math

        return 2 * math.atan(sensor_height / (2 * focal_length)) * 180 / math.pi


def get_camera_rays(
    camera_pose: ArrayLike,
    intrinsics: ArrayLike,
    image_size: tuple[int, int],
) -> tuple[ArrayLike, ArrayLike]:
    """Compute camera rays for each pixel.

    Args:
        camera_pose: Camera pose matrix (4x4).
        intrinsics: Camera intrinsics matrix (3x3).
        image_size: (height, width) of the image.

    Returns:
        Tuple of (ray_origins, ray_directions) each with shape (H, W, 3).
    """
    if not HAS_NUMPY:
        return None, None

    height, width = image_size

    # Create pixel grid
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Convert to normalized camera coordinates
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    x = (u - cx) / fx
    y = (v - cy) / fy
    z = np.ones_like(x)

    # Ray directions in camera space
    directions = np.stack([x, y, z], axis=-1)
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    # Transform to world space
    rotation = camera_pose[:3, :3]
    t = camera_pose[:3, 3]

    ray_directions = (rotation @ directions.reshape(-1, 3).T).T.reshape(
        height, width, 3
    )
    ray_origins = np.broadcast_to(t, ray_directions.shape)

    return ray_origins, ray_directions


# =============================================================================
# Viewer Creation Helpers
# =============================================================================


def create_viewer(viewer_camera_config) -> Optional[Any]:
    """Creates a viewer with the given camera config.

    Args:
        viewer_camera_config: Configuration for the viewer camera.

    Returns:
        Viewer object or None if Genesis not available.
    """
    if not HAS_GENESIS:
        return None

    import sys

    if GENESIS_RENDER_SYSTEM == "1.0":
        if hasattr(gs, "render") and hasattr(gs.render, "set_viewer_shader_dir"):
            gs.render.set_viewer_shader_dir(
                viewer_camera_config.shader_config.shader_pack
            )
            if viewer_camera_config.shader_config.shader_pack[:2] == "rt":
                gs.render.set_ray_tracing_denoiser(
                    viewer_camera_config.shader_config.shader_pack_config[
                        "ray_tracing_denoiser"
                    ]
                )
                gs.render.set_ray_tracing_path_depth(
                    viewer_camera_config.shader_config.shader_pack_config[
                        "ray_tracing_path_depth"
                    ]
                )
                gs.render.set_ray_tracing_samples_per_pixel(
                    viewer_camera_config.shader_config.shader_pack_config[
                        "ray_tracing_samples_per_pixel"
                    ]
                )

        if hasattr(gs, "Viewer"):
            viewer = gs.Viewer(
                resolutions=(viewer_camera_config.width, viewer_camera_config.height)
            )
            if sys.platform == "darwin":  # macOS
                if hasattr(viewer, "window") and hasattr(
                    viewer.window, "set_content_scale"
                ):
                    viewer.window.set_content_scale(1)
            return viewer

    elif GENESIS_RENDER_SYSTEM == "1.1":
        if (
            hasattr(gs, "Viewer")
            and hasattr(gs, "render")
            and hasattr(gs.render, "get_shader_pack")
        ):
            viewer = gs.Viewer(
                resolutions=(viewer_camera_config.width, viewer_camera_config.height),
                shader_pack=gs.render.get_shader_pack(
                    viewer_camera_config.shader_config.shader_pack
                ),
            )
            return viewer

    return None


__all__ = [
    # Pose conversion
    "genesis_pose_to_opencv_extrinsic",
    # Camera positioning
    "look_at",
    "spherical_to_cartesian",
    # Color utilities
    "hex2rgba",
    "rgba2hex",
    # Camera parameters
    "compute_fovy",
    "get_camera_rays",
    # Viewer
    "create_viewer",
    # Re-exports
    "Pose",
    "GENESIS_RENDER_SYSTEM",
]
