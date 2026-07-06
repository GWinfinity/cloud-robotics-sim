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

"""Rendering utilities for Genesis simulations.

This module provides rendering-related utilities adapted from ManiSkill's
genesis_utils.py to work with genesis-cloud-sim's architecture.

Includes:
- Material property setters
- Articulation material configuration
- Shader configuration helpers
- Texture utilities

References:
    - Original: ManiSkill-main/mani_skill/utils/genesis_utils.py
    - Genesis: https://github.com/Genesis-Embodied-AI/Genesis
"""

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

logger = logging.getLogger(__name__)

# Optional dependencies
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

# Genesis imports
try:
    import genesis as gs

    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False
    gs = None

# Try to import color utilities from camera module
try:
    from .camera import hex2rgba, rgba2hex
except ImportError:
    # Fallback implementations
    def hex2rgba(h: str, correction: bool = True):
        h = h.lstrip("#")
        r = int(h[0:2], 16) / 255
        g = int(h[2:4], 16) / 255
        b = int(h[4:6], 16) / 255
        rgba = [r, g, b, 1.0]
        if correction:
            rgba = [c**2.2 for c in rgba[:3]] + [1.0]
        return rgba

    def rgba2hex(rgba):
        r, g, b = rgba[:3]
        if max(r, g, b) <= 1.0:
            r, g, b = int(r * 255), int(g * 255), int(b * 255)
        return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))


# =============================================================================
# Material Utilities
# =============================================================================


def set_render_material(material: Any, **kwargs) -> Any:
    """Set render material properties.

    Args:
        material: Render material object.
        **kwargs: Material properties to set.
            - color: Base color (RGBA list or hex string)
            - metallic: Metallic factor (0-1)
            - roughness: Roughness factor (0-1)
            - specular: Specular factor (0-1)
            - emission: Emission color/intensity

    Returns:
        The modified material object.
    """
    for k, v in kwargs.items():
        if k == "color":
            # Handle hex color strings
            if isinstance(v, str):
                v = hex2rgba(v)
            if hasattr(material, "set_base_color"):
                material.set_base_color(v)
            elif hasattr(material, "base_color"):
                material.base_color = v
        elif k == "metallic":
            if hasattr(material, "set_metallic"):
                material.set_metallic(v)
            elif hasattr(material, "metallic"):
                material.metallic = v
        elif k == "roughness":
            if hasattr(material, "set_roughness"):
                material.set_roughness(v)
            elif hasattr(material, "roughness"):
                material.roughness = v
        elif k == "specular":
            if hasattr(material, "set_specular"):
                material.set_specular(v)
            elif hasattr(material, "specular"):
                material.specular = v
        elif k == "emission":
            if hasattr(material, "set_emission"):
                material.set_emission(v)
            elif hasattr(material, "emission"):
                material.emission = v
        else:
            # Generic attribute setting
            setattr(material, k, v)
    return material


def set_articulation_render_material(articulation: Any, **kwargs):
    """Set render material properties for an articulation.

    Note:
        Avoid using this function when using render server as it may
        not play nice. Prefer editing URDF files directly.

    Args:
        articulation: Articulation object.
        **kwargs: Material properties to set (passed to set_render_material).
    """
    if not hasattr(articulation, "get_links"):
        return

    for link in articulation.get_links():
        if hasattr(link, "entity"):
            entity = link.entity
            if hasattr(entity, "find_component_by_type"):
                # Try to find render component
                try:
                    render_component = entity.find_component_by_type(
                        gs.RenderBodyComponent
                    )
                    if render_component is not None:
                        for s in getattr(render_component, "render_shapes", []):
                            if hasattr(s, "parts"):
                                for part in s.parts:
                                    if hasattr(part, "material"):
                                        set_render_material(part.material, **kwargs)
                            else:
                                # Directly set material if no parts
                                if hasattr(s, "material"):
                                    set_render_material(s.material, **kwargs)
                except Exception:
                    pass


def set_entity_color(entity: Any, color: Union[str, List], recursive: bool = True):
    """Set color for an entity.

    Args:
        entity: Genesis entity.
        color: Color as hex string or RGBA list.
        recursive: Whether to apply to child entities.
    """
    if isinstance(color, str):
        color = hex2rgba(color)

    # Try different methods to set color
    if hasattr(entity, "set_color"):
        entity.set_color(color)
    elif hasattr(entity, "color"):
        entity.color = color

    # Apply to render shapes
    if hasattr(entity, "render_shapes"):
        for shape in entity.render_shapes:
            if hasattr(shape, "material") and shape.material is not None:
                set_render_material(shape.material, color=color)

    # Recursive application
    if recursive and hasattr(entity, "get_children"):
        for child in entity.get_children():
            set_entity_color(child, color, recursive)


# =============================================================================
# Shader and Rendering Configuration
# =============================================================================


class ShaderConfig:
    """Configuration for rendering shaders."""

    def __init__(
        self,
        shader_pack: str = "default",
        ray_tracing_denoiser: str = "optix",
        ray_tracing_path_depth: int = 4,
        ray_tracing_samples_per_pixel: int = 32,
    ):
        """Args:
        shader_pack: Shader pack to use ("default", "rt", "rt-fast", etc.)
        ray_tracing_denoiser: Denoiser for ray tracing ("optix", "oidn")
        ray_tracing_path_depth: Max ray tracing path depth
        ray_tracing_samples_per_pixel: Samples per pixel for ray tracing
        """
        self.shader_pack = shader_pack
        self.shader_pack_config = {
            "ray_tracing_denoiser": ray_tracing_denoiser,
            "ray_tracing_path_depth": ray_tracing_path_depth,
            "ray_tracing_samples_per_pixel": ray_tracing_samples_per_pixel,
        }


def configure_rendering(
    shader_pack: str = "default", enable_ray_tracing: bool = False, **kwargs
) -> bool:
    """Configure global rendering settings.

    Args:
        shader_pack: Shader pack name.
        enable_ray_tracing: Whether to enable ray tracing.
        **kwargs: Additional settings.

    Returns:
        True if configuration succeeded.
    """
    if not HAS_GENESIS:
        return False

    try:
        if hasattr(gs, "render"):
            if hasattr(gs.render, "set_viewer_shader_dir"):
                gs.render.set_viewer_shader_dir(shader_pack)

            if enable_ray_tracing and shader_pack.startswith("rt"):
                if hasattr(gs.render, "set_ray_tracing_denoiser"):
                    gs.render.set_ray_tracing_denoiser(kwargs.get("denoiser", "optix"))
                if hasattr(gs.render, "set_ray_tracing_path_depth"):
                    gs.render.set_ray_tracing_path_depth(kwargs.get("path_depth", 4))
                if hasattr(gs.render, "set_ray_tracing_samples_per_pixel"):
                    gs.render.set_ray_tracing_samples_per_pixel(
                        kwargs.get("samples_per_pixel", 32)
                    )
        return True
    except Exception as e:
        logger.warning("Failed to configure rendering: %s", e)
        return False


# =============================================================================
# Texture Utilities
# =============================================================================


def load_texture(path: Union[str, Path], **kwargs) -> Optional[Any]:
    """Load a texture from file.

    Args:
        path: Path to texture file.
        **kwargs: Additional loading options.

    Returns:
        Texture object or None.
    """
    if not HAS_GENESIS:
        return None

    path = Path(path)
    if not path.exists():
        return None

    try:
        # Try different methods to load texture
        if hasattr(gs, "Texture"):
            return gs.Texture(str(path), **kwargs)
        elif hasattr(gs.render, "Texture"):
            return gs.render.Texture(str(path), **kwargs)
    except Exception as e:
        logger.warning("Failed to load texture %s: %s", path, e)

    return None


def create_checkerboard_texture(
    size: int = 512,
    check_size: int = 64,
    color1: List[float] = [1.0, 1.0, 1.0],
    color2: List[float] = [0.5, 0.5, 0.5],
) -> Optional[np.ndarray]:
    """Create a checkerboard texture pattern.

    Args:
        size: Texture size (square).
        check_size: Size of each checker square.
        color1: First color (RGB).
        color2: Second color (RGB).

    Returns:
        Texture array or None.
    """
    if not HAS_NUMPY:
        return None

    texture = np.zeros((size, size, 3), dtype=np.float32)

    for i in range(size):
        for j in range(size):
            x = i // check_size
            y = j // check_size
            if (x + y) % 2 == 0:
                texture[i, j] = color1
            else:
                texture[i, j] = color2

    return texture


# =============================================================================
# Screenshot and Recording
# =============================================================================


def save_screenshot(camera_or_viewer, path: Union[str, Path], rgb: bool = True):
    """Save a screenshot from camera or viewer.

    Args:
        camera_or_viewer: Camera or viewer object.
        path: Output file path.
        rgb: Whether to capture RGB.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Try to render and save
        if hasattr(camera_or_viewer, "render"):
            img = camera_or_viewer.render(rgb=rgb)
            if HAS_NUMPY and isinstance(img, np.ndarray):
                # Save using PIL or similar
                try:
                    from PIL import Image

                    if img.dtype == np.float32 or img.dtype == np.float64:
                        img = (img * 255).astype(np.uint8)
                    Image.fromarray(img).save(path)
                except ImportError:
                    # Fallback: save as numpy array
                    np.save(path.with_suffix(".npy"), img)
    except Exception as e:
        logger.warning("Failed to save screenshot: %s", e)


def start_recording(viewer, path: Union[str, Path], fps: int = 30):
    """Start recording from viewer.

    Args:
        viewer: Viewer object.
        path: Output video path.
        fps: Frames per second.

    Returns:
        Recording handle or None.
    """
    try:
        if hasattr(viewer, "start_recording"):
            return viewer.start_recording(str(path), fps=fps)
    except Exception as e:
        logger.warning("Failed to start recording: %s", e)
    return None


def stop_recording(recording_handle):
    """Stop recording.

    Args:
        recording_handle: Handle from start_recording.
    """
    try:
        if recording_handle and hasattr(recording_handle, "stop"):
            recording_handle.stop()
    except Exception as e:
        logger.warning("Failed to stop recording: %s", e)


__all__ = [
    # Material
    "set_render_material",
    "set_articulation_render_material",
    "set_entity_color",
    # Configuration
    "ShaderConfig",
    "configure_rendering",
    # Texture
    "load_texture",
    "create_checkerboard_texture",
    # Recording
    "save_screenshot",
    "start_recording",
    "stop_recording",
    # Re-exports
    "hex2rgba",
    "rgba2hex",
]
