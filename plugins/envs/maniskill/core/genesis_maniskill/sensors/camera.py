"""
Camera sensor for Genesis ManiSkill.
"""

from dataclasses import dataclass
from typing import Tuple
import torch
import genesis as gs


@dataclass
class CameraConfig:
    """Camera configuration."""
    width: int = 128
    height: int = 128
    fov: float = 45.0
    near: float = 0.01
    far: float = 100.0


class Camera:
    """
    Camera sensor wrapper for Genesis.
    """
    
    def __init__(
        self,
        scene: gs.Scene,
        config: CameraConfig = None,
        pos: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        lookat: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.scene = scene
        self.config = config or CameraConfig()
        
        # Create Genesis camera
        self.camera = scene.add_camera(
            res=(self.config.width, self.config.height),
            pos=pos,
            lookat=lookat,
            fov=self.config.fov,
        )
    
    def get_rgb(self) -> torch.Tensor:
        """Get RGB image."""
        rgb = self.camera.render(rgb=True)
        return rgb
    
    def get_depth(self) -> torch.Tensor:
        """Get depth image."""
        depth = self.camera.render(depth=True)
        return depth
    
    def get_rgbd(self) -> torch.Tensor:
        """Get RGB-D image."""
        rgb, depth = self.camera.render(rgb=True, depth=True)
        # Concatenate RGB and depth
        rgbd = torch.cat([rgb, depth.unsqueeze(-1)], dim=-1)
        return rgbd
    
    def get_segmentation(self) -> torch.Tensor:
        """Get segmentation mask."""
        seg = self.camera.render(segmentation=True)
        return seg
    
    def set_pose(self, pos: Tuple[float, float, float], lookat: Tuple[float, float, float]):
        """Set camera pose."""
        self.camera.set_pose(pos=pos, lookat=lookat)
