"""
Geometry utilities for sampling and calculations.
"""

import numpy as np
from typing import Tuple


def sample_sphere(radius: float, num_points: int = 1) -> np.ndarray:
    """
    Sample random points on a sphere surface.
    
    Args:
        radius: Sphere radius
        num_points: Number of points to sample
    
    Returns:
        Array of shape (num_points, 3) with sampled points
    """
    # Uniform sampling on sphere
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(0, np.pi, num_points)
    
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    
    return np.stack([x, y, z], axis=-1)


def sample_cylinder(radius: float, height: float, num_points: int = 1) -> np.ndarray:
    """
    Sample random points inside a cylinder.
    
    Args:
        radius: Cylinder radius
        height: Cylinder height
        num_points: Number of points to sample
    
    Returns:
        Array of shape (num_points, 3) with sampled points
    """
    # Sample random radius and angle
    r = radius * np.sqrt(np.random.uniform(0, 1, num_points))
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    z = np.random.uniform(-height/2, height/2, num_points)
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    return np.stack([x, y, z], axis=-1)


def sample_box(size: Tuple[float, float, float], num_points: int = 1) -> np.ndarray:
    """
    Sample random points inside a box.
    
    Args:
        size: Box size (x, y, z)
        num_points: Number of points to sample
    
    Returns:
        Array of shape (num_points, 3) with sampled points
    """
    x = np.random.uniform(-size[0]/2, size[0]/2, num_points)
    y = np.random.uniform(-size[1]/2, size[1]/2, num_points)
    z = np.random.uniform(-size[2]/2, size[2]/2, num_points)
    
    return np.stack([x, y, z], axis=-1)


def compute_bounding_box(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute axis-aligned bounding box.
    
    Args:
        points: Array of points with shape (N, 3)
    
    Returns:
        Min and max corners of bounding box
    """
    min_corner = np.min(points, axis=0)
    max_corner = np.max(points, axis=0)
    return min_corner, max_corner


def ray_plane_intersection(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray
) -> np.ndarray:
    """
    Compute intersection of ray with plane.
    
    Args:
        ray_origin: Ray origin point
        ray_dir: Ray direction (should be normalized)
        plane_point: Point on plane
        plane_normal: Plane normal
    
    Returns:
        Intersection point or None if no intersection
    """
    denom = np.dot(ray_dir, plane_normal)
    if abs(denom) < 1e-6:
        return None
    
    t = np.dot(plane_point - ray_origin, plane_normal) / denom
    if t < 0:
        return None
    
    return ray_origin + t * ray_dir
