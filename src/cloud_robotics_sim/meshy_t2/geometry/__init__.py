"""Geometry utilities for the Meshy T2 reproduction."""

from .assembly import assemble_mesh
from .halfedge import (
    boundary_vertices,
    extract_topology,
    faces_from_topology,
    orient_faces_consistently,
)
from .mesh_utils import (
    load_mesh,
    normalize_vertices,
    repair_nonmanifold,
    sample_surface,
    voxelize,
)

__all__ = [
    "assemble_mesh",
    "boundary_vertices",
    "extract_topology",
    "faces_from_topology",
    "load_mesh",
    "normalize_vertices",
    "orient_faces_consistently",
    "repair_nonmanifold",
    "sample_surface",
    "voxelize",
]
