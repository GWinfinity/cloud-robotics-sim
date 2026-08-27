"""Tests for mesh repair, normalization, sampling and voxelization."""

import numpy as np
import trimesh

from cloud_robotics_sim.meshy_t2.data import prepare_mesh_sample
from cloud_robotics_sim.meshy_t2.geometry.mesh_utils import (
    normalize_vertices,
    repair_nonmanifold,
    sample_surface,
    voxelize,
)


def test_repair_nonmanifold_edge_split():
    """Check repair nonmanifold edge split."""
    # Two tetrahedra sharing an edge -> the shared edge has 4 incident
    # faces (non-manifold). Fan splitting must repair it.
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    t1 = np.array([0.0, 1.0, 0.0])
    t2 = np.array([0.0, -1.0, 0.0])
    apex1 = np.array([0.25, 0.0, 1.0])
    apex2 = np.array([0.25, 0.0, -1.0])
    # Two tetrahedra sharing the (a, b) edge region: build explicit
    # non-manifold fans around the a-b edge instead.
    verts = np.array([a, b, t1, t2, apex1])
    faces = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 4],
            [1, 4, 2],
            [0, 4, 3],
            [1, 3, 4],
        ]
    )
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    # Edge (0, 1) is shared by faces 0 and 1 only — make it non-manifold
    # by adding two more faces on the same edge with a second apex.
    verts2 = np.vstack([verts, [apex2]])
    faces2 = np.vstack([faces, [[0, 5, 1], [0, 1, 5]]])
    mesh = trimesh.Trimesh(vertices=verts2, faces=faces2, process=False)
    # The shared edge (0, 1) has four incident faces: non-manifold.
    repaired = repair_nonmanifold(mesh)
    # After splitting, every edge is shared by at most 2 faces.
    from collections import Counter

    count = Counter()
    for f in repaired.faces:
        for u, v in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            count[(min(u, v), max(u, v))] += 1
    assert max(count.values()) <= 2


def test_normalize_vertices_unit_cube():
    """Check normalize vertices unit cube."""
    rng = np.random.default_rng(0)
    verts = rng.normal(size=(50, 3)) * 7 + 3
    norm, origin, scale = normalize_vertices(verts)
    assert norm.min() >= 0 and norm.max() <= 1 + 1e-9
    np.testing.assert_allclose(norm * scale + origin, verts, rtol=1e-9)


def test_sample_surface_counts_and_normals():
    """Check sample surface counts and normals."""
    mesh = trimesh.creation.icosphere(subdivisions=2)
    points, normals = sample_surface(mesh, 256, edge_ratio=0.5, seed=1)
    assert points.shape == (256, 3)
    assert normals.shape == (256, 3)
    # Normals are unit-length.
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-4)


def test_voxelize_filled_sphere():
    """Check voxelize filled sphere."""
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.45)
    mesh.apply_translation([0.5, 0.5, 0.5])
    grid = voxelize(mesh, res=32)
    assert grid.shape == (32, 32, 32)
    assert grid.dtype == bool
    # Center of a filled sphere must be occupied.
    assert grid[16, 16, 16]


def test_prepare_mesh_sample_consistency():
    """Check prepare mesh sample consistency."""
    mesh = trimesh.creation.icosphere(subdivisions=1)
    sample = prepare_mesh_sample(mesh, num_samples=128, voxel_res=32, seed=0)
    v = sample.num_vertices
    assert sample.vertices.min() >= 0 and sample.vertices.max() <= 1
    assert sample.points.shape == (128, 3)
    assert sample.occupancy.shape == (32, 32, 32)
    assert len(sample.neighbors) == v == len(sample.successors)
    for nbrs, succ in zip(sample.neighbors, sample.successors, strict=True):
        assert len(succ) == len(nbrs) + 1
