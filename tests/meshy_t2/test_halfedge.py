"""Tests for halfedge-successor topology extraction and reconstruction."""

import numpy as np
import trimesh

from cloud_robotics_sim.meshy_t2.geometry.halfedge import (
    boundary_vertices,
    extract_topology,
    faces_from_topology,
    orient_faces_consistently,
)


def _face_set(faces: np.ndarray) -> set[tuple[int, int, int]]:
    """Canonical (sorted) face set for orientation-agnostic comparison."""
    return {tuple(sorted(map(int, f))) for f in faces}


def test_roundtrip_box():
    """Check roundtrip box."""
    mesh = trimesh.creation.box()
    faces = orient_faces_consistently(mesh.vertices, mesh.faces)
    edges, neighbors, successors = extract_topology(faces, len(mesh.vertices))
    assert edges.shape == (18, 2)  # triangulated cube: 12 + 6 diagonal edges
    rebuilt = faces_from_topology(neighbors, successors)
    assert _face_set(rebuilt) == _face_set(faces)


def test_roundtrip_icosphere():
    """Check roundtrip icosphere."""
    mesh = trimesh.creation.icosphere(subdivisions=2)
    faces = orient_faces_consistently(mesh.vertices, mesh.faces)
    _, neighbors, successors = extract_topology(faces, len(mesh.vertices))
    rebuilt = faces_from_topology(neighbors, successors)
    assert _face_set(rebuilt) == _face_set(faces)
    # Watertight: no boundary vertices, every fan is a closed cycle.
    assert not boundary_vertices(neighbors, successors).any()


def test_boundary_null_open_grid():
    """Check boundary null open grid."""
    # Open surface (two triangles): boundary fans use NULL transitions.
    grid = trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [1, 3, 2]]),
        process=False,
    )
    faces = orient_faces_consistently(grid.vertices, grid.faces)
    _, neighbors, successors = extract_topology(faces, len(grid.vertices))
    mask = boundary_vertices(neighbors, successors)
    assert mask.any()  # all four vertices lie on the boundary
    rebuilt = faces_from_topology(neighbors, successors)
    assert _face_set(rebuilt) == _face_set(faces)


def test_multi_part_components():
    """Check multi part components."""
    parts = [
        trimesh.creation.box(),
        trimesh.creation.icosphere(subdivisions=1),
    ]
    parts[1].apply_translation([5, 0, 0])
    mesh = trimesh.util.concatenate(parts)
    faces = orient_faces_consistently(mesh.vertices, mesh.faces)
    edges, neighbors, successors = extract_topology(faces, len(mesh.vertices))
    rebuilt = faces_from_topology(neighbors, successors)
    assert _face_set(rebuilt) == _face_set(faces)
    # The edge set has two connected components.
    import networkx as nx

    graph = nx.Graph()
    graph.add_edges_from(edges.tolist())
    assert nx.number_connected_components(graph) == 2
