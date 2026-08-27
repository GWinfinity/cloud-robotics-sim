"""Mesh loading, repair, normalization, sampling and voxelization utilities."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a mesh file into a single ``trimesh.Trimesh``.

    Scenes are concatenated into one mesh. Faces are triangulated.

    Args:
        path: Path to an ``.obj`` / ``.glb`` / ``.off`` / ``.stl`` file.

    Returns:
        Loaded triangle mesh.
    """
    loaded = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Could not load a triangle mesh from {path}")
    return loaded


def repair_nonmanifold(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Repair non-manifold sites by splitting edges and vertices.

    Two steps, mirroring the dataset preprocessing of Sec. 2.1:

    1. *Edge splitting*: an edge shared by more than two faces keeps the
       first pair; every further pair of faces is moved onto a duplicated
       copy of the edge (both endpoints duplicated).
    2. *Vertex fan splitting*: faces incident to a vertex are grouped
       into fans connected through edges containing that vertex; each
       extra fan receives its own vertex copy, so every remaining vertex
       has a single manifold fan as required by the halfedge-successor
       representation.

    Args:
        mesh: Input mesh.

    Returns:
        A new mesh with duplicated vertices at non-manifold sites.
    """
    mesh = mesh.copy()
    mesh.process(validate=False)
    faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    vertices = [v for v in np.asarray(mesh.vertices, dtype=np.float64)]

    def _add_vertex(src: int) -> int:
        vertices.append(vertices[src])
        return len(vertices) - 1

    # --- Step 1: split edges with > 2 incident faces. ----------------------
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_faces[(min(u, v), max(u, v))].append(fi)
    for (u, v), fs in edge_faces.items():
        if len(fs) <= 2:
            continue
        for k in range(2, len(fs), 2):
            u2, v2 = _add_vertex(u), _add_vertex(v)
            for fi in fs[k : k + 2]:
                row = faces[fi]
                row[row == u] = u2
                row[row == v] = v2

    # --- Step 2: split vertices with multiple disconnected fans. -----------
    vert_faces: dict[int, list[int]] = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        vert_faces[a].append(fi)
        vert_faces[b].append(fi)
        vert_faces[c].append(fi)

    for v, fs in vert_faces.items():
        if len(fs) <= 1:
            continue
        # Local union-find over the faces incident to v.
        parent = {fi: fi for fi in fs}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        by_edge: dict[int, list[int]] = defaultdict(list)
        for fi in fs:
            row = faces[fi]
            others = [int(x) for x in row if x != v]
            for w in others:
                by_edge[w].append(fi)
        for group in by_edge.values():
            for k in range(1, len(group)):
                parent[find(group[0])] = find(group[k])

        fans: dict[int, list[int]] = defaultdict(list)
        for fi in fs:
            fans[find(fi)].append(fi)
        if len(fans) <= 1:
            continue
        fan_list = sorted(fans.values(), key=len, reverse=True)
        for extra in fan_list[1:]:
            v2 = _add_vertex(v)
            for fi in extra:
                row = faces[fi]
                row[row == v] = v2

    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=faces,
        process=False,
    )


def normalize_vertices(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalize vertices into the unit cube ``[0, 1]^3``.

    Args:
        vertices: ``(V, 3)`` positions.

    Returns:
        ``(normalized, origin, scale)`` such that
        ``vertices = normalized * scale + origin``.
    """
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    scale = float(max(hi - lo).item())
    if scale <= 0:
        scale = 1.0
    normalized = (vertices - lo) / scale
    return normalized, lo, scale


def sample_surface(
    mesh: trimesh.Trimesh,
    count: int,
    edge_ratio: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample surface points and normals, preferentially along mesh edges.

    The Mesh VAE encoder draws surface samples preferentially along mesh
    edges (Sec. 2.1) so that the voxel context captures tessellation
    structure, not just the smooth surface.

    Args:
        mesh: Source mesh.
        count: Total number of samples.
        edge_ratio: Fraction of samples placed on mesh edges.
        seed: RNG seed.

    Returns:
        ``(points, normals)`` float arrays of shape ``(count, 3)``.
    """
    rng = np.random.default_rng(seed)
    n_edge = int(count * edge_ratio)
    n_face = count - n_edge

    points = np.zeros((count, 3))
    normals = np.zeros((count, 3))
    if n_face > 0 and len(mesh.faces) > 0:
        pts, face_idx = trimesh.sample.sample_surface(mesh, n_face, seed=seed)
        points[:n_face] = pts
        normals[:n_face] = mesh.face_normals[face_idx]
    if n_edge > 0 and len(mesh.faces) > 0:
        edges = mesh.edges_unique
        lengths = mesh.edges_unique_length
        probs = lengths / lengths.sum()
        pick = rng.choice(len(edges), size=n_edge, p=probs)
        t = rng.random((n_edge, 1))
        e = edges[pick]
        pts = mesh.vertices[e[:, 0]] * (1 - t) + mesh.vertices[e[:, 1]] * t
        # Normal: average of the normals of the faces sharing the edge.
        nrm = np.zeros((n_edge, 3))
        for row, ei in enumerate(pick):
            faces_for_edge = _faces_of_edge(mesh, edges[ei])
            nrm[row] = mesh.face_normals[faces_for_edge].mean(axis=0)
        norm = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm = nrm / np.maximum(norm, 1e-12)
        points[n_face:] = pts
        normals[n_face:] = nrm
    return points.astype(np.float32), normals.astype(np.float32)


def _faces_of_edge(mesh: trimesh.Trimesh, edge: np.ndarray) -> np.ndarray:
    """Return face indices incident to an undirected edge ``(u, v)``."""
    u, v = int(edge[0]), int(edge[1])
    f0 = set(mesh.vertex_faces[u][mesh.vertex_faces[u] >= 0])
    f1 = set(mesh.vertex_faces[v][mesh.vertex_faces[v] >= 0])
    return np.asarray(sorted(f0 & f1), dtype=np.int64)


def voxelize(mesh: trimesh.Trimesh, res: int = 64) -> np.ndarray:
    """Voxelize a (possibly non-watertight) mesh into a ``res^3`` grid.

    Vertices are assumed normalized to ``[0, 1]^3``. Surface voxels are
    rasterized from the triangles; watertight interiors are filled.

    Args:
        mesh: Mesh with vertices in ``[0, 1]^3``.
        res: Grid resolution.

    Returns:
        Boolean array of shape ``(res, res, res)``.
    """
    grid = np.zeros((res, res, res), dtype=bool)
    try:
        vox = mesh.voxelized(1.0 / res)
        filled = vox.fill()
        idx = np.floor(filled.points / (1.0 / res)).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < res), axis=1)
        idx = idx[inside]
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    except Exception:  # noqa: BLE001 - voxelization is best-effort
        pass
    # Always rasterize vertex-occupied cells so open surfaces are covered.
    vidx = np.floor(np.asarray(mesh.vertices) * res).astype(np.int64)
    vidx = np.clip(vidx, 0, res - 1)
    grid[vidx[:, 0], vidx[:, 1], vidx[:, 2]] = True
    return grid
