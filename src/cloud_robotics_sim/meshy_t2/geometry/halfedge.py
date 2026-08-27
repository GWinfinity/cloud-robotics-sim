"""Halfedge successor topology for the vertex-set mesh VAE.

Reproduces the SpaceMesh-style topology representation of Meshy T2
(Sec. 2.1): the undirected edge set plus, per vertex, a successor
permutation over its neighbors extended with a ``NULL`` element so open
boundaries are represented exactly.

Conventions
-----------
* Faces are consistently oriented before extraction (see
  :func:`orient_faces_consistently`).
* For a face ``(p, i, n)`` the successor mapping at vertex ``i`` maps the
  incoming neighbor ``p`` to the outgoing neighbor ``n``:
  ``pi_i(p) = n``.
* Per vertex ``i`` with degree ``D_i``, targets are stored as an index
  array ``succ[i]`` of length ``D_i + 1`` over ``neighbors[i]`` plus one
  extra slot for ``NULL`` at index ``D_i``. ``succ[i][p] = D_i`` means
  ``pi_i(p) = NULL``; the ``NULL`` element itself maps to the first
  neighbor of an open fan (boundary) or to ``NULL`` (interior vertex).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def orient_faces_consistently(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Propagate a consistent winding across adjacent faces.

    Args:
        vertices: ``(V, 3)`` vertex positions.
        faces: ``(F, 3)`` triangle indices with arbitrary winding.

    Returns:
        ``(F, 3)`` faces with per-component consistent orientation.
    """
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    trimesh.repair.fix_normals(mesh)
    return np.asarray(mesh.faces, dtype=np.int64)


def extract_topology(
    faces: np.ndarray, num_vertices: int
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Extract edges and per-vertex successor targets from oriented faces.

    Args:
        faces: ``(F, 3)`` consistently oriented triangle indices.
        num_vertices: Number of vertices ``V``.

    Returns:
        A tuple ``(edges, neighbors, successors)`` where ``edges`` is an
        ``(E, 2)`` int64 array of unique undirected edges with ``i < j``,
        ``neighbors[i]`` is the sorted int64 neighbor index array of vertex
        ``i``, and ``successors[i]`` is the int64 target array aligned with
        ``neighbors[i]`` plus the ``NULL`` slot at index ``D_i``.
    """
    faces = np.asarray(faces, dtype=np.int64)
    succ_map: dict[int, dict[int, int]] = defaultdict(dict)
    for a, b, c in faces:
        # At vertex b: incoming neighbor a -> outgoing neighbor c, etc.
        succ_map[b][a] = c
        succ_map[c][b] = a
        succ_map[a][c] = b

    edge_set: set[tuple[int, int]] = set()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            edge_set.add((min(u, v), max(u, v)))
    edges = np.asarray(sorted(edge_set), dtype=np.int64).reshape(-1, 2)

    nbr_of: list[set[int]] = [set() for _ in range(num_vertices)]
    for u, v in edges:
        nbr_of[int(u)].add(int(v))
        nbr_of[int(v)].add(int(u))

    neighbors: list[np.ndarray] = []
    successors: list[np.ndarray] = []
    for i in range(num_vertices):
        nbrs = sorted(nbr_of[i])
        nbr_index = {p: k for k, p in enumerate(nbrs)}
        d = len(nbrs)
        succ = np.full(d + 1, d, dtype=np.int64)  # default: NULL -> NULL
        hit = np.zeros(d, dtype=bool)
        for p, n in succ_map.get(i, {}).items():
            succ[nbr_index[p]] = nbr_index[n]
            hit[nbr_index[n]] = True
        if d > 0 and not hit.all():
            # Boundary vertex: NULL maps to the first neighbor of the open
            # fan, i.e. the neighbor nobody maps to.
            succ[d] = int(np.flatnonzero(~hit)[0])
        neighbors.append(np.asarray(nbrs, dtype=np.int64))
        successors.append(succ)
    return edges, neighbors, successors


def faces_from_topology(
    neighbors: list[np.ndarray], successors: list[np.ndarray]
) -> np.ndarray:
    """Reconstruct oriented triangles from successor mappings.

    Every directed halfedge ``(p, i)`` with ``pi_i(p) = n != NULL`` yields
    the triangle ``(p, i, n)``; each directed halfedge is consumed at most
    once so the output is consistently oriented.

    Args:
        neighbors: Per-vertex sorted neighbor arrays.
        successors: Per-vertex successor targets aligned with ``neighbors``.

    Returns:
        ``(F, 3)`` int64 triangle indices.
    """
    faces: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for i, (nbrs, succ) in enumerate(zip(neighbors, successors, strict=True)):
        d = len(nbrs)
        for k, p in enumerate(nbrs):
            t = int(succ[k])
            if t >= d:
                continue  # pi_i(p) = NULL
            n = int(nbrs[t])
            tri = (int(p), int(i), n)
            half = ((p, i), (i, n), (n, p))
            if any(h in seen for h in half):
                continue
            seen.update(half)
            faces.append(tri)
    if not faces:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(faces, dtype=np.int64)


def boundary_vertices(
    neighbors: list[np.ndarray], successors: list[np.ndarray]
) -> np.ndarray:
    """Return a boolean mask of vertices whose fan is open (boundaries)."""
    out = []
    for nbrs, succ in zip(neighbors, successors, strict=True):
        d = len(nbrs)
        out.append(bool(d > 0 and np.any(succ[:d] >= d)))
    return np.asarray(out, dtype=bool)
