"""Mesh preprocessing and datasets for the Meshy T2 reproduction.

``prepare_mesh_sample`` runs the full preprocessing chain of Sec. 2.1:
non-manifold repair (fan splitting), orientation propagation,
normalization to ``[0, 1]^3``, halfedge-successor extraction, surface
sampling (preferential along edges) and occupancy voxelization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset

from .geometry.halfedge import extract_topology, orient_faces_consistently
from .geometry.mesh_utils import (
    load_mesh,
    normalize_vertices,
    repair_nonmanifold,
    sample_surface,
    voxelize,
)

MESH_SUFFIXES = {".obj", ".glb", ".gltf", ".off", ".stl", ".ply"}


@dataclass
class MeshSample:
    """Preprocessed tensors of a single training mesh."""

    vertices: torch.Tensor  # (V, 3) normalized to [0, 1]^3
    faces: torch.Tensor  # (F, 3) oriented
    edges: torch.Tensor  # (E, 2), i < j
    neighbors: list[torch.Tensor]  # per-vertex (D_i,)
    successors: list[torch.Tensor]  # per-vertex (D_i + 1,)
    points: torch.Tensor  # (P, 3) surface samples
    normals: torch.Tensor  # (P, 3)
    occupancy: torch.Tensor  # (res, res, res) bool
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale: float = 1.0

    @property
    def num_vertices(self) -> int:
        """Number of vertices ``V``."""
        return self.vertices.shape[0]

    def unnormalize(self, vertices: np.ndarray) -> np.ndarray:
        """Map normalized vertices back to the original coordinate frame."""
        return vertices * self.scale + self.origin


def prepare_mesh_sample(
    mesh: trimesh.Trimesh,
    num_samples: int = 4096,
    voxel_res: int = 64,
    edge_ratio: float = 0.5,
    seed: int = 0,
) -> MeshSample:
    """Run the full preprocessing chain on a raw mesh.

    Args:
        mesh: Input triangle mesh.
        num_samples: Number of surface samples for the point context.
        voxel_res: Occupancy scaffold resolution.
        edge_ratio: Fraction of samples drawn along edges.
        seed: Sampling RNG seed.

    Returns:
        The prepared :class:`MeshSample`.
    """
    mesh = repair_nonmanifold(mesh)
    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=False
    )
    faces = orient_faces_consistently(mesh.vertices, mesh.faces)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertices, origin, scale = normalize_vertices(vertices)
    norm_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    edges, neighbors, successors = extract_topology(faces, len(vertices))
    points, normals = sample_surface(norm_mesh, num_samples, edge_ratio, seed)
    occupancy = voxelize(norm_mesh, res=voxel_res)

    return MeshSample(
        vertices=torch.tensor(vertices, dtype=torch.float32),
        faces=torch.tensor(faces, dtype=torch.long),
        edges=torch.tensor(edges, dtype=torch.long),
        neighbors=[torch.tensor(n, dtype=torch.long) for n in neighbors],
        successors=[torch.tensor(s, dtype=torch.long) for s in successors],
        points=torch.tensor(points, dtype=torch.float32),
        normals=torch.tensor(normals, dtype=torch.float32),
        occupancy=torch.tensor(occupancy, dtype=torch.bool),
        origin=origin,
        scale=scale,
    )


class MeshFolderDataset(Dataset):
    """Dataset over a folder of mesh files with on-the-fly preprocessing.

    Args:
        root: Directory scanned recursively for mesh files.
        num_samples: Surface samples per mesh.
        voxel_res: Occupancy resolution.
        max_vertices: Skip meshes above this vertex count.
    """

    def __init__(
        self,
        root: str | Path,
        num_samples: int = 4096,
        voxel_res: int = 64,
        max_vertices: int = 8192,
    ) -> None:
        self.paths = sorted(
            p for p in Path(root).rglob("*") if p.suffix.lower() in MESH_SUFFIXES
        )
        if not self.paths:
            raise FileNotFoundError(f"no mesh files under {root}")
        self.num_samples = num_samples
        self.voxel_res = voxel_res
        self.max_vertices = max_vertices

    def __len__(self) -> int:
        """Number of mesh files."""
        return len(self.paths)

    def __getitem__(self, idx: int) -> MeshSample:
        """Load and preprocess the mesh at ``idx``."""
        mesh = load_mesh(self.paths[idx])
        sample = prepare_mesh_sample(mesh, self.num_samples, self.voxel_res, seed=idx)
        if sample.num_vertices > self.max_vertices:
            return self[(idx + 1) % len(self)]
        return sample


class SyntheticMeshDataset(Dataset):
    """Procedural primitive dataset for smoke training without assets.

    Each item is a random trimesh primitive (box, icosphere, cylinder,
    cone, torus or capsule) with randomized subdivisions and a random
    rotation/scale augmentation as used in Mesh VAE training (Sec. 2.1).
    """

    def __init__(
        self,
        size: int = 128,
        num_samples: int = 4096,
        voxel_res: int = 64,
        seed: int = 0,
    ) -> None:
        self.size = size
        self.num_samples = num_samples
        self.voxel_res = voxel_res
        self.seed = seed

    def __len__(self) -> int:
        """Number of procedural samples per epoch."""
        return self.size

    def _primitive(self, rng: np.random.Generator) -> trimesh.Trimesh:
        kind = rng.integers(0, 5)
        if kind == 0:
            extents = rng.uniform(0.3, 1.0, size=3)
            return trimesh.creation.box(extents=extents)
        if kind == 1:
            return trimesh.creation.icosphere(
                subdivisions=int(rng.integers(1, 3)),
                radius=float(rng.uniform(0.3, 0.6)),
            )
        if kind == 2:
            return trimesh.creation.cylinder(
                radius=float(rng.uniform(0.2, 0.4)),
                height=float(rng.uniform(0.4, 1.0)),
                sections=int(rng.integers(8, 24)),
            )
        if kind == 3:
            return trimesh.creation.cone(
                radius=float(rng.uniform(0.2, 0.5)),
                height=float(rng.uniform(0.4, 1.0)),
                sections=int(rng.integers(8, 24)),
            )
        return trimesh.creation.capsule(
            radius=float(rng.uniform(0.2, 0.4)), height=float(rng.uniform(0.3, 0.8))
        )

    def __getitem__(self, idx: int) -> MeshSample:
        """Generate and preprocess the primitive at ``idx``."""
        rng = np.random.default_rng(self.seed + idx)
        mesh = self._primitive(rng)
        # Random rotation + anisotropic scale augmentation.
        angle = rng.uniform(0, 2 * np.pi, size=3)
        rot = trimesh.transformations.euler_matrix(*angle)
        scale = np.eye(4) * np.concatenate([rng.uniform(0.7, 1.3, size=3), [1.0]])
        mesh.apply_transform(rot @ scale)
        return prepare_mesh_sample(
            mesh, self.num_samples, self.voxel_res, seed=self.seed + idx
        )
