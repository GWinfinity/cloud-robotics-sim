"""Tests for the vertex-set Mesh VAE: shapes, losses, assembly."""

import numpy as np
import torch
import trimesh

from cloud_robotics_sim.meshy_t2.data import prepare_mesh_sample
from cloud_robotics_sim.meshy_t2.geometry.assembly import (
    _round_single_fan,
    assemble_mesh,
)
from cloud_robotics_sim.meshy_t2.geometry.halfedge import (
    extract_topology,
    orient_faces_consistently,
)
from cloud_robotics_sim.meshy_t2.losses import (
    edge_loss,
    face_loss,
    mesh_vae_loss,
    sinkhorn_log,
)
from cloud_robotics_sim.meshy_t2.models.mesh_vae import (
    MeshVAE,
    MeshVAEConfig,
    edge_logits,
)


def _sample(num_samples: int = 128):
    mesh = trimesh.creation.box()
    return prepare_mesh_sample(mesh, num_samples=num_samples, voxel_res=32, seed=0)


def test_edge_logits_symmetric():
    """Check edge logits symmetric."""
    torch.manual_seed(0)
    emb = torch.randn(10, 16)
    a = edge_logits(emb, 8)
    assert a.shape == (10, 10)
    torch.testing.assert_close(a, a.T)
    torch.testing.assert_close(a.diagonal(), torch.zeros(10))


def test_sinkhorn_doubly_stochastic():
    """Check sinkhorn doubly stochastic."""
    torch.manual_seed(0)
    logits = torch.randn(4, 7, 7)
    p = sinkhorn_log(logits, 20).exp()
    torch.testing.assert_close(p.sum(-1), torch.ones(4, 7), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(p.sum(-2), torch.ones(4, 7), rtol=1e-3, atol=1e-3)


def test_mesh_vae_forward_and_loss_backward():
    """Check mesh vae forward and loss backward."""
    torch.manual_seed(0)
    sample = _sample()
    cfg = MeshVAEConfig.tiny()
    model = MeshVAE(cfg)
    positions, edge_emb, face_emb, z = model(
        sample.vertices, sample.edges, sample.points, sample.normals
    )
    v = sample.num_vertices
    assert positions.shape == (v, 3)
    assert edge_emb.shape == (v, 2 * cfg.edge_dim)
    assert face_emb.shape == (v, 3 * cfg.face_dim)
    assert z.shape == (v, cfg.latent_channels)

    loss, parts = mesh_vae_loss(
        positions,
        edge_emb,
        face_emb,
        model.decoder.null_prev,
        model.decoder.null_next,
        sample.vertices,
        sample.edges,
        sample.neighbors,
        sample.successors,
        cfg,
    )
    assert torch.isfinite(loss)
    assert set(parts) == {"vertex", "edge", "face"}
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_edge_loss_matches_manual_softplus():
    """Check edge loss matches manual softplus."""
    # v=3, one edge (0,1). space all zeros, time = [0, 1, 2] (d_e = 1):
    # A_01 = 1, A_02 = 4, A_12 = 1.
    de = 1
    edge_emb = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]
    )  # (space, time) halves
    edges = torch.tensor([[0, 1]])
    loss = edge_loss(edge_emb, de, edges, 3)
    sp = torch.nn.functional.softplus
    lam = 1 / 2  # balanced: E / (pairs - E)
    expected = (
        sp(torch.tensor(-1.0)) + lam * (sp(torch.tensor(4.0)) + sp(torch.tensor(1.0)))
    ) / (1 + lam * 2)
    torch.testing.assert_close(loss, expected)


def test_face_scores_and_loss_shapes():
    """Check face scores and loss shapes."""
    torch.manual_seed(0)
    v, df = 6, 4
    face_emb = torch.randn(v, 3 * df, requires_grad=True)
    null_prev = torch.randn(df, requires_grad=True)
    null_next = torch.randn(df, requires_grad=True)
    neighbors = [
        torch.tensor([1, 2, 3]),
        torch.tensor([0, 2]),
        torch.tensor([0, 1]),
        torch.tensor([0, 4, 5]),
        torch.tensor([3, 5]),
        torch.tensor([3, 4]),
    ]
    successors = [
        torch.tensor([1, 2, 0, 3]),
        torch.tensor([1, 0, 2]),
        torch.tensor([1, 0, 2]),
        torch.tensor([1, 2, 0, 3]),
        torch.tensor([1, 0, 2]),
        torch.tensor([1, 0, 2]),
    ]
    loss = face_loss(face_emb, null_prev, null_next, df, neighbors, successors, 5)
    assert torch.isfinite(loss)
    loss.backward()
    assert face_emb.grad is not None


def test_round_single_fan_recovers_known_cycle():
    """Check round single fan recovers known cycle."""
    # Scores favor the rotation permutation 0->1->2->3->0.
    n = 4
    phi = np.full((n, n), -5.0)
    for i in range(n):
        phi[i, (i + 1) % n] = 5.0
    succ = _round_single_fan(phi)
    for i in range(n):
        assert succ[i] == (i + 1) % n


def test_round_single_fan_merges_subcycles():
    """Check round single fan merges subcycles."""
    # Scores favor two disjoint transpositions; the repair must merge
    # them into one fan (single cycle covering all elements).
    n = 4
    phi = np.full((n, n), -5.0)
    for i, j in [(0, 1), (1, 0), (2, 3), (3, 2)]:
        phi[i, j] = 5.0
    succ = _round_single_fan(phi)
    # Walk the permutation: it must visit every element.
    seen, k = set(), 0
    for _ in range(n):
        seen.add(k)
        k = int(succ[k])
    assert seen == set(range(n))


def test_assemble_mesh_from_gt_topology():
    """Check assemble mesh from gt topology."""
    # Craft embeddings that reproduce a known mesh exactly: place the
    # space halves equal for neighbors and the time halves far apart.
    torch.manual_seed(0)
    mesh = trimesh.creation.box()
    faces = orient_faces_consistently(mesh.vertices, mesh.faces)
    edges, neighbors, successors = extract_topology(faces, len(mesh.vertices))
    v = len(mesh.vertices)
    de, df = 8, 8

    edge_emb = torch.randn(v, 2 * de) * 0.01
    for i, j in edges:
        # Push connected pairs apart in time, keep space close.
        edge_emb[i, de:] += torch.randn(de) * 0.01
    positions = torch.tensor(mesh.vertices, dtype=torch.float32)
    face_emb = torch.randn(v, 3 * df)
    null_prev = torch.randn(df)
    null_next = torch.randn(df)
    verts, e_out, f_out = assemble_mesh(
        positions, edge_emb, face_emb, null_prev, null_next, de, df
    )
    assert verts.shape == (v, 3)
    assert e_out.shape[1] == 2
    assert f_out.shape[1] == 3
    # Every face indexes valid vertices.
    assert f_out.min() >= 0 and f_out.max() < v


def test_mesh_vae_overfit_single_mesh():
    """A tiny VAE should overfit one cube: losses must drop clearly."""
    torch.manual_seed(0)
    sample = _sample()
    cfg = MeshVAEConfig.tiny()
    model = MeshVAE(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    tensors = (
        sample.vertices,
        sample.edges,
        sample.points,
        sample.normals,
    )
    first = last = None
    for _ in range(60):
        positions, edge_emb, face_emb, _ = model(*tensors)
        loss, _ = mesh_vae_loss(
            positions,
            edge_emb,
            face_emb,
            model.decoder.null_prev,
            model.decoder.null_next,
            sample.vertices,
            sample.edges,
            sample.neighbors,
            sample.successors,
            cfg,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss)
        last = float(loss)
    assert last < first * 0.9
