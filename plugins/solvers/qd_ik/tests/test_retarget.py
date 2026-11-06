"""End-to-end retargeting on a synthetic floating-base arm clip."""

import numpy as np

from plugins.solvers.qd_ik import (
    DiffIKSolver,
    FloorConstraint,
    FrameTask,
    JointLimits,
    Retargeter,
    SelfCollisionConstraint,
    SphereModel,
)
from plugins.solvers.qd_ik import kinematics as kin


def _setup(model, constraints):
    task = FrameTask(
        model,
        ["wrist"],
        gain_pos=30.0,
        gain_rot=10.0,
        weight_pos=100.0,
        weight_rot=10.0,
    )
    solver = DiffIKSolver(model, [task], constraints, regularization=1.0)
    return solver


def test_retarget_clip_tracks(toy_model):
    """Batched clip retargeting tracks a moving wrist target."""
    model = toy_model
    spheres = SphereModel(
        model,
        links=[0, "shoulder", "elbow", "wrist"],
        offsets=np.array([[0, 0, 0.02], [0.1, 0, 0], [0.1, 0, 0], [0.05, 0, 0]]),
        radii=np.array([0.06, 0.05, 0.05, 0.04]),
    )
    solver = _setup(
        model, [JointLimits(model), FloorConstraint(model, spheres, gain=30.0)]
    )
    B, T, dt = 2, 60, 1.0 / 30
    t = np.arange(T)[:, None]
    tgt = np.zeros((T, B, 1, 3))
    tgt[:, :, 0, 0] = 0.3 + 0.1 * np.cos(2 * np.pi * t / 30)
    tgt[:, :, 0, 1] = 0.1 * np.sin(2 * np.pi * t / 30)
    tgt[:, :, 0, 2] = 0.3 + 0.05 * np.sin(2 * np.pi * t / 15)

    def preprocess(frames):
        B_ = frames.shape[0]
        return {
            "frame_pos": frames,
            "frame_rmat": np.broadcast_to(np.eye(3), (B_, 1, 3, 3)),
        }

    def initial_q(frames):
        q0 = model.default_configuration(frames.shape[0])
        q0[:, 2] = 0.3
        return q0

    rt = Retargeter(solver, preprocess, initial_q)
    qs = rt.retarget_clip(tgt, dt)
    assert qs.shape == (T, B, model.nq)

    kd = kin.Data(model, B, n_tracked=1)
    wrist = model.joint_index("wrist")
    worst = 0.0
    for i in range(T):
        Ro, po = kin.forward_kinematics(model, kd, qs[i])
        worst = max(
            worst,
            np.linalg.norm(po[:, wrist, :] - tgt[i, :, 0, :], axis=1).max(),
        )
    assert worst < 0.02


def test_retarget_clips_batches_and_pads(toy_model):
    """Variable-length clips are padded, batched, and unbatched correctly."""
    model = toy_model
    solver = _setup(model, [JointLimits(model)])
    dt = 1.0 / 30

    def preprocess(frames):
        B_ = frames.shape[0]
        return {
            "frame_pos": frames,
            "frame_rmat": np.broadcast_to(np.eye(3), (B_, 1, 3, 3)),
        }

    def initial_q(frames):
        q0 = model.default_configuration(frames.shape[0])
        q0[:, 2] = 0.3
        return q0

    rng = np.random.default_rng(4)
    clips = []
    for n in (20, 35):
        base = np.array([[0.3, 0.0, 0.3]])
        walk = np.cumsum(rng.normal(scale=0.01, size=(n, 1, 3)), axis=0)
        clips.append(base + walk)
    rt = Retargeter(solver, preprocess, initial_q)
    out = rt.retarget_clips(clips, dt)
    assert len(out) == 2
    assert out[0].shape == (20, model.nq)
    assert out[1].shape == (35, model.nq)
    for i, clip in enumerate(clips):
        kd = kin.Data(model, 1, n_tracked=1)
        wrist = model.joint_index("wrist")
        errs = []
        for t in range(len(clip)):
            Ro, po = kin.forward_kinematics(model, kd, out[i][t][None])
            errs.append(np.linalg.norm(po[0, wrist] - clip[t, 0]))
        assert max(errs) < 0.02


def test_floor_constraint_holds(toy_model):
    """Push the wrist target near the floor: spheres must not penetrate."""
    model = toy_model
    spheres = SphereModel(
        model,
        links=[0, "shoulder", "elbow", "wrist"],
        offsets=np.array([[0, 0, 0.02], [0.1, 0, 0], [0.1, 0, 0], [0.05, 0, 0]]),
        radii=np.array([0.06, 0.05, 0.05, 0.04]),
    )
    solver = _setup(
        model, [JointLimits(model), FloorConstraint(model, spheres, gain=30.0)]
    )
    B, T, dt = 1, 40, 1.0 / 30
    tgt = np.zeros((T, B, 1, 3))
    tgt[:, :, 0, 0] = 0.3
    tgt[:, :, 0, 2] = np.linspace(0.1, 0.02, T)[:, None]  # sinking target

    def preprocess(frames):
        return {
            "frame_pos": frames,
            "frame_rmat": np.broadcast_to(np.eye(3), (B, 1, 3, 3)),
        }

    def initial_q(frames):
        q0 = model.default_configuration(B)
        q0[:, 2] = 0.3
        return q0

    rt = Retargeter(solver, preprocess, initial_q)
    qs = rt.retarget_clip(tgt, dt)
    kd = kin.Data(model, B, n_tracked=1)
    for i in range(T):
        Ro, po = kin.forward_kinematics(model, kd, qs[i])
        sp_links = [0, 1, 2, 3]
        for s, lk in enumerate(sp_links):
            pw = po[0, lk] + Ro[0, lk] @ spheres.offsets[s]
            assert pw[2] >= spheres.radii[s] - 1e-3, (i, s, pw[2])


def test_retarget_with_self_collision_topk(toy_model):
    """Clip retargeting runs clean with top-k self-collision rows."""
    model = toy_model
    spheres = SphereModel(
        model,
        links=[0, "shoulder", "elbow", "wrist"],
        offsets=np.array([[0, 0, 0.02], [0.1, 0, 0], [0.1, 0, 0], [0.05, 0, 0]]),
        radii=np.array([0.06, 0.05, 0.05, 0.04]),
    )
    sc = SelfCollisionConstraint(
        model,
        spheres,
        pairs=np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]),
        gain=30.0,
        max_rows=2,
    )
    solver = _setup(model, [JointLimits(model), sc])
    B, T, dt = 1, 30, 1.0 / 30
    t = np.arange(T)[:, None]
    tgt = np.zeros((T, B, 1, 3))
    tgt[:, :, 0, 0] = 0.3 + 0.08 * np.cos(2 * np.pi * t / 30)
    tgt[:, :, 0, 2] = 0.3 + 0.04 * np.sin(2 * np.pi * t / 15)

    def preprocess(frames):
        return {
            "frame_pos": frames,
            "frame_rmat": np.broadcast_to(np.eye(3), (B, 1, 3, 3)),
        }

    def initial_q(frames):
        q0 = model.default_configuration(B)
        q0[:, 2] = 0.3
        return q0

    rt = Retargeter(solver, preprocess, initial_q)
    qs = rt.retarget_clip(tgt, dt)
    assert np.all(np.isfinite(qs))
    assert solver.m_total == JointLimits(model).num_rows + 2
