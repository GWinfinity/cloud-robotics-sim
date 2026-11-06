"""Solver-level tests: QP-data build against numpy references, and one-step
diff-IK tracking behavior.
"""

import numpy as np

from plugins.solvers.qd_ik import (
    DiffIKSolver,
    FloorConstraint,
    FrameTask,
    JointLimits,
    SelfCollisionConstraint,
    SphereModel,
    integrate,
)
from plugins.solvers.qd_ik import kinematics as kin


def _rmat_error(Rc, Rd):
    """orientation_error_quat (frax), numpy twin."""
    R = Rc @ Rd.T
    tr = np.trace(R)
    cands = [
        (1 + tr, R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]),
        (R[2, 1] - R[1, 2], 1 + 2 * R[0, 0] - tr, R[0, 1] + R[1, 0], R[0, 2] + R[2, 0]),
        (R[0, 2] - R[2, 0], R[0, 1] + R[1, 0], 1 + 2 * R[1, 1] - tr, R[1, 2] + R[2, 1]),
        (R[1, 0] - R[0, 1], R[0, 2] + R[2, 0], R[1, 2] + R[2, 1], 1 + 2 * R[2, 2] - tr),
    ]
    q = np.array(max(cands, key=lambda c: c[0]))
    q = q / np.linalg.norm(q)
    s = 2.0 if q[0] >= 0 else -2.0
    return s * q[1:]


def test_qp_data_matches_numpy(toy_model, toy_config):
    """Fused build kernel matches numpy reference assembly entry by entry."""
    model, q = toy_model, toy_config
    B = q.shape[0]
    spheres = SphereModel(
        model,
        links=[0, "shoulder", "elbow", "wrist"],
        offsets=np.array([[0, 0, 0.02], [0.1, 0, 0], [0.1, 0, 0], [0.05, 0, 0]]),
        radii=np.array([0.06, 0.05, 0.05, 0.04]),
    )
    task = FrameTask(
        model,
        ["wrist"],
        gain_pos=30.0,
        gain_rot=10.0,
        weight_pos=100.0,
        weight_rot=10.0,
    )
    limits = JointLimits(model)
    floor = FloorConstraint(model, spheres, gain=30.0)
    sc = SelfCollisionConstraint(
        model, spheres, pairs=np.array([[1, 2], [2, 3], [1, 3]]), gain=30.0
    )
    solver = DiffIKSolver(
        model, [task], [limits, floor, sc], regularization=10.0, damping=10.0
    )
    tgt_pos = np.array([[[0.3, 0.0, 0.3]], [[0.25, 0.05, 0.28]]])
    ref = {
        "frame_pos": tgt_pos,
        "frame_rmat": np.broadcast_to(np.eye(3), (B, 1, 3, 3)),
    }
    data = solver.data(B)
    P, c, G, h, cost = solver.qp_data(q, ref, data)

    # ---- numpy reference from the independently validated kinematics kernels
    kd = kin.Data(model, B, n_tracked=1)
    Ro, po = kin.forward_kinematics(model, kd, q)
    Jv, Jw = kin.frame_jacobians(model, kd, q, [model.joint_index("wrist")])
    wrist = model.joint_index("wrist")
    w = np.array([100.0] * 3 + [10.0] * 3)
    e_pos = po[:, wrist, :] - tgt_pos[:, 0, :]
    e_rot = np.stack([_rmat_error(Ro[b, wrist], np.eye(3)) for b in range(B)])
    v_des = np.concatenate([-30.0 * e_pos, -10.0 * e_rot], axis=1)
    Jrows = np.concatenate([Jv[:, 0], Jw[:, 0]], axis=1)
    WJm = w[None, :, None] * Jrows
    P_ref = np.einsum("bkr,bkc->brc", WJm, WJm)
    task_err = ((w * v_des) ** 2).sum(axis=1)
    P_ref += (10.0 * task_err)[:, None, None] * np.eye(model.nv)[None]
    P_ref += np.diag(solver.regularization)[None]
    c_ref = -np.einsum("br,brc->bc", w * v_des, WJm)
    assert np.allclose(P, P_ref, rtol=1e-9, atol=1e-6)
    assert np.allclose(c, c_ref, rtol=1e-9, atol=1e-4)

    # limit rows
    m_lim = limits.num_rows
    for b in range(B):
        row = 0
        for k in range(model.n_act):
            qi = 7 + k
            assert np.isclose(
                h[b, row],
                min(model.v_max[k], limits.gain * (model.q_max[k] - q[b, qi])),
            )
            assert G[b, row, 6 + k] == 1.0
            row += 1
        for k in range(model.n_act):
            qi = 7 + k
            assert np.isclose(
                h[b, row],
                -max(-model.v_max[k], limits.gain * (model.q_min[k] - q[b, qi])),
            )
            assert G[b, row, 6 + k] == -1.0
            row += 1

    # sphere positions and floor rows
    sp_links = [0, 1, 2, 3]
    sp_ref = np.stack(
        [
            po[:, lk] + np.einsum("bij,j->bi", Ro[:, lk], spheres.offsets[s])
            for s, lk in enumerate(sp_links)
        ],
        axis=1,
    )
    assert np.allclose(data.sp, sp_ref, atol=1e-12)
    for s in range(4):
        hz = sp_ref[:, s, 2] - spheres.radii[s]
        assert np.allclose(h[:, m_lim + s], floor.gain * hz, atol=1e-9)

    # self-collision rows: h matches distances
    for k, (a, b_) in enumerate(sc.pairs):
        d = np.linalg.norm(sp_ref[:, a] - sp_ref[:, b_], axis=1)
        hpair = d - (spheres.radii[a] + spheres.radii[b_] + sc.tols[k] + sc.margin)
        assert np.allclose(h[:, m_lim + floor.num_rows + k], sc.gain * hpair, atol=1e-9)

    # cost = task error + elastic penalties (all rows feasible here)
    assert np.allclose(cost, task_err, rtol=1e-6, atol=1e-6)


def test_one_step_tracks_and_respects_limits(toy_model, toy_config):
    """Repeated diff-IK steps converge to a target under joint limits."""
    model = toy_model
    B = 2
    task = FrameTask(
        model,
        ["wrist"],
        gain_pos=30.0,
        gain_rot=10.0,
        weight_pos=100.0,
        weight_rot=10.0,
    )
    limits = JointLimits(model, q_max=np.minimum(model.q_max, 0.5))
    solver = DiffIKSolver(model, [task], [limits], regularization=1.0)
    q = model.default_configuration(B)
    q[:, 2] = 0.3
    kd = kin.Data(model, B, n_tracked=1)
    wrist = model.joint_index("wrist")
    data = solver.data(B)
    tgt = np.array([[[0.3, 0.0, 0.3]], [[0.25, 0.05, 0.28]]])
    for _ in range(60):
        Ro, po = kin.forward_kinematics(model, kd, q)
        err = np.linalg.norm(po[:, wrist, :] - tgt[:, 0, :], axis=1).max()
        ref = {
            "frame_pos": tgt,
            "frame_rmat": np.broadcast_to(np.eye(3), (B, 1, 3, 3)),
        }
        q = solver.step(q, ref, 1.0 / 30, data)
        assert np.all(q[:, 7:] <= 0.5 + 1e-9)
    assert err < 0.02


def test_integrate_quaternion_base(toy_model):
    """Quaternion base integration rotates by the expected angle."""
    model = toy_model
    q = model.default_configuration(2)
    v = np.zeros((2, model.nv))
    v[:, 3] = np.pi  # body-frame angular velocity about x
    q2 = integrate(model, q, v, 0.5)
    # half a second at pi rad/s about x: quaternion should be ~ 90 deg rotation
    dq = q2[:, 3:7]
    assert np.allclose(np.linalg.norm(dq, axis=1), 1.0, atol=1e-12)
    assert np.allclose(np.abs(dq[:, 0]), np.cos(np.pi / 4), atol=1e-9)
    assert np.allclose(q2[:, 7:], q[:, 7:], atol=1e-12)


def test_self_collision_topk_selects_closest(toy_model, toy_config):
    """top-k self-collision keeps the k closest pairs, in ascending-h order."""
    model, q = toy_model, toy_config
    B = q.shape[0]
    spheres = SphereModel(
        model,
        links=[0, "shoulder", "elbow", "wrist"],
        offsets=np.array([[0, 0, 0.02], [0.1, 0, 0], [0.1, 0, 0], [0.05, 0, 0]]),
        radii=np.array([0.06, 0.05, 0.05, 0.04]),
    )
    pairs = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    task = FrameTask(
        model,
        ["wrist"],
        gain_pos=30.0,
        gain_rot=10.0,
        weight_pos=100.0,
        weight_rot=10.0,
    )
    sc_full = SelfCollisionConstraint(model, spheres, pairs=pairs, gain=30.0)
    sc_topk = SelfCollisionConstraint(
        model, spheres, pairs=pairs, gain=30.0, max_rows=2
    )
    assert sc_topk.num_rows == 2
    solver_full = DiffIKSolver(model, [task], [sc_full])
    solver_topk = DiffIKSolver(model, [task], [sc_topk])
    assert solver_topk.m_total == 2

    tgt_pos = np.array([[[0.3, 0.0, 0.3]], [[0.25, 0.05, 0.28]]])
    ref = {
        "frame_pos": tgt_pos,
        "frame_rmat": np.broadcast_to(np.eye(3), (B, 1, 3, 3)),
    }
    data_f = solver_full.data(B)
    _, _, G_f, h_f, _ = solver_full.qp_data(q, ref, data_f)
    data_t = solver_topk.data(B)
    _, _, G_t, h_t, _ = solver_topk.qp_data(q, ref, data_t)

    # reference pair gaps from the validated FK
    kd = kin.Data(model, B, n_tracked=1)
    Ro, po = kin.forward_kinematics(model, kd, q)
    sp_ref = np.stack(
        [
            po[:, lk] + np.einsum("bij,j->bi", Ro[:, lk], spheres.offsets[s])
            for s, lk in enumerate([0, 1, 2, 3])
        ],
        axis=1,
    )
    hp = np.linalg.norm(sp_ref[:, pairs[:, 0]] - sp_ref[:, pairs[:, 1]], axis=2) - (
        spheres.radii[pairs[:, 0]] + spheres.radii[pairs[:, 1]]
    )

    for b in range(B):
        order = np.argsort(hp[b], kind="stable")
        for j in range(2):
            want = order[j]
            assert np.isclose(h_t[b, j], 30.0 * hp[b, want], atol=1e-9)
            assert np.allclose(G_t[b, j], G_f[b, want], atol=1e-9)
