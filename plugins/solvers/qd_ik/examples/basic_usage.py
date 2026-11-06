"""qd_ik basic usage: retarget a synthetic motion clip onto a floating-base
arm, batched over two clips.

Run from the repo root:

    PYTHONPATH=. .venv/Scripts/python.exe plugins/solvers/qd_ik/examples/basic_usage.py

No downloads, no external assets: the "mocap" is a generated wrist trajectory
and the robot is built from raw arrays.  For a G1/LAFAN1-scale example see
the porting notes in the plugin README -- the pipeline is identical, only the
model source (Model.from_urdf) and the preprocess scaling differ.
"""

import time

import numpy as np

from plugins.solvers.qd_ik import (
    DiffIKSolver,
    FloorConstraint,
    FrameTask,
    JointLimits,
    Retargeter,
    SphereModel,
)
from plugins.solvers.qd_ik import kinematics as kin
from plugins.solvers.qd_ik.model import Model


def build_model():
    """Floating-base 3R arm: a miniature humanoid arm chain."""
    nj = 4
    return Model(
        parents=np.array([0, 0, 1, 2]),
        jtype=np.array([2, 0, 0, 0]),  # universe + 3 revolute joints
        axis=np.array([[1.0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0]]),
        Rpl=np.tile(np.eye(3), (nj, 1, 1)),
        ppl=np.array([[0.0, 0, 0], [0, 0, 0.05], [0.2, 0, 0], [0.2, 0, 0]]),
        q_min=np.array([-2.5, -2.5, -2.5]),
        q_max=np.array([2.5, 2.5, 2.5]),
        v_max=np.array([10.0, 10.0, 10.0]),
        names=["universe", "shoulder", "elbow", "wrist"],
        floating_base=True,
    )


def make_clip(rng, n_frames, phase):
    """A wrist target circling in the air: (T, n, 3) positions."""
    t = np.arange(n_frames) + phase
    tgt = np.zeros((n_frames, 1, 3))
    tgt[:, 0, 0] = 0.3 + 0.1 * np.cos(2 * np.pi * t / 30)
    tgt[:, 0, 1] = 0.1 * np.sin(2 * np.pi * t / 30)
    tgt[:, 0, 2] = 0.3 + 0.05 * np.sin(2 * np.pi * t / 15)
    return tgt


def preprocess(frames):
    """(B, n, 3) target positions (+ identity orientations) -> ref dict."""
    B, n = frames.shape[0], frames.shape[1]
    return {
        "frame_pos": frames,
        "frame_rmat": np.broadcast_to(np.eye(3), (B, n, 3, 3)),
    }


def initial_q(frames):
    """Default configuration with the base at a working height."""
    q0 = build_model().default_configuration(frames.shape[0])
    q0[:, 2] = 0.3
    return q0


def main():
    """Build the model, retarget two synthetic clips, report errors."""
    model = build_model()
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
    solver = DiffIKSolver(
        model,
        [task],
        [JointLimits(model), FloorConstraint(model, spheres, gain=30.0)],
        regularization=1.0,
    )
    print(f"{model}; QP rows m={solver.m_total}, auto rho={solver.qp_config.rho:g}")

    rng = np.random.default_rng(0)
    clips = [make_clip(rng, 60, 0.0), make_clip(rng, 45, 7.5)]
    dt = 1.0 / 30
    rt = Retargeter(solver, preprocess, initial_q)

    start = time.perf_counter()
    out = rt.retarget_clips(clips, dt)
    elapsed = time.perf_counter() - start
    frames_total = sum(len(c) for c in clips)
    print(
        f"retargeted {len(clips)} clips ({frames_total} frames, batched) "
        f"in {elapsed:.2f} s"
    )

    kd = kin.Data(model, 1, n_tracked=1)
    wrist = model.joint_index("wrist")
    for i, clip in enumerate(clips):
        errs = []
        for t in range(len(clip)):
            Ro, po = kin.forward_kinematics(model, kd, out[i][t][None])
            errs.append(np.linalg.norm(po[0, wrist] - clip[t, 0]))
        print(f"clip {i}: max wrist error {max(errs):.4f} m over {len(clip)} frames")


if __name__ == "__main__":
    main()
