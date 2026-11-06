"""Retargeter: the frax.ik.retargeter port.

Pipeline per batch of B independent clips stepped in lockstep:

* ``init``: converge onto the first frame from ``initial_q`` (batched
  adaptive-step differential IK, the frax Retargeter.init contract);
* ``step``: one diff-IK step per frame;
* ``retarget_clip``: init + T-1 steps over a (T, B, ...) frame array;
* ``retarget_clips``: pad a list of variable-length clips to a common length,
  batch them, unbatch the results (the OfflineRetargeter contract, minus the
  multi-device dispatch -- the batch dimension plays that role here).

``preprocess`` and ``initial_q`` are user callables, exactly as in frax:

* preprocess(frame_batch) -> ref dict of (B, ...) numpy arrays; the keys
  must match what the solver's tasks declare;
* initial_q(frame_batch) -> (B, nq) starting configuration for the batch.
"""

from collections.abc import Callable, Sequence

import numpy as np

from .solver import DiffIKSolver, integrate

MIN_STEP_RATIO = 1e-3  # frax solver.MIN_STEP_RATIO


class _RestartConfig:
    """A shorter-iteration view of a QPConfig that forces a cold start.

    ADMM warm starts only help when the previous iterate solved the *same*
    problem; handing a 1000-iterate state to a 100-iterate solver mostly
    wastes the first restart.  Marking reduced-budget configs with this type
    makes the retargeter reset (v, z, u) when they are installed.
    """

    def __init__(self, cfg, iters):
        self.cfg = cfg
        self.iters = iters
        self.rho = cfg.rho
        self.relax = cfg.relax


class Retargeter:
    """Differential-IK motion retargeter (batched over clips).

    Args:
        solver: :class:`~qd_ik.solver.DiffIKSolver`
        preprocess: frame batch -> ref dict
        initial_q: first frame batch -> (B, nq)
        converge_max_iters: iteration budget for the first-frame converge
        converge_dt: largest convergence step; defaults to 1 / solver.max_gain
        converge_tol: convergence tolerance on max |v|
        qp_iters_step: ADMM iterations for per-frame steps (warm-started);
            the first frame uses ``qp_iters_init``
        qp_iters_init: ADMM iterations for the converge loop
    """

    def __init__(
        self,
        solver: DiffIKSolver,
        preprocess: Callable,
        initial_q: Callable,
        converge_max_iters: int = 300,
        converge_dt: float | None = None,
        converge_tol: float = 1e-3,
        qp_iters_step: int | None = None,
        qp_iters_init: int | None = None,
    ):
        self.solver = solver
        self.preprocess = preprocess
        self.initial_q = initial_q
        self.converge_max_iters = int(converge_max_iters)
        if converge_dt is None:
            converge_dt = 1.0 / solver.max_gain
        self.converge_dt = float(converge_dt)
        self.converge_tol = float(converge_tol)
        base = solver.qp_config
        self._cfg_step = self._with_iters(base, qp_iters_step)
        self._cfg_init = self._with_iters(base, qp_iters_init)

    @staticmethod
    def _with_iters(cfg, iters):
        if iters is None:
            return cfg
        from .qp import QPConfig

        it = int(iters)
        if it >= cfg.iters:
            return QPConfig(iters=it, rho=cfg.rho, relax=cfg.relax)
        return _RestartConfig(cfg=cfg, iters=it)

    def _set_iters(self, data, cfg):
        """Point data.qp at a config; restarts get a cold ADMM warm start."""
        data.qp.config = cfg
        if isinstance(cfg, _RestartConfig):
            data.qp.reset()

    # ------------------------------------------------------------------ core
    def init(self, first_frames, data):
        """Converge onto the first frame batch.  Returns (q, iters)."""
        solver = self.solver
        B, model = data.B, solver.model
        q = np.ascontiguousarray(self.initial_q(first_frames), dtype=np.float64)
        if q.shape != (B, model.nq):
            raise ValueError(f"initial_q returned {q.shape}, want {(B, model.nq)}")
        ref = self.preprocess(first_frames)
        self._set_iters(data, self._cfg_init)
        solver.solve_v(q, ref, data, warm=False)
        cost = data.cost.copy()
        dt = np.full(B, self.converge_dt)
        dt_min = MIN_STEP_RATIO * self.converge_dt
        v_abs = np.abs(data.qp.v).max(axis=1)
        done = v_abs <= self.converge_tol
        it = np.zeros(B, dtype=np.int64)
        while not np.all(done) and int(it.max()) < self.converge_max_iters:
            active = ~done
            v_prev = data.qp.v.copy()
            u_prev = data.qp.u.copy()
            z_prev = data.qp.z.copy()
            q_new = integrate(model, q, data.qp.v, dt)
            solver.solve_v(q_new, ref, data, warm=True)
            cost_new = data.cost.copy()
            accept = (cost_new < cost) & active
            q[accept] = q_new[accept]
            cost[accept] = cost_new[accept]
            # frax keeps the warm start only on accepted steps
            keep_new = accept[:, None]
            data.qp.v[:] = np.where(keep_new, data.qp.v, v_prev)
            data.qp.u[:] = np.where(keep_new, data.qp.u, u_prev)
            data.qp.z[:] = np.where(keep_new, data.qp.z, z_prev)
            dt = np.where(accept, np.minimum(2.0 * dt, self.converge_dt), 0.5 * dt)
            it[active] += 1
            v_abs = np.abs(data.qp.v).max(axis=1)
            done = done | ((accept & (v_abs <= self.converge_tol)) | (dt < dt_min))
        return q, it

    def step(self, q, frames, dt, data):
        """One retargeting step.  Returns the new (B, nq) configuration."""
        self._set_iters(data, self._cfg_step)
        return self.solver.step(q, self.preprocess(frames), dt, data, warm=True)

    # ----------------------------------------------------------------- clips
    def retarget_clip(self, frames, dt, data=None):
        """Retarget a (T, B, ...) batch of frames.  Returns (T, B, nq)."""
        T = len(frames)
        if data is None:
            data = self.solver.data(self._batch_of(frames))
        q, _ = self.init(frames[0], data)
        qs = np.zeros((T, data.B, self.solver.model.nq))
        qs[0] = q
        for t in range(1, T):
            q = self.step(q, frames[t], dt, data)
            qs[t] = q
        return qs

    def retarget_clips(self, clips: Sequence, dt):
        """Retarget variable-length clips, batched.  Returns a list of
        (T_i, nq) arrays, one per input clip.
        """
        if not clips:
            return []
        lengths = [len(c) for c in clips]
        T = max(lengths)
        padded = [
            np.concatenate([c, np.repeat(c[-1:], T - len(c), axis=0)]) for c in clips
        ]
        batch = np.stack(padded, axis=1)  # (T, B, ...) time-major
        data = self.solver.data(len(clips))
        qs = self.retarget_clip(batch, dt, data)
        return [qs[: lengths[i], i] for i in range(len(clips))]

    @staticmethod
    def _batch_of(frames):
        arr = np.asarray(frames)
        return arr.shape[1] if arr.ndim >= 2 else 1
