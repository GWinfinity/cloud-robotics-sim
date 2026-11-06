"""End-to-end SuperMimic training on the toy reference task (CPU, minutes).

Run from the repo root:

    python plugins/learn/supermimic/examples/toy_train.py

The toy task (two-joint arm tracking a sinusoid with context-dependent
dynamics) is tiny, but the learning machinery is the full SuperMimic loop:
world model trained on real transitions, policy improved by BPTT through
imagined rollouts. Watch ``imag_rew`` climb toward 1 and ``wm_err8`` fall
in ``runs/supermimic_toy/train_log.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the plugin's ``supermimic`` package importable without installation.
_CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_DIR))

from supermimic.learner import SuperMimicConfig, SuperMimicLearner  # noqa: E402
from supermimic.toys import ToyEnv  # noqa: E402


def main() -> None:
    """Train a SuperMimic policy on the toy tracking task."""
    # Small networks keep this example CPU-friendly; scale the hidden sizes
    # up (e.g. the upstream 2048-wide defaults) for real tasks.
    cfg = SuperMimicConfig(
        iterations=200,
        buffer_steps=1000,
        utd=0.4,
        horizon=16,
        goal_frames=8,
        hist=8,
        wm_hidden=(256, 256),
        policy_hidden=(256, 128),
        wm_window=8,
        wm_batches=4,
        wm_batch=512,
        pi_batch=512,
        warmup_steps=64,
        wm_only_iters=20,
        ckpt_every=100,
        log_every=10,
        log_backend="none",
        compile=False,  # CPU smoke-friendly; enable on CUDA
        # Stabilizers ported from allmimic/InstantTrack practice:
        ema_decay=0.999,
        value_clip=True,
        noise_free_frac=0.25,
        pi_ref_skip=True,  # policy = residual on the next-frame reference
    )
    env = ToyEnv(num_envs=64)
    learner = SuperMimicLearner(env, cfg, "runs/supermimic_toy", backend="toy")
    result = learner.train()
    print("done:", result)


if __name__ == "__main__":
    main()
