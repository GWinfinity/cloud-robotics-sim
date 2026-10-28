"""Roll out FrankaPickCubeVecTask episodes and export a trajclass dataset.

Labels are derived from the recorded observation stream (no extra getters —
the obs already carries ``cube_pos - target`` in channels 18..20):

- ``success``  — episode ended by the task's success termination with the
  recorded terminal obs inside the target sphere (``--synthetic-success-frac``
  of envs reach it via scripted cube guidance; flagged ``synthetic=True``)
- ``no_grasp`` — episode truncated with the cube never lifted
- ``dropped``   — lifted mid-episode, lost by the end (needs a real grasp
  policy to produce; random motion bats the cube sideways, not upward)

Class balance: random-walk actions produce ``no_grasp``; a fraction of envs
(``--synthetic-success-frac``) get their cube kinematically guided to the
target a couple of steps in (arm quiet, so the flailing cannot knock the
cube into the sphere first) so the success class exists before a real grasp
policy (OMPL/RL) lands. Synthetic episodes carry the real observation
signature with a faked physics path — keep them out of real-grasp
evaluations.

CPU-first (``--backend cpu``): on the 4 GB GTX 1650 box CPU genesis is the
reliable path, and this probe needs no GPU.

Example:
    python tools/gen_traj_classifier_data.py \
        --out outputs/trajclass/pickcube_probe.npz \
        --num-envs 8 --episodes 64 --max-steps 60 --backend cpu
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch

from cloud_robotics_sim.core.vec_tasks import (
    NUM_OBSERVATIONS,
    SUCCESS_THRESHOLD,
    FrankaPickCubeVecTask,
)
from cloud_robotics_sim.core.vectorized import (
    GenesisVectorizedEnv,
    VecEnvConfig,
)
from cloud_robotics_sim.trajclass import LABEL_NAMES, EpisodeBatch, save_npz

#: obs channels carrying (cube_pos - target); target z is fixed by the task.
_CUBE_REL_SLICE = slice(18, 21)
_TARGET_Z = 0.1


class _Episode(TypedDict):
    """One finished rollout, before batch-level padding."""

    obs: np.ndarray
    act: np.ndarray
    rew: np.ndarray
    label: int
    synthetic: bool


def _label_episode(obs: np.ndarray, lift_min: float, terminated: bool) -> int:
    """Class index from a finished episode's stacked obs (T, 24).

    ``success`` requires the episode to have ended by the task's success
    termination *and* the final recorded obs to already sit inside the
    target sphere — a cube merely knocked through the sphere by a flailing
    arm (termination fires on proximity, not on grasping) must not count.
    """
    dist = np.linalg.norm(obs[:, _CUBE_REL_SLICE], axis=1)
    cube_z = obs[:, 20] + _TARGET_Z
    if terminated and float(dist[-1]) < SUCCESS_THRESHOLD:
        return LABEL_NAMES.index("success")
    if float(cube_z.max() - cube_z[0]) > lift_min:
        return LABEL_NAMES.index("dropped")
    return LABEL_NAMES.index("no_grasp")


def generate(args: argparse.Namespace) -> EpisodeBatch:
    """Roll out episodes in a batched scene until ``args.episodes`` are done."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = VecEnvConfig(num_envs=args.num_envs, use_cuda=args.backend == "cuda")
    task = FrankaPickCubeVecTask(max_episode_steps=args.max_steps)
    env = GenesisVectorizedEnv(config, task)
    env.initialize()

    n = args.num_envs
    n_synth = int(round(n * args.synthetic_success_frac))
    synth_envs = set(range(n_synth))
    # Lift almost immediately: with a flailing arm the cube gets knocked into
    # the target sphere (the task's success = proximity, not grasping) and
    # the episode terminates before the scripted guidance ever runs.
    synth_lift_start = 2
    lift_ramp = 15

    # Per-episode accumulation (tensors are cloned — the task reuses its
    # obs buffer across steps).
    obs_bufs: list[list[torch.Tensor]] = [[] for _ in range(n)]
    act_bufs: list[list[torch.Tensor]] = [[] for _ in range(n)]
    rew_bufs: list[list[float]] = [[] for _ in range(n)]
    steps = [0] * n
    lift_from: dict[int, torch.Tensor] = {}
    episodes: list[_Episode] = []

    obs, _ = env.reset()
    t0 = time.time()
    try:
        while len(episodes) < args.episodes:
            actions = torch.randn(n, task.num_actions) * args.random_scale
            for i in synth_envs:
                # Arm quiet for the whole scripted episode: a flailing arm
                # fights the kinematic cube guidance (and bats the cube
                # under the table); the gripper closes during the lift.
                actions[i, :7] = 0.0
                if steps[i] >= synth_lift_start:
                    actions[i, 7:] = 1.0

            for i in list(lift_from):
                k = steps[i] - synth_lift_start
                frac = min(1.0, k / lift_ramp)
                goal = torch.tensor([0.5, 0.0, _TARGET_Z], dtype=torch.float32)
                pos = lift_from[i] + (goal - lift_from[i]) * frac
                idx = torch.tensor([i], dtype=torch.long)
                task.cube.set_pos(pos.unsqueeze(0), envs_idx=idx)
                if frac >= 1.0:
                    lift_from.pop(i)

            for i in range(n):
                if steps[i] == synth_lift_start and i in synth_envs:
                    idx = torch.tensor([i], dtype=torch.long)
                    lift_from[i] = (
                        task.cube.get_pos(envs_idx=idx).reshape(3).cpu().float().clone()
                    )

            obs, rew, terminated, truncated, _ = env.step(actions)
            # Post-step obs, paired with the action that produced it: the
            # terminal state (which decides the outcome) is then part of the
            # recorded episode. Clone now — the task reuses its obs buffer.
            obs_next = obs.clone()
            for i in range(n):
                obs_bufs[i].append(obs_next[i])
                act_bufs[i].append(actions[i])
                rew_bufs[i].append(float(rew[i]))

            done_idx = (terminated | truncated).nonzero().reshape(-1).tolist()
            if done_idx:
                idx_t = torch.tensor(done_idx, dtype=torch.long)
                new_obs, _ = env.reset(envs_idx=idx_t)
                for j, i in enumerate(done_idx):
                    ep_obs = torch.stack(obs_bufs[i]).cpu().numpy().astype(np.float32)
                    ep_act = torch.stack(act_bufs[i]).cpu().numpy().astype(np.float32)
                    ep_rew = np.asarray(rew_bufs[i], dtype=np.float32)
                    episodes.append(
                        {
                            "obs": ep_obs,
                            "act": ep_act,
                            "rew": ep_rew,
                            "label": _label_episode(
                                ep_obs, args.lift_min, bool(terminated[i])
                            ),
                            "synthetic": i in synth_envs,
                        }
                    )
                    obs_bufs[i], act_bufs[i], rew_bufs[i] = [], [], []
                    steps[i] = 0
                    obs[idx_t[j]] = new_obs[j]
            for i in range(n):
                if i not in done_idx:
                    steps[i] += 1
            if len(episodes) % 16 < len(done_idx):
                print(
                    f"  episodes {len(episodes)}/{args.episodes} "
                    f"({time.time() - t0:.0f}s)"
                )
    finally:
        env.close()

    max_len = max(len(ep["rew"]) for ep in episodes)
    e = len(episodes)
    obs_dim = NUM_OBSERVATIONS
    act_dim = task.num_actions
    padded_obs = np.zeros((e, max_len, obs_dim), dtype=np.float32)
    padded_act = np.zeros((e, max_len, act_dim), dtype=np.float32)
    padded_rew = np.zeros((e, max_len), dtype=np.float32)
    lengths = np.zeros(e, dtype=np.int64)
    labels = np.zeros(e, dtype=np.int64)
    synthetic = np.zeros(e, dtype=bool)
    for k, ep in enumerate(episodes):
        length = len(ep["rew"])
        padded_obs[k, :length] = ep["obs"]
        padded_act[k, :length] = ep["act"]
        padded_rew[k, :length] = ep["rew"]
        lengths[k] = length
        labels[k] = ep["label"]
        synthetic[k] = ep["synthetic"]

    return EpisodeBatch(
        obs=padded_obs,
        act=padded_act,
        rew=padded_rew,
        lengths=lengths,
        labels=labels,
        synthetic=synthetic,
    )


def main() -> None:
    """CLI entry: parse args, generate, save, print the label distribution."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="output .npz path")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--synthetic-success-frac",
        type=float,
        default=0.4,
        help="fraction of envs whose success is kinematically scripted",
    )
    parser.add_argument(
        "--random-scale",
        type=float,
        default=0.3,
        help="stddev of the random-walk delta-joint actions",
    )
    parser.add_argument(
        "--lift-min",
        type=float,
        default=0.03,
        help="metres of cube lift that separates dropped from no_grasp",
    )
    args = parser.parse_args()

    batch = generate(args)
    save_npz(batch, args.out)
    counts = batch.label_counts()
    print(
        json.dumps(
            {
                "out": str(args.out),
                "episodes": batch.num_episodes,
                "labels": counts,
                "synthetic": int(batch.synthetic.sum()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
