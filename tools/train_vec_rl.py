"""PPO training entry point for batched VecTask environments (rsl-rl 5.x).

Trains :class:`FrankaPickCubeVecTask` on a single batched Genesis scene
through the rsl-rl 5.x stack: :class:`RslRlVecEnvWrapper` adapts the
gymnasium-style 5-tuple env to the TensorDict contract, and
:class:`CrsOnPolicyRunner` stamps every ``model_*.pt`` checkpoint with a
self-description (obs/action dims, reward mode, network config, sim snapshot,
seed, versions) so :mod:`tools.eval_vec_policy` can rebuild the actor without
re-running training.

Requires the optional ``rl`` extra (``uv sync --extra rl``).

Example:
    python tools/train_vec_rl.py --device cuda --envs 256 --iterations 200 \
        --reward-mode guided --hidden-dims 512,512 --out outputs/rl/runs/exp1
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import platform
import sys
import time
from pathlib import Path

logger = logging.getLogger("train_vec_rl")


def _parse_pair(text: str, name: str) -> tuple[float, float]:
    """Parse ``"lo,hi"`` into a float pair (shared by range CLI knobs)."""
    try:
        lo, hi = (float(x) for x in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{name} must be 'lo,hi', got {text!r}"
        ) from exc
    return (lo, hi)


def _parse_target(text: str) -> tuple[float, float, float]:
    """Parse ``"x,y,z"`` into the cube target position."""
    try:
        parts = [float(x) for x in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--target must be 'x,y,z', got {text!r}"
        ) from exc
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--target must have 3 components, got {text!r}"
        )
    return (parts[0], parts[1], parts[2])


def _parse_hidden_dims(text: str) -> list[int]:
    """Parse ``"256,256,256"`` into hidden layer widths."""
    try:
        dims = [int(x) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--hidden-dims must be a comma-separated int list, got {text!r}"
        ) from exc
    if not dims or any(d <= 0 for d in dims):
        raise argparse.ArgumentTypeError(
            f"--hidden-dims must be positive, got {text!r}"
        )
    return dims


def _versions() -> dict:
    """Version snapshot embedded into the self-describing checkpoint."""
    import torch

    out = {"torch": torch.__version__, "python": platform.python_version()}
    for dist in ("cloud-robotics-sim", "rsl-rl-lib", "genesis-world"):
        try:
            out[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover
            out[dist] = "unavailable"
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    env_g = parser.add_argument_group("environment / task")
    env_g.add_argument("--envs", type=int, default=128, help="Batched env count.")
    env_g.add_argument(
        "--device", choices=["cuda", "cpu"], default="cuda", help="Compute device."
    )
    env_g.add_argument("--sim-dt", type=float, default=0.02, help="Scene timestep.")
    env_g.add_argument(
        "--substeps",
        type=int,
        default=2,
        help="Solver substeps per scene.step (solver dt = sim_dt/substeps).",
    )
    env_g.add_argument(
        "--performance",
        action="store_true",
        help="gs.init(performance_mode=True) (static arrays; no scene edits).",
    )
    env_g.add_argument(
        "--reward-mode",
        choices=["sparse", "guided"],
        default="sparse",
        help="'sparse' = historical success-dist reward (default, bit-for-bit "
        "compatible); 'guided' = dense lift/action shaping.",
    )
    env_g.add_argument(
        "--init-x-range",
        type=lambda s: _parse_pair(s, "--init-x-range"),
        default=(0.4, 0.6),
        metavar="LO,HI",
        help="Cube x sampling interval (default: 0.4,0.6).",
    )
    env_g.add_argument(
        "--init-y-range",
        type=lambda s: _parse_pair(s, "--init-y-range"),
        default=(-0.2, 0.2),
        metavar="LO,HI",
        help="Cube y sampling interval (default: -0.2,0.2).",
    )
    env_g.add_argument(
        "--max-episode-steps", type=int, default=200, help="Episode step cap."
    )
    env_g.add_argument(
        "--target",
        type=_parse_target,
        default=(0.5, 0.0, 0.1),
        metavar="X,Y,Z",
        help="Cube target position (default: 0.5,0,0.1).",
    )

    net_g = parser.add_argument_group("network")
    net_g.add_argument(
        "--hidden-dims",
        type=_parse_hidden_dims,
        default=[256, 256, 256],
        metavar="D,...",
        help="MLP hidden layer widths (default: 256,256,256).",
    )
    net_g.add_argument("--activation", default="elu", help="MLP activation.")
    net_g.add_argument(
        "--no-obs-normalization",
        action="store_true",
        help="Disable actor/critic observation normalization (on by default).",
    )
    net_g.add_argument(
        "--init-std", type=float, default=1.0, help="Initial Gaussian action std."
    )

    ppo_g = parser.add_argument_group("ppo")
    ppo_g.add_argument("--iterations", type=int, default=100, help="PPO iterations.")
    ppo_g.add_argument(
        "--steps-per-env",
        type=int,
        default=24,
        help="Rollout steps per env per iteration.",
    )
    ppo_g.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    ppo_g.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    ppo_g.add_argument("--lam", type=float, default=0.95, help="GAE lambda.")
    ppo_g.add_argument("--clip-param", type=float, default=0.2, help="PPO clip param.")
    ppo_g.add_argument("--entropy-coef", type=float, default=0.01)
    ppo_g.add_argument("--value-loss-coef", type=float, default=1.0)
    ppo_g.add_argument("--max-grad-norm", type=float, default=1.0)
    ppo_g.add_argument(
        "--epochs", type=int, default=5, help="Learning epochs per iteration."
    )
    ppo_g.add_argument(
        "--mini-batches", type=int, default=4, help="Mini-batches per learning epoch."
    )
    ppo_g.add_argument(
        "--desired-kl",
        type=float,
        default=0.01,
        help="Target KL for the adaptive learning-rate schedule.",
    )
    ppo_g.add_argument(
        "--save-interval",
        type=int,
        default=50,
        help="Iterations between model_*.pt saves.",
    )

    misc_g = parser.add_argument_group("misc")
    misc_g.add_argument("--seed", type=int, default=1)
    misc_g.add_argument(
        "--resume",
        default=None,
        help="Checkpoint to continue from (runner.load; iteration resumes).",
    )
    misc_g.add_argument(
        "--out",
        default=None,
        help="Run dir (default: outputs/rl/runs/<timestamp>); tensorboard logs "
        "go to <out>/logs and the final checkpoint to <out>/model_<it>.pt.",
    )
    misc_g.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error("--device cuda requested but torch.cuda is unavailable")
        return 2
    if args.envs <= 0 or args.iterations <= 0:
        logger.error("--envs and --iterations must be positive")
        return 2

    out_dir = Path(args.out or f"outputs/rl/runs/{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    import numpy as np

    np.random.seed(args.seed)

    from cloud_robotics_sim.core.vec_rsl_rl import (
        CrsOnPolicyRunner,
        RslRlVecEnvWrapper,
        build_train_cfg,
    )
    from cloud_robotics_sim.core.vec_tasks import FrankaPickCubeVecTask
    from cloud_robotics_sim.core.vectorized import GenesisVectorizedEnv, VecEnvConfig

    config = VecEnvConfig(
        num_envs=args.envs,
        use_cuda=args.device == "cuda",
        sim_dt=args.sim_dt,
        sim_substeps=args.substeps,
        performance_mode=args.performance,
    )
    task = FrankaPickCubeVecTask(
        target=args.target,
        max_episode_steps=args.max_episode_steps,
        reward_mode=args.reward_mode,
        init_x_range=args.init_x_range,
        init_y_range=args.init_y_range,
    )

    env: GenesisVectorizedEnv | None = None
    try:
        env = GenesisVectorizedEnv(config=config, task=task)
        env.initialize()

        train_cfg = build_train_cfg(
            num_steps_per_env=args.steps_per_env,
            save_interval=args.save_interval,
            seed=args.seed,
            hidden_dims=args.hidden_dims,
            activation=args.activation,
            obs_normalization=not args.no_obs_normalization,
            init_std=args.init_std,
            learning_rate=args.lr,
            gamma=args.gamma,
            lam=args.lam,
            clip_param=args.clip_param,
            entropy_coef=args.entropy_coef,
            value_loss_coef=args.value_loss_coef,
            max_grad_norm=args.max_grad_norm,
            num_learning_epochs=args.epochs,
            num_mini_batches=args.mini_batches,
            desired_kl=args.desired_kl,
        )

        checkpoint_info = {
            "task": "franka_pick_cube",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reward_mode": task.reward_mode,
            "obs_dim": env.num_observations,
            "num_actions": env.num_actions,
            "num_envs": env.num_envs,
            "max_episode_steps": task.max_episode_steps,
            "target": [float(v) for v in task.target],
            "init_x_range": [float(v) for v in task.init_x_range],
            "init_y_range": [float(v) for v in task.init_y_range],
            "seed": args.seed,
            "sim": {
                "sim_dt": config.sim_dt,
                "sim_substeps": config.sim_substeps,
                "performance_mode": config.performance_mode,
                "integrator": config.integrator,
                "solver_iterations": config.solver_iterations,
                "ls_iterations": config.ls_iterations,
                "noslip_iterations": config.noslip_iterations,
                "self_collision": config.self_collision,
                "hibernation": config.hibernation,
            },
            "network": {
                "hidden_dims": list(train_cfg["actor"]["hidden_dims"]),
                "activation": train_cfg["actor"]["activation"],
                "obs_normalization": train_cfg["actor"]["obs_normalization"],
                "distribution_cfg": train_cfg["actor"]["distribution_cfg"],
            },
            "versions": _versions(),
            "train_args": vars(args),
        }

        runner = CrsOnPolicyRunner(
            env=RslRlVecEnvWrapper(env),
            train_cfg=train_cfg,
            log_dir=str(out_dir / "logs"),
            device=args.device,
            checkpoint_info=checkpoint_info,
        )
        if args.resume:
            runner.load(args.resume)
            logger.info(
                "resumed from %s at iteration %d",
                args.resume,
                runner.current_learning_iteration,
            )

        runner.learn(args.iterations, init_at_random_ep_len=True)

        final_path = out_dir / f"model_{runner.current_learning_iteration}.pt"
        runner.save(str(final_path))
        logger.info("final checkpoint: %s", final_path)

        (out_dir / "train_config.json").write_text(
            json.dumps(
                {"train_cfg": train_cfg, "checkpoint_info": checkpoint_info},
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"\ncheckpoint: {final_path}")
        print(f"tensorboard logs: {(out_dir / 'logs').resolve()}")
        return 0
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001 - teardown best effort
                logger.debug("env close failed: %s", exc)


if __name__ == "__main__":
    sys.exit(main())
