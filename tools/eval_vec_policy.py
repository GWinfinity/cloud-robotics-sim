"""Three-way policy evaluation for batched VecTask environments.

Compares a trained PPO checkpoint against two baselines — the zero action
and seeded random actions — on the same Franka pick-cube task, and writes
``report.json`` + ``report.md`` (success rate / mean return / mean episode
length / throughput per policy).

The actor is rebuilt from the checkpoint's self-description
(``checkpoint["infos"]`` written by :class:`CrsOnPolicyRunner`), so no
training-side code or config has to be replayed: only the checkpoint file.

Each policy starts from a full reset of the same batched env; the cube
layout draws come from the same distribution per policy (independent draws —
comparable in aggregate, not paired per episode).

Requires the optional ``rl`` extra (``uv sync --extra rl``).

Example:
    python tools/eval_vec_policy.py --checkpoint outputs/rl/runs/exp1/model_200.pt \
        --envs 16 --episodes 48 --device cuda
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger("eval_vec_policy")


@dataclass
class PolicyMetrics:
    """Aggregated rollout metrics for one policy."""

    policy: str
    episodes: int = 0
    successes: int = 0
    success_rate: float = 0.0
    mean_return: float = 0.0
    mean_episode_length: float = 0.0
    steps: int = 0
    wall_s: float = 0.0
    steps_per_s: float = 0.0


def _evaluate(
    wrapper: object,
    act_fn: object,
    episodes_target: int,
    max_steps: int,
) -> PolicyMetrics:
    """Roll one policy until ``episodes_target`` episodes complete.

    Args:
        wrapper: :class:`RslRlVecEnvWrapper` (already reset).
        act_fn: Callable taking the observations TensorDict and returning
            ``(num_envs, num_actions)`` actions on the env device.
        episodes_target: Number of completed episodes to collect.
        max_steps: Episode cap (also the rollout's per-episode upper bound).

    Returns:
        Aggregated :class:`PolicyMetrics`.
    """
    import torch

    metrics = PolicyMetrics(policy=getattr(act_fn, "__name__", "policy"))
    num_envs = wrapper.num_envs
    device = wrapper.device

    running_return = torch.zeros(num_envs, dtype=torch.float64, device=device)
    running_length = torch.zeros(num_envs, dtype=torch.long, device=device)
    sum_return = 0.0
    sum_length = 0
    successes = 0
    episodes = 0
    steps = 0
    # Every episode lasts <= max_steps steps, so this cap is never hit in
    # practice; it only guards against a pathological no-done env.
    hard_cap = max(1000, episodes_target * max_steps)

    obs_td, _infos = wrapper.reset()
    t0 = time.perf_counter()
    while episodes < episodes_target and steps < hard_cap:
        actions = act_fn(obs_td)
        obs_td, rewards, dones, extras = wrapper.step(actions)
        steps += 1
        running_return += rewards.to(torch.float64)
        running_length += 1
        if bool(dones.any()):
            idx = dones.nonzero(as_tuple=True)[0]
            success = extras.get("success")
            for i in idx.tolist():
                sum_return += float(running_return[i])
                sum_length += int(running_length[i])
                if success is not None and bool(success[i]):
                    successes += 1
                episodes += 1
                running_return[i] = 0.0
                running_length[i] = 0
    wall = time.perf_counter() - t0

    metrics.episodes = episodes
    metrics.successes = successes
    metrics.success_rate = round(successes / max(episodes, 1), 4)
    metrics.mean_return = round(sum_return / max(episodes, 1), 4)
    metrics.mean_episode_length = round(sum_length / max(episodes, 1), 2)
    metrics.steps = steps
    metrics.wall_s = round(wall, 2)
    metrics.steps_per_s = round(steps / wall, 1) if wall > 0 else 0.0
    return metrics


def _write_report(
    out_dir: Path, rows: list[PolicyMetrics], meta: dict, checkpoint_info: dict
) -> None:
    """Write report.json and a Markdown comparison table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "checkpoint_info": checkpoint_info,
        "policies": [asdict(r) for r in rows],
    }
    (out_dir / "report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    lines = [
        "# Vec policy evaluation: policy vs zero vs random",
        "",
        f"- checkpoint: `{meta['checkpoint']}`",
        f"- device: {meta['device']}, envs: {meta['num_envs']}, "
        f"episodes per policy: {meta['episodes_target']}, "
        f"max episode steps: {meta['max_steps']}",
        f"- reward_mode: {meta['reward_mode']}, seed: {meta['seed']}",
        f"- host: torch {meta['torch']}, python {meta['python']}",
        "",
        "| policy | episodes | success rate | mean return "
        "| mean episode length | steps/s |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.policy} | {r.episodes} | {r.success_rate:.2%} "
            f"| {r.mean_return} | {r.mean_episode_length} | {r.steps_per_s} |"
        )
    lines += [
        "",
        "Policies start from independent full-reset draws of the same cube "
        "sampling distribution — compare in aggregate, not per episode.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="model_*.pt path.")
    parser.add_argument("--envs", type=int, default=16, help="Batched env count.")
    parser.add_argument(
        "--episodes", type=int, default=48, help="Completed episodes per policy."
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Episode cap (default: the checkpoint's max_episode_steps).",
    )
    parser.add_argument(
        "--reward-mode",
        choices=["sparse", "guided"],
        default=None,
        help="Override the checkpoint's reward mode (default: read from infos).",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Default: read " "the checkpoint's train_args if present, else cuda.",
    )
    parser.add_argument("--sim-dt", type=float, default=None)
    parser.add_argument("--substeps", type=int, default=None)
    parser.add_argument(
        "--performance",
        action="store_true",
        help="gs.init(performance_mode=True) (off unless passed).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default=None,
        help="Output dir (default: outputs/rl/eval_<timestamp>).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    import torch

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("checkpoint not found: %s", ckpt_path)
        return 2
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    infos = checkpoint.get("infos") or {}
    sim_info = infos.get("sim") or {}

    device = args.device or (infos.get("train_args") or {}).get("device") or "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning(
            "cuda unavailable per torch.cuda.is_available() — falling back to cpu"
        )
        device = "cpu"
    reward_mode = args.reward_mode or infos.get("reward_mode", "sparse")
    max_steps = int(args.max_steps or infos.get("max_episode_steps") or 200)
    sim_dt = float(
        args.sim_dt if args.sim_dt is not None else sim_info.get("sim_dt", 0.02)
    )
    substeps = int(
        args.substeps if args.substeps is not None else sim_info.get("sim_substeps", 2)
    )
    target = infos.get("target") or [0.5, 0.0, 0.1]
    init_x = infos.get("init_x_range") or [0.4, 0.6]
    init_y = infos.get("init_y_range") or [-0.2, 0.2]

    if args.envs <= 0 or args.episodes <= 0 or max_steps <= 0:
        logger.error("--envs, --episodes and --max-steps must be positive")
        return 2

    from cloud_robotics_sim.core.vec_rsl_rl import (
        RslRlVecEnvWrapper,
        make_actor_from_checkpoint,
    )
    from cloud_robotics_sim.core.vec_tasks import FrankaPickCubeVecTask
    from cloud_robotics_sim.core.vectorized import GenesisVectorizedEnv, VecEnvConfig

    actor = make_actor_from_checkpoint(checkpoint, device=device)

    config = VecEnvConfig(
        num_envs=args.envs,
        use_cuda=device == "cuda",
        sim_dt=sim_dt,
        sim_substeps=substeps,
        performance_mode=args.performance,
    )
    task = FrankaPickCubeVecTask(
        target=(float(target[0]), float(target[1]), float(target[2])),
        max_episode_steps=max_steps,
        reward_mode=reward_mode,
        init_x_range=(float(init_x[0]), float(init_x[1])),
        init_y_range=(float(init_y[0]), float(init_y[1])),
    )

    env: GenesisVectorizedEnv | None = None
    try:
        env = GenesisVectorizedEnv(config=config, task=task)
        env.initialize()
        wrapper = RslRlVecEnvWrapper(env)

        gen = torch.Generator().manual_seed(args.seed)

        def policy_actions(obs_td: object) -> torch.Tensor:
            return actor(obs_td, stochastic_output=False)

        def zero_actions(obs_td: object) -> torch.Tensor:
            return torch.zeros(
                wrapper.num_envs, wrapper.num_actions, device=wrapper.device
            )

        def random_actions(obs_td: object) -> torch.Tensor:
            # gs.init(backend=cuda) sets torch's default device to cuda, so
            # CPU-generator draws pin device="cpu" explicitly (bench tool同款).
            return (
                torch.randn(
                    wrapper.num_envs, wrapper.num_actions, generator=gen, device="cpu"
                )
                .clamp_(-1.0, 1.0)
                .to(wrapper.device)
            )

        rows = []
        for name, fn in (
            ("policy", policy_actions),
            ("zero", zero_actions),
            ("random", random_actions),
        ):
            metrics = _evaluate(wrapper, fn, args.episodes, max_steps)
            metrics.policy = name
            rows.append(metrics)
            logger.info(
                "%s: %d episodes, success %.1f%%, return %.3f",
                name,
                metrics.episodes,
                metrics.success_rate * 100,
                metrics.mean_return,
            )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001 - teardown best effort
                logger.debug("env close failed: %s", exc)

    out_dir = Path(args.out or f"outputs/rl/eval_{time.strftime('%Y%m%d_%H%M%S')}")
    meta = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(ckpt_path.resolve()),
        "device": device,
        "num_envs": args.envs,
        "episodes_target": args.episodes,
        "max_steps": max_steps,
        "reward_mode": reward_mode,
        "seed": args.seed,
        "sim_dt": sim_dt,
        "sim_substeps": substeps,
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
    for dist in ("cloud-robotics-sim", "rsl-rl-lib", "genesis-world"):
        try:
            meta[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover
            meta[dist] = "unavailable"
    _write_report(out_dir, rows, meta, infos)
    print(f"\nreports: {(out_dir / 'report.md').resolve()}")
    print(f"         {(out_dir / 'report.json').resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
