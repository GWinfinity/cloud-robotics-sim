"""Throughput benchmark for batched (vectorized) Genesis environments.

Measures steady-state physics/control throughput of
:class:`GenesisVectorizedEnv` running :class:`FrankaPickCubeVecTask` across a
grid of (device, num_envs) combinations. This answers the L1 question posed in
docs/ROBODOJO_P0_PLAN.md: how far the project's single-batched-scene path is
from GPU-parallel data production (ManiSkill3-style), and where it saturates.

Timing methodology:

- build time is measured separately and *excluded* from the throughput number
  (steady-state FPS is what RL/data pipelines amortize);
- a short warmup rolls the scene with random actions before timing;
- random actions come from a seeded CPU generator (host-side RNG cost is part
  of the honest per-step wall time on small GPUs, and is reported separately
  as ``action_gen_ms``);
- GPU memory is ``torch.cuda.max_memory_allocated()`` reset after build;
- OOM / build failures are recorded as result rows with ``error`` set and do
  not abort the remaining grid points.

Outputs ``results.json`` (machine-readable) and ``report.md`` (table) under
``--out``.

Example:
    python tools/bench_vec_throughput.py --n-envs 1,16,64,256 \
        --device cuda --steps 200 --out outputs/benchmarks/vec_throughput
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger("bench_vec_throughput")


@dataclass
class BenchRow:
    """One (device, num_envs) measurement."""

    device: str
    num_envs: int
    steps: int = 0
    build_s: float = 0.0
    warmup_s: float = 0.0
    total_s: float = 0.0
    steps_per_s: float = 0.0
    ms_per_step: float = 0.0
    action_gen_ms: float = 0.0
    resets: int = 0
    gpu_mem_mb: float | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)


def _device_info() -> dict:
    """Collect host/device metadata for the report header."""
    import torch

    info: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_total_mb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 2**20
        )
        info["torch_cuda"] = torch.version.cuda
    try:
        import genesis as gs

        info["genesis"] = getattr(gs, "__version__", "unknown")
    except Exception:  # pragma: no cover - genesis is installed here
        info["genesis"] = "unavailable"
    return info


def _bench_one(
    device: str,
    num_envs: int,
    steps: int,
    warmup_steps: int,
    seed: int,
    sim_dt: float,
    substeps: int = 2,
    performance_mode: bool = False,
    solver_iterations: int = 50,
    ls_iterations: int = 50,
    noslip_iterations: int = 0,
    self_collision: bool = True,
    hibernation: bool = False,
    render_config: "str | None" = None,
    add_camera: bool = False,
) -> BenchRow:
    """Measure one grid point; never raises (failures land in the row)."""
    import torch

    from cloud_robotics_sim.core.vec_tasks import FrankaPickCubeVecTask
    from cloud_robotics_sim.core.vectorized import GenesisVectorizedEnv, VecEnvConfig

    row = BenchRow(device=device, num_envs=num_envs)
    env: GenesisVectorizedEnv | None = None
    try:
        use_cuda = device == "cuda"
        config = VecEnvConfig(
            num_envs=num_envs,
            use_cuda=use_cuda,
            sim_dt=sim_dt,
            sim_substeps=substeps,
            performance_mode=performance_mode,
            solver_iterations=solver_iterations,
            ls_iterations=ls_iterations,
            noslip_iterations=noslip_iterations,
            self_collision=self_collision,
            hibernation=hibernation,
            render_config=render_config,
        )
        task = FrankaPickCubeVecTask(add_camera=add_camera)
        env = GenesisVectorizedEnv(config=config, task=task)

        t0 = time.perf_counter()
        env.initialize()
        row.build_s = time.perf_counter() - t0

        if use_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            row.gpu_mem_mb = round(torch.cuda.max_memory_allocated() / 2**20, 1)

        gen = torch.Generator().manual_seed(seed)

        def random_actions() -> torch.Tensor:
            # gs.init(backend=cuda) sets torch's *default* device to cuda,
            # so CPU-generator draws must pin device="cpu" explicitly.
            return torch.randn(num_envs, task.num_actions, generator=gen, device="cpu")

        # Warmup: initial reset + un-timed rollout (settles JIT/allocations).
        t0 = time.perf_counter()
        env.reset()
        for _ in range(warmup_steps):
            env.step(random_actions())
        if use_cuda:
            torch.cuda.synchronize()
        row.warmup_s = time.perf_counter() - t0

        # Timed loop (includes any mid-episode resets, as a real RL loop has).
        t0 = time.perf_counter()
        t_actions = 0.0
        resets = 0
        for _ in range(steps):
            ta = time.perf_counter()
            actions = random_actions()
            t_actions += time.perf_counter() - ta
            _obs, _rew, terminated, truncated, _infos = env.step(actions)
            done = terminated | truncated
            if bool(done.any()):
                resets += int(done.sum())
                env.reset_idx(done.nonzero(as_tuple=True)[0])
        if use_cuda:
            torch.cuda.synchronize()
        row.total_s = time.perf_counter() - t0
        row.steps = steps
        row.resets = resets
        row.action_gen_ms = round(1000.0 * t_actions / steps, 4)
        row.steps_per_s = round(steps / row.total_s, 1)
        row.ms_per_step = round(1000.0 * row.total_s / steps, 3)
        if use_cuda:
            row.gpu_mem_mb = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        row.notes.append(
            f"seed={seed}, sim_dt={sim_dt}, substeps={substeps}, "
            f"newton_iter={solver_iterations}, ls_iter={ls_iterations}, "
            f"noslip={noslip_iterations}, self_collision={self_collision}, "
            f"hibernation={hibernation}, render_config={render_config}, "
            f"camera={add_camera}"
        )
        return row
    except Exception as exc:  # noqa: BLE001 - recorded, never aborts the grid
        row.error = f"{type(exc).__name__}: {exc}"
        logger.warning("grid point failed: %s x %d: %s", device, num_envs, row.error)
        return row
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001 - teardown best effort
                logger.debug("env close failed: %s", exc)


def _write_report(
    out_dir: Path, rows: list[BenchRow], meta: dict, args: argparse.Namespace
) -> None:
    """Write results.json and a Markdown comparison table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "args": vars(args),
        "rows": [asdict(r) for r in rows],
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if meta.get("gpu"):
        gpu_line = f"- GPU: {meta['gpu']} ({meta.get('gpu_total_mb', '?')} MB)"
    else:
        gpu_line = "- GPU: none"
    lines = [
        "# Vectorized throughput baseline",
        "",
        gpu_line,
        f"- torch {meta['torch']}, genesis {meta.get('genesis', '?')}, "
        f"python {meta['python']}",
        f"- steps per point: {args.steps} (warmup {args.warmup}), "
        f"random pd_joint_delta_pos actions, no rendering (physics-only)",
        f"- solver: sim_dt={args.sim_dt} s x {args.substeps} substeps "
        f"(solver dt {args.sim_dt / args.substeps:g} s), "
        f"performance_mode={args.performance}, "
        f"newton_iter={args.solver_iters}, ls_iter={args.ls_iters}, "
        f"noslip={args.noslip}, self_collision={not args.no_self_collision}, "
        f"hibernation={args.hibernation}, render_config={args.render_config}, "
        f"camera={args.camera}",
        "",
        "| device | n_envs | steps/s | ms/step | action gen ms/step "
        "| resets | peak GPU MB | build s | error |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.device} | {r.num_envs} | {r.steps_per_s or '-'} "
            f"| {r.ms_per_step or '-'} | {r.action_gen_ms or '-'} "
            f"| {r.resets or '-'} | {r.gpu_mem_mb if r.gpu_mem_mb else '-'} "
            f"| {round(r.build_s, 2) if r.build_s else '-'} "
            f"| {r.error or '-'} |"
        )
    lines += [
        "",
        "## Reference point (different口径，仅作数量级参照）",
        "",
        "ManiSkill3 (RSS 2025) reports 30,000+ FPS for GPU simulation+rendering "
        "of homogeneous benchmark environments on a high-end GPU. The numbers "
        "above are physics-only (no cameras), on the local GPU — compare "
        "orders of magnitude, not exact values.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--n-envs",
        default="1,16,64,256",
        help="Comma-separated num_envs grid (default: 1,16,64,256).",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        required=True,
        help="Compute device for the batched scene (run twice to compare).",
    )
    parser.add_argument("--steps", type=int, default=200, help="Timed steps per point.")
    parser.add_argument(
        "--warmup", type=int, default=10, help="Warmup steps per point."
    )
    parser.add_argument("--seed", type=int, default=0, help="Action RNG seed.")
    parser.add_argument("--sim-dt", type=float, default=0.02, help="Scene timestep.")
    parser.add_argument(
        "--substeps",
        type=int,
        default=2,
        help="Solver substeps per scene.step (solver dt = sim_dt/substeps).",
    )
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Enable gs.init(performance_mode=True) (static arrays; no scene edits).",
    )
    parser.add_argument(
        "--solver-iters",
        type=int,
        default=50,
        help="Newton constraint-solver iterations (RigidOptions.iterations).",
    )
    parser.add_argument(
        "--ls-iters",
        type=int,
        default=50,
        help="Newton line-search iterations (RigidOptions.ls_iterations).",
    )
    parser.add_argument(
        "--noslip",
        type=int,
        default=0,
        help="Noslip solver iterations (RigidOptions.noslip_iterations; "
        "the project's earlier default of 5 made the noslip kernel ~80% of "
        "CUDA step time — see outputs/benchmarks/profile_vec_step_20261004).",
    )
    parser.add_argument(
        "--no-self-collision",
        action="store_true",
        help="Disable self-collision within each articulated entity.",
    )
    parser.add_argument(
        "--hibernation",
        action="store_true",
        help="Enable solver hibernation for near-static envs.",
    )
    parser.add_argument(
        "--render-config",
        default=None,
        help="Path to a render config yaml (e.g. configs/render/"
        "batch_madrona.yaml); mode=batch needs the Linux-only gs-madrona.",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Add a batched third-person camera to the task and render every "
        "step (sim+render 口径).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output dir (default: outputs/benchmarks/vec_throughput_<device>).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            logger.error("--device cuda requested but torch.cuda is unavailable")
            return 2

    try:
        n_envs_grid = [int(x) for x in args.n_envs.split(",") if x.strip()]
    except ValueError:
        logger.error("could not parse --n-envs %r", args.n_envs)
        return 2
    if not n_envs_grid:
        logger.error("empty --n-envs grid")
        return 2

    meta = _device_info()
    rows = [
        _bench_one(
            args.device,
            n,
            args.steps,
            args.warmup,
            args.seed,
            args.sim_dt,
            substeps=args.substeps,
            performance_mode=args.performance,
            solver_iterations=args.solver_iters,
            ls_iterations=args.ls_iters,
            noslip_iterations=args.noslip,
            self_collision=not args.no_self_collision,
            hibernation=args.hibernation,
            render_config=args.render_config,
            add_camera=args.camera,
        )
        for n in n_envs_grid
    ]

    out_dir = Path(args.out or f"outputs/benchmarks/vec_throughput_{args.device}")
    _write_report(out_dir, rows, meta, args)

    print(f"\n{'device':<8} {'n_envs':>6} {'steps/s':>10} {'ms/step':>9} {'GPU MB':>8}")
    for r in rows:
        print(
            f"{r.device:<8} {r.num_envs:>6} {r.steps_per_s:>10} "
            f"{r.ms_per_step:>9} {r.gpu_mem_mb or '-':>8}  {r.error or ''}"
        )
    print(f"\nreports: {out_dir.resolve() / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
