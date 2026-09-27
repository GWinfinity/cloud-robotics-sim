"""Command-line interface for Cloud Robotics Simulation Platform.

Provides commands for training, evaluation, and agent deployment.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_command(args: argparse.Namespace) -> int:
    """Run training with specified configuration.

    Supports both reinforcement learning (RL) and imitation learning (IL).
    """
    logger.info(f"Starting training with config: {args.config}")

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    # Training implementation would go here
    logger.info("Training completed")
    return 0


def eval_command(args: argparse.Namespace) -> int:
    """Evaluate a trained policy."""
    logger.info(f"Evaluating checkpoint: {args.checkpoint}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return 1

    # Evaluation implementation would go here
    logger.info("Evaluation complete. Success rate: 0.0")
    return 0


def _parse_cli_value(value: str) -> Any:
    """Parse a CLI string into int/float/bool/JSON/native string."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.startswith(("{", "[")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def _parse_kv_params(items: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--param must be KEY=VALUE, got: {item!r}")
        key, value = item.split("=", 1)
        params[key] = _parse_cli_value(value)
    return params


def _find_plugin_info(pm: Any, name: str, category: str | None = None) -> Any:
    """按名字解析插件（跨类别重名时报错并列出候选）。"""
    if category:
        return pm.get_plugin_info(category, name)
    matches = [
        (cat, n) for cat, names in pm.list_plugins().items() for n in names if n == name
    ]
    if not matches:
        available = ", ".join(n for names in pm.list_plugins().values() for n in names)
        raise ValueError(f"plugin not found: {name!r}; available: {available}")
    if len(matches) > 1:
        cats = ", ".join(f"{cat}/{n}" for cat, n in matches)
        raise ValueError(f"ambiguous plugin name {name!r}; specify --category: {cats}")
    return pm.get_plugin_info(*matches[0])


def plugins_list_command(args: argparse.Namespace) -> int:
    """列出所有（或某类别下的）插件。"""
    from cloud_robotics_sim.core.plugin_manager import get_plugin_manager

    pm = get_plugin_manager()
    rows = []
    for cat, names in sorted(pm.list_plugins(args.category).items()):
        for name in sorted(names):
            info = pm.get_plugin_info(cat, name)
            rows.append((cat, name, info.version, info.description))
    if not rows:
        print(f"no plugins found (category={args.category!r})")
        return 0
    w_cat = max(len(r[0]) for r in rows)
    w_name = max(len(r[1]) for r in rows)
    w_ver = max(len(r[2]) for r in rows)
    desc_width = 72 - (w_cat + w_name + w_ver + 6)
    for cat, name, ver, desc in rows:
        short = desc if len(desc) <= desc_width else desc[: desc_width - 1] + "…"
        print(f"{cat:<{w_cat}}  {name:<{w_name}}  {ver:<{w_ver}}  {short}")
    if args.verbose:
        print()
        for cat, name, _, desc in rows:
            info = pm.get_plugin_info(cat, name)
            print(f"[{cat}/{name}] {info.path}")
            if desc:
                print(f"  {desc}")
    return 0


_USAGE_HEADINGS = ("使用", "用法", "usage", "getting started", "快速开始", "quickstart")


def _extract_usage(readme_path: Path, max_lines: int = 40) -> list[str] | None:
    """从插件 README 提取使用说明段（## 使用 / ## Usage 等标题到下一个 ## ）"""
    if not readme_path.exists():
        return None
    lines = readme_path.read_text(encoding="utf-8", errors="replace").splitlines()
    start: int | None = None
    for i, line in enumerate(lines):
        if line.startswith("## ") and not line.startswith("### "):
            heading = line[3:].strip().lower()
            if any(h in heading for h in _USAGE_HEADINGS):
                start = i + 1
                break
    if start is None:
        return None
    body: list[str] = []
    for line in lines[start:]:
        if line.startswith("## "):
            break
        body.append(line)
    while body and not body[0].strip():
        body.pop(0)
    return body[:max_lines] or None


def plugins_info_command(args: argparse.Namespace) -> int:
    """查看插件详情：介绍、用法、导出、依赖、配置。"""
    from cloud_robotics_sim.core.plugin_config import config_defaults, read_overrides
    from cloud_robotics_sim.core.plugin_manager import get_plugin_manager

    pm = get_plugin_manager()
    try:
        info = _find_plugin_info(pm, args.name, args.category)
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    yaml_meta = info.config or {}
    print(f"{info.name} ({info.category}) v{info.version}")
    print(f"  path: {info.path}")
    if info.description:
        print(f"  description: {info.description}")
    for key in ("source_project", "type", "author", "tags"):
        if key in yaml_meta:
            print(f"  {key}: {yaml_meta[key]}")
    if info.exports:
        print(f"  exports: {', '.join(str(e) for e in info.exports)}")

    deps = yaml_meta.get("dependencies")
    if isinstance(deps, dict):  # 新格式: required/optional
        if deps.get("required"):
            print(f"  deps(required): {', '.join(map(str, deps['required']))}")
        if deps.get("optional"):
            print(f"  deps(optional): {', '.join(map(str, deps['optional']))}")
    elif deps:
        print(f"  dependencies: {', '.join(map(str, deps))}")

    entry_points = yaml_meta.get("entry_points")
    if entry_points:
        print(f"  entry_points: {entry_points}")

    defaults = config_defaults(yaml_meta)
    overrides = read_overrides(info.name)
    if defaults or overrides:
        print("  config:")
        for key in sorted(set(defaults) | set(overrides)):
            tag = "override" if key in overrides else "default"
            value = overrides.get(key, defaults.get(key))
            print(f"    {key} = {value!r}  ({tag})")

    usage = _extract_usage(info.path / "README.md")
    if usage:
        print("\n  Usage (from README):")
        for line in usage:
            print(f"    {line}")
    else:
        print(f"\n  (no usage section in {info.path / 'README.md'})")
    return 0


def plugins_config_command(args: argparse.Namespace) -> int:
    """查看/设置插件用户配置（覆盖 plugin.yaml 默认值）。"""
    from cloud_robotics_sim.core.plugin_config import (
        config_defaults,
        get_plugin_config,
        read_overrides,
        set_override,
        unset_override,
    )
    from cloud_robotics_sim.core.plugin_manager import get_plugin_manager

    pm = get_plugin_manager()
    try:
        info = _find_plugin_info(pm, args.name, args.category)
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    try:
        for item in args.set or []:
            if "=" not in item:
                logger.error("--set must be KEY=VALUE, got: %r", item)
                return 1
            key, value = item.split("=", 1)
            set_override(info.name, key, _parse_cli_value(value))
        for key in args.unset or []:
            unset_override(info.name, key)
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    defaults = config_defaults(info.config)
    overrides = read_overrides(info.name)
    merged = get_plugin_config(info.name, defaults)

    if args.defaults:
        print(f"# defaults for {info.name} (from plugin.yaml)")
        for key in sorted(defaults):
            print(f"{key} = {defaults[key]!r}")
        return 0

    print(f"# config for {info.name} ({'defaults + ' if merged else ''}user overrides)")
    if not merged:
        print("(no config)")
    for key in sorted(merged):
        tag = "override" if key in overrides else "default"
        print(f"{key} = {merged[key]!r}  ({tag})")
    return 0


def agent_command(args: argparse.Namespace) -> int:
    """Run an agent goal, a specific skill, or list available skills."""
    from cloud_robotics_sim.runtime.agent_hub import SimHub

    hub = SimHub()

    if args.list_skills:
        for skill in hub.registry.list_skills():
            print(f"{skill['name']}: {skill['description']}")
        return 0

    if args.skill:
        try:
            params = _parse_kv_params(args.param)
        except ValueError as exc:
            logger.error(str(exc))
            return 1
        record = hub.executor.execute(args.skill, params)
    elif args.goal:
        matches = hub.executor.resolve_goal(args.goal)
        if not matches:
            available = ", ".join(s["name"] for s in hub.registry.list_skills())
            logger.error(
                "no skill matches goal %r; available skills: %s", args.goal, available
            )
            return 1
        best, score = matches[0]
        logger.info("goal resolved to skill %r (score=%d)", best.name, score)
        record = hub.executor.execute_goal(args.goal)
    else:
        logger.error("provide --goal GOAL, --skill SKILL, or --list-skills")
        return 1

    print(json.dumps(record.to_dict(), ensure_ascii=False, indent=2))
    return 0 if record.status == "ok" else 1


def worker_command(args: argparse.Namespace) -> int:
    """Run the Redis queue worker (for Kubernetes/KEDA deployments)."""
    from cloud_robotics_sim.runtime.queue_worker import run_worker

    return run_worker(
        redis_url=args.redis_url,
        queue=args.queue,
        poll_interval=args.poll_interval,
    )


def test_command(args: argparse.Namespace) -> int:
    """Run test suite."""
    import subprocess

    logger.info("Running test suite")

    test_path = Path(__file__).parent.parent.parent / "tests"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_path), "-v"],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    return result.returncode


def clean_cache_command(args: argparse.Namespace) -> int:
    """Clean Genesis simulation cache or stage it for the dataset pipeline."""
    from cloud_robotics_sim.utils.cache_cleanup import cleanup_after_simulation

    cleanup_after_simulation(
        cache_dir=args.cache_dir,
        pipeline_dir=args.pipeline_dir,
        dataset_pipeline=args.dataset_pipeline,
    )
    return 0


def patents_command(args: argparse.Namespace) -> int:
    """Run or list classic patent simulations."""
    from cloud_robotics_sim.patents import list_patents, run_patent_simulation

    if args.list:
        print("Registered patent simulations:")
        for patent_id in list_patents():
            print(f"  {patent_id}")
        return 0

    if args.run is None:
        logger.error("Use --run PATENT_ID or --list")
        return 1

    parameters: dict = {}
    if args.param:
        for item in args.param:
            key, value = item.split("=", 1)
            try:
                value = float(value)
            except ValueError:
                pass
            parameters[key] = value

    final_state = run_patent_simulation(
        args.run,
        headless=args.headless,
        steps=args.steps,
        dt=args.dt,
        substeps=args.substeps,
        resolution=tuple(args.resolution),
        device=args.device,
        seed=args.seed,
        parameters=parameters,
        record_path=args.record,
        follow_entity=args.follow,
    )

    print(f"Final state for {args.run}:")
    print(f"  time: {final_state.time:.2f}s")
    print(f"  parameters: {final_state.parameters}")
    print(f"  metrics: {final_state.metrics}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="cloud-robotics-sim",
        description="Cloud Robotics Simulation Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s train --config configs/franka_pickplace.yaml
  %(prog)s eval --checkpoint checkpoints/latest.pt
  %(prog)s agent --goal "run patent US821393 headlessly"
  %(prog)s agent --skill run_patent --param run=US821393
  %(prog)s test
  %(prog)s clean-cache
  %(prog)s worker --queue sim-tasks-cpu
  %(prog)s patents --list
  %(prog)s patents --run US821393 --param thrust=0.8 --param wind_speed=5
  %(prog)s plugins list [--category solvers]
  %(prog)s plugins info wfc_scenes
  %(prog)s plugins config vr_bridge --set retarget.enabled=true
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train a policy using RL or IL",
    )
    train_parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to training configuration file",
    )
    train_parser.add_argument(
        "--output",
        "-o",
        default="./outputs",
        help="Output directory for checkpoints and logs",
    )
    train_parser.set_defaults(func=train_command)

    # Eval command
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate a trained policy",
    )
    eval_parser.add_argument(
        "--checkpoint",
        "-ckpt",
        required=True,
        help="Path to model checkpoint",
    )
    eval_parser.add_argument(
        "--num-episodes",
        "-n",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )
    eval_parser.set_defaults(func=eval_command)

    # Agent command
    agent_parser = subparsers.add_parser(
        "agent",
        help="Run an agent goal, a specific skill, or list available skills",
    )
    agent_parser.add_argument(
        "--goal",
        "-g",
        help="Natural language goal, resolved to the best-matching skill",
    )
    agent_parser.add_argument(
        "--skill",
        "-s",
        help="Run a specific skill by name (see --list-skills)",
    )
    agent_parser.add_argument(
        "--param",
        action="append",
        metavar="KEY=VALUE",
        help="Skill parameter (repeatable); numbers/bools/JSON auto-parsed",
    )
    agent_parser.add_argument(
        "--list-skills",
        action="store_true",
        help="List available skills and exit",
    )
    agent_parser.set_defaults(func=agent_command)

    # Worker command
    worker_parser = subparsers.add_parser(
        "worker",
        help="Run the Redis queue worker (Kubernetes/KEDA autoscaling)",
    )
    worker_parser.add_argument(
        "--redis-url",
        default=None,
        help="Redis connection URL (default: REDIS_URL env or redis://localhost:6379/0)",
    )
    worker_parser.add_argument(
        "--queue",
        default=None,
        help="Queue (Redis list) to consume (default: QUEUE_NAME env or sim-tasks-cpu)",
    )
    worker_parser.add_argument(
        "--poll-interval",
        type=float,
        default=None,
        help="Seconds between polls when idle (default: POLL_INTERVAL env or 2.0)",
    )
    worker_parser.set_defaults(func=worker_command)

    # Test command
    test_parser = subparsers.add_parser(
        "test",
        help="Run the test suite",
    )
    test_parser.set_defaults(func=test_command)

    # Clean-cache command
    clean_parser = subparsers.add_parser(
        "clean-cache",
        help="Clean Genesis simulation cache or stage it for the dataset pipeline",
    )
    clean_parser.add_argument(
        "--cache-dir",
        help="Simulation cache directory (env: CRS_SIM_CACHE_DIR)",
    )
    clean_parser.add_argument(
        "--pipeline-dir",
        help="Dataset pipeline staging directory (env: CRS_DATASET_PIPELINE_DIR)",
    )
    clean_parser.add_argument(
        "--dataset-pipeline",
        action="store_true",
        default=None,
        help="Stage cache for downstream dataset pipeline instead of deleting it",
    )
    clean_parser.set_defaults(func=clean_cache_command)

    # Patents command
    patents_parser = subparsers.add_parser(
        "patents",
        help="Run classic patent simulations in Genesis",
    )
    patents_parser.add_argument(
        "--list",
        action="store_true",
        help="List registered patent simulations",
    )
    patents_parser.add_argument(
        "--run",
        help="Patent ID to run (e.g. US821393)",
    )
    patents_parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run without the interactive viewer",
    )
    patents_parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of control steps",
    )
    patents_parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Physics timestep",
    )
    patents_parser.add_argument(
        "--substeps",
        type=int,
        default=10,
        help="Physics substeps per control step",
    )
    patents_parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[640, 480],
        help="Camera resolution width height",
    )
    patents_parser.add_argument(
        "--device",
        default="cuda",
        help="Genesis compute device (cuda or cpu)",
    )
    patents_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    patents_parser.add_argument(
        "--param",
        action="append",
        help="Parameter override in key=value format (repeatable)",
    )
    patents_parser.add_argument(
        "--record",
        help="Optional path to save an MP4 recording",
    )
    patents_parser.add_argument(
        "--follow",
        help="Entity name for the camera to follow (e.g. aircraft for US821393)",
    )
    patents_parser.set_defaults(func=patents_command)

    # Plugins command (list / info / config)
    plugins_parser = subparsers.add_parser(
        "plugins",
        help="Discover plugins: list, show info/usage, manage per-plugin config",
    )
    plugins_sub = plugins_parser.add_subparsers(dest="plugins_command")

    def _plugins_help_command(_args: argparse.Namespace) -> int:
        plugins_parser.print_help()
        return 0

    plugins_parser.set_defaults(func=_plugins_help_command)

    plugins_list = plugins_sub.add_parser("list", help="List discovered plugins")
    plugins_list.add_argument(
        "--category",
        help="Only list plugins under this category (e.g. controllers, solvers)",
    )
    plugins_list.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Also print each plugin's path and full description",
    )
    plugins_list.set_defaults(func=plugins_list_command)

    plugins_info = plugins_sub.add_parser(
        "info", help="Show plugin description, usage, exports, deps and config"
    )
    plugins_info.add_argument(
        "name", help="Plugin name (use --category to disambiguate)"
    )
    plugins_info.add_argument(
        "--category",
        help="Disambiguate when plugin names collide across categories",
    )
    plugins_info.set_defaults(func=plugins_info_command)

    plugins_config = plugins_sub.add_parser(
        "config",
        help="Show or set per-plugin user config (overrides plugin.yaml defaults)",
    )
    plugins_config.add_argument("name", help="Plugin name")
    plugins_config.add_argument(
        "--category",
        help="Disambiguate when plugin names collide across categories",
    )
    plugins_config.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="Set a config override (repeatable); numbers/bools/JSON auto-parsed",
    )
    plugins_config.add_argument(
        "--unset",
        action="append",
        metavar="KEY",
        help="Remove a config override (repeatable)",
    )
    plugins_config.add_argument(
        "--defaults",
        action="store_true",
        help="Show only the plugin.yaml default config and exit",
    )
    plugins_config.set_defaults(func=plugins_config_command)

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    result = args.func(args)
    return 0 if result is None else int(result)


if __name__ == "__main__":
    sys.exit(main())
