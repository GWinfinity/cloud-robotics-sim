"""Command-line interface for Cloud Robotics Simulation Platform.

Provides commands for training, evaluation, and agent deployment.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

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


def agent_command(args: argparse.Namespace) -> int:
    """Run agent in interactive mode."""
    logger.info(f"Starting agent with goal: {args.goal}")

    # Agent runtime implementation would go here
    logger.info("Agent execution completed")
    return 0


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
  %(prog)s agent --goal "pick up the red cube"
  %(prog)s test
  %(prog)s clean-cache
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
        help="Run agent in interactive mode",
    )
    agent_parser.add_argument(
        "--goal",
        "-g",
        required=True,
        help="Natural language goal for the agent",
    )
    agent_parser.add_argument(
        "--config",
        help="Optional agent configuration",
    )
    agent_parser.set_defaults(func=agent_command)

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

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    result = args.func(args)
    return 0 if result is None else int(result)


if __name__ == "__main__":
    sys.exit(main())
