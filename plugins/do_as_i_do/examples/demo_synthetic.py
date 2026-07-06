"""End-to-end demo: synthetic reconstruction -> retargeting -> simulation replay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from do_as_i_do.core.pipeline import DoAsIDoPipeline


def main():
    """Run the end-to-end scaffold with synthetic input."""
    parser = argparse.ArgumentParser(description="Run the do-as-i-do scaffold end-to-end.")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--num-frames", type=int, default=60)
    parser.add_argument("--output-dir", type=str, default="outputs/do_as_i_do")
    parser.add_argument(
        "--hand-type",
        type=str,
        default="allegro",
        choices=["allegro", "sharpa"],
        help="Hand type to use for the bimanual robot (default: allegro).",
    )
    args = parser.parse_args()

    config = DoAsIDoPipeline.default_config()
    config["headless"] = args.headless
    config["robot"]["hand_type"] = args.hand_type
    config["reconstruction"]["num_frames"] = args.num_frames

    pipeline = DoAsIDoPipeline(config)
    result = pipeline.run(video_path="synthetic_demo", output_dir=args.output_dir)

    print("\n=== Pipeline finished ===")
    print(f"Demo frames: {len(result['demo'])}")
    print(f"Robot trajectory frames: {len(result['robot_trajectory'])}")
    print(f"Simulation metrics: {result['simulation_metrics']}")
    print(f"Deployment files: {result['deployment_paths']}")


if __name__ == "__main__":
    main()
