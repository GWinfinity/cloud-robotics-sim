"""Wright Flyer interactive demo.

Run with:
    uv run python examples/patents/wright_flyer_demo.py

Controls (keyboard, when not headless):
    q/a : increase/decrease thrust
    w/s : pitch up/down (elevator)
    e/d : roll left/right (wing warp)
    r/f : yaw left/right (rudder)
    x   : exit
"""

from __future__ import annotations

import argparse
import logging

from cloud_robotics_sim.patents import PatentSimConfig, create_simulation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> int:
    """Run the Wright Flyer interactive demonstration."""
    parser = argparse.ArgumentParser(description="Wright Flyer Genesis Demo")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the interactive viewer",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Simulation steps to run",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Genesis device (cuda or cpu)",
    )
    args = parser.parse_args()

    config = PatentSimConfig(
        patent_id="US821393",
        headless=args.headless,
        dt=0.01,
        substeps=10,
        resolution=(800, 600),
        device=args.device,
        parameters={
            "thrust": 0.65,
            "wind_speed": 5.0,
        },
    )

    sim = create_simulation("US821393", config=config)
    sim.build()
    sim.reset()

    logger.info("Running Wright Flyer demo for %d steps", args.steps)
    for step in range(args.steps):
        state = sim.step()
        if step % 100 == 0:
            logger.info(
                "Step %d: altitude=%.1fm airspeed=%.1fm/s roll=%.1fdeg pitch=%.1fdeg",
                step,
                state.metrics.get("altitude", 0.0),
                state.metrics.get("airspeed", 0.0),
                state.metrics.get("roll_deg", 0.0),
                state.metrics.get("pitch_deg", 0.0),
            )

    sim.close()
    logger.info("Demo complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
