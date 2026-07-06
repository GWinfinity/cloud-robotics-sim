"""do-as-i-do reproduction plugin for genesis-cloud-sim.

This plugin provides an end-to-end scaffold of the pipeline described in
"Do as I Do: Dexterous Manipulation Data from Everyday Human Videos":

    video/demo -> reconstruction -> retargeting -> simulation replay -> deployment

Because the original pipeline depends on heavy external modules (SAM3, HaWoR,
TAPIR, MuJoCo Warp) and on UR3e + Sharpa Wave hardware assets, this scaffold:

* uses locally available UR3 arms + Allegro Hands as stand-in hardware;
* keeps the real reconstruction/retargeting modules behind a pluggable interface;
* generates synthetic demonstration data so the full loop can run on CPU.
"""

__version__ = "0.1.0"
__source__ = "https://github.com/GWinfinity/do-as-i-do"
__paper__ = "arXiv:2606.19333"

from .core.data import DemoSequence, HandTrajectory, ObjectTrajectory, RobotTrajectory
from .core.deployment_stub import DeploymentStub
from .core.env import DoAsIDoEnv
from .core.pipeline import DoAsIDoPipeline
from .core.reconstruction_stub import (
    OriginalReconstructionStage,
    ReconstructionStage,
    SyntheticReconstructionStage,
)
from .core.retargeting import IKRetargeter, SamplingRetargeter

__all__ = [
    "DemoSequence",
    "RobotTrajectory",
    "ObjectTrajectory",
    "HandTrajectory",
    "ReconstructionStage",
    "SyntheticReconstructionStage",
    "OriginalReconstructionStage",
    "IKRetargeter",
    "SamplingRetargeter",
    "DoAsIDoEnv",
    "DeploymentStub",
    "DoAsIDoPipeline",
]
