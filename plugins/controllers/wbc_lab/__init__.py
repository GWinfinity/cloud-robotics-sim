"""WBC Motion Tracking Controller - Genesis port of WBC-Lab.

Migrated from WBC-Lab (MuJoCo/mjlab) to Genesis physics engine.

Core features:
  - Multi-clip motion library training (NPZ format)
  - Adaptive RSI (Reference State Initialization) sampling
  - Assistive wrench curriculum
  - Unitree G1 humanoid robot support
  - Deploy export (ONNX policy + tracking params YAML)

Paper references:
  - ZEST (arXiv:2602.00401)
  - BeyondMimic (arXiv:2508.08241)
  - SONIC (arXiv:2511.07820)
"""

__version__ = "0.1.0"
__source__ = "WBC-Lab"

from .core.envs.wbc_env import WbcGenesisEnv
from .core.envs.rewards import RewardComputer
from .core.envs.terminations import TerminationChecker
from .core.motion.motion_command import MotionCommand, MotionLoader
from .core.motion.sampling import AdaptiveRsiSampler
from .core.robots.g1_config import G1RobotConfig
from .core.export.tracking_params import TrackingParamsExporter

__all__ = [
    "WbcGenesisEnv",
    "RewardComputer",
    "TerminationChecker",
    "MotionCommand",
    "MotionLoader",
    "AdaptiveRsiSampler",
    "G1RobotConfig",
    "TrackingParamsExporter",
]
