"""VR teleoperation bridge plugin for genesis-cloud-sim.

Architecture (see README.md and protocol/v1/README.md):

    headset client --UDP/TCP--> transport -> sync mailbox -> filters
        -> semantic mapping -> clutch retargeting -> safety -> IK/control
"""

__version__ = "0.1.0"

from .core.bridge import BridgeConfig, VRBridge
from .core.mapping import SemanticAction, SemanticMapper
from .core.messages import ControllerState, EventMsg, HandState, PoseMsg
from .core.recorder_hook import TeleopRecorder
from .core.retargeting import ClutchRetargeter
from .core.safety import SafetyConfig, SafetyState, SafetySupervisor

__all__ = [
    "BridgeConfig",
    "VRBridge",
    "ClutchRetargeter",
    "SemanticMapper",
    "SemanticAction",
    "SafetyConfig",
    "SafetyState",
    "SafetySupervisor",
    "TeleopRecorder",
    "PoseMsg",
    "HandState",
    "ControllerState",
    "EventMsg",
]
