"""
Genesis ManiSkill: Robot Manipulation Simulation Platform
"""

__version__ = "0.1.0"

# Environments
from .envs.base_env import BaseEnv
from .envs.kitchen_env import KitchenEnv
from .envs.tabletop_env import TableTopEnv

# Robots
from .agents import (
    get_agent,
    list_available_agents,
    get_agent_info,
    FrankaAgent,
    UR5Agent,
    KinovaGen3Agent,
    XArmAgent,
    G1Agent,
    GR1Agent,
    MobileManipulatorAgent,
)

# Tasks
from .tasks import (
    get_task,
    list_available_tasks,
    PickPlaceTask,
    PrepareFoodTask,
    CleanupTask,
    OrganizeCabinetTask,
    InsertTask,
    SortTask,
    AssemblyTask,
    MobileManipulationTask,
)

__all__ = [
    # Version
    "__version__",
    # Environments
    "BaseEnv",
    "KitchenEnv",
    "TableTopEnv",
    # Robot utilities
    "get_agent",
    "list_available_agents",
    "get_agent_info",
    # Robot classes
    "FrankaAgent",
    "UR5Agent",
    "KinovaGen3Agent",
    "XArmAgent",
    "G1Agent",
    "GR1Agent",
    "MobileManipulatorAgent",
    # Task utilities
    "get_task",
    "list_available_tasks",
    # Task classes
    "PickPlaceTask",
    "PrepareFoodTask",
    "CleanupTask",
    "OrganizeCabinetTask",
    "InsertTask",
    "SortTask",
    "AssemblyTask",
    "MobileManipulationTask",
]
