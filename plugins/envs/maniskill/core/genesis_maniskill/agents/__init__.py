"""Robot agents for Genesis ManiSkill

This module provides flexible robot loading from URDF/MJCF formats.

New style (recommended):
    from genesis_maniskill.agents import get_robot
    robot = get_robot(scene, 'franka', num_envs=16)

Or load directly from URDF:
    from genesis_maniskill.agents import RobotLoader
    robot = RobotLoader.from_urdf(scene, 'path/to/robot.urdf', num_envs=16)

Legacy style (still supported):
    from genesis_maniskill.agents import FrankaAgent
    robot = FrankaAgent(scene, num_envs=16)
"""

# Generic robot loader (new recommended way)
from genesis_maniskill.agents.robot_loader import (
    RobotLoader,
    RobotConfig,
    get_robot,
)

# Legacy robot classes (for backward compatibility)
from genesis_maniskill.agents.franka import FrankaAgent
from genesis_maniskill.agents.g1 import G1Agent
from genesis_maniskill.agents.gr1 import GR1Agent
from genesis_maniskill.agents.ur5 import UR5Agent
from genesis_maniskill.agents.kinova import KinovaGen3Agent
from genesis_maniskill.agents.xarm import XArmAgent
from genesis_maniskill.agents.mobile_manipulator import MobileManipulatorAgent


def get_agent(agent_name: str, scene, num_envs: int = 1, **kwargs):
    """
    Get robot agent by name.
    
    This function tries the new generic loader first, then falls back
    to legacy classes for backward compatibility.
    
    Args:
        agent_name: Name of the agent (preset, config path, or URDF path)
        scene: Genesis scene
        num_envs: Number of parallel environments
        **kwargs: Additional arguments
    
    Returns:
        Robot instance (RobotLoader or legacy agent)
    """
    # Try new generic loader first
    try:
        return get_robot(scene, agent_name, num_envs, **kwargs)
    except (FileNotFoundError, ValueError):
        pass
    
    # Fall back to legacy classes
    legacy_registry = {
        'franka': FrankaAgent,
        'panda': FrankaAgent,
        'franka_emika_panda': FrankaAgent,
        'g1': G1Agent,
        'unitree_g1': G1Agent,
        'gr1': GR1Agent,
        'fourier_gr1': GR1Agent,
        'ur5': UR5Agent,
        'universal_robots_ur5': UR5Agent,
        'kinova': KinovaGen3Agent,
        'kinova_gen3': KinovaGen3Agent,
        'gen3': KinovaGen3Agent,
        'xarm': XArmAgent,
        'xarm5': lambda **kw: XArmAgent(**{**kw, 'model': 'xarm5'}),
        'xarm6': lambda **kw: XArmAgent(**{**kw, 'model': 'xarm6'}),
        'xarm7': lambda **kw: XArmAgent(**{**kw, 'model': 'xarm7'}),
        'fetch': lambda **kw: MobileManipulatorAgent(**{**kw, 'preset': 'fetch'}),
        'tiago': lambda **kw: MobileManipulatorAgent(**{**kw, 'preset': 'tiago'}),
        'stretch': lambda **kw: MobileManipulatorAgent(**{**kw, 'preset': 'stretch'}),
        'turtlebot_arm': lambda **kw: MobileManipulatorAgent(**{**kw, 'preset': 'turtlebot_arm'}),
    }
    
    agent_class = legacy_registry.get(agent_name.lower())
    
    if agent_class is None:
        available = list(legacy_registry.keys())
        raise ValueError(
            f"Unknown agent: {agent_name}. "
            f"Provide a config file path, URDF path, or one of: {available}"
        )
    
    return agent_class(scene=scene, num_envs=num_envs, **kwargs)


def list_available_agents():
    """List all available agents by category."""
    return {
        'Manipulators': ['franka', 'ur5', 'kinova', 'xarm5', 'xarm6', 'xarm7'],
        'Humanoids': ['g1', 'gr1'],
        'Mobile Manipulators': ['fetch', 'tiago', 'stretch', 'turtlebot_arm'],
    }


def list_robot_presets():
    """List available robot preset configurations."""
    import os
    preset_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'robots')
    
    if os.path.exists(preset_dir):
        presets = [f.replace('.yaml', '').replace('.json', '') 
                  for f in os.listdir(preset_dir)
                  if f.endswith(('.yaml', '.json'))]
        return sorted(set(presets))
    
    return ['franka', 'ur5', 'kinova_gen3', 'xarm7', 'fetch', 'g1']


def get_agent_info(agent_name: str) -> dict:
    """
    Get information about an agent.
    
    Args:
        agent_name: Name of the agent
    
    Returns:
        Dictionary with agent information
    """
    info = {
        'franka': {
            'name': 'Franka Emika Panda',
            'type': 'manipulator',
            'dof': 7,
            'reach': '0.855m',
            'payload': '3kg',
            'description': 'Collaborative 7-DOF arm',
            'config_path': 'configs/robots/franka.yaml',
        },
        'ur5': {
            'name': 'Universal Robots UR5',
            'type': 'manipulator',
            'dof': 6,
            'reach': '0.850m',
            'payload': '5kg',
            'description': 'Industrial 6-DOF arm',
            'config_path': 'configs/robots/ur5.yaml',
        },
        'kinova': {
            'name': 'Kinova Gen3',
            'type': 'manipulator',
            'dof': 7,
            'reach': '0.902m',
            'payload': '4kg',
            'description': 'Lightweight collaborative arm',
            'config_path': 'configs/robots/kinova_gen3.yaml',
        },
        'xarm7': {
            'name': 'UFACTORY xArm 7',
            'type': 'manipulator',
            'dof': 7,
            'reach': '0.700m',
            'payload': '3.5kg',
            'description': 'Cost-effective collaborative arm',
            'config_path': 'configs/robots/xarm7.yaml',
        },
        'g1': {
            'name': 'Unitree G1',
            'type': 'humanoid',
            'dof': 23,
            'height': '~1.3m',
            'description': 'Humanoid robot with dexterous hands',
            'config_path': 'configs/robots/g1.yaml',
        },
        'gr1': {
            'name': 'Fourier GR-1',
            'type': 'humanoid',
            'dof': 29,
            'height': '~1.65m',
            'description': 'Full-size humanoid robot',
        },
        'fetch': {
            'name': 'Fetch Mobile Manipulator',
            'type': 'mobile_manipulator',
            'base_dof': 3,
            'arm_dof': 7,
            'description': 'Differential drive + 7DOF arm',
            'config_path': 'configs/robots/fetch.yaml',
        },
    }
    
    return info.get(agent_name.lower(), {
        'name': agent_name,
        'description': 'Custom robot (use config file or URDF)'
    })


# New-style exports
__all__ = [
    # New generic loader
    'RobotLoader',
    'RobotConfig',
    'get_robot',
    # Legacy classes (backward compatibility)
    'FrankaAgent',
    'G1Agent',
    'GR1Agent',
    'UR5Agent',
    'KinovaGen3Agent',
    'XArmAgent',
    'MobileManipulatorAgent',
    # Utilities
    'get_agent',
    'list_available_agents',
    'list_robot_presets',
    'get_agent_info',
]
