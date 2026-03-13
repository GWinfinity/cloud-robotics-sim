"""Tasks for Genesis ManiSkill"""

# Basic tasks
from genesis_maniskill.tasks.pick_place import PickPlaceTask
from genesis_maniskill.tasks.open_drawer import OpenDrawerTask
from genesis_maniskill.tasks.push import PushTask
from genesis_maniskill.tasks.stack import StackTask

# Kitchen tasks
from genesis_maniskill.tasks.prepare_food import PrepareFoodTask
from genesis_maniskill.tasks.cleanup import CleanupTask
from genesis_maniskill.tasks.organize_cabinet import OrganizeCabinetTask

# Tabletop tasks
from genesis_maniskill.tasks.insert import InsertTask
from genesis_maniskill.tasks.sort import SortTask
from genesis_maniskill.tasks.assembly import AssemblyTask

# Mobile manipulation
from genesis_maniskill.tasks.mobile_manipulation import MobileManipulationTask


TASK_REGISTRY = {
    # Basic tasks
    'pick_place': PickPlaceTask,
    'pick_and_place': PickPlaceTask,
    'open_drawer': OpenDrawerTask,
    'push': PushTask,
    'stack': StackTask,
    
    # Kitchen tasks
    'prepare_food': PrepareFoodTask,
    'cleanup': CleanupTask,
    'organize_cabinet': OrganizeCabinetTask,
    
    # Tabletop tasks
    'insert': InsertTask,
    'sort': SortTask,
    'assembly': AssemblyTask,
    
    # Mobile manipulation
    'mobile_manipulation': MobileManipulationTask,
}


def get_task(task_name: str, env, scene, robot):
    """
    Get task by name.
    
    Args:
        task_name: Name of the task
        env: Environment instance
        scene: Genesis scene
        robot: Robot agent
    
    Returns:
        Task instance
    """
    task_class = TASK_REGISTRY.get(task_name.lower())
    
    if task_class is None:
        available = list(TASK_REGISTRY.keys())
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available tasks: {available}"
        )
    
    return task_class(env=env, scene=scene, robot=robot)


def list_available_tasks():
    """List all available tasks."""
    tasks_by_category = {
        'Basic': ['pick_place', 'open_drawer', 'push', 'stack'],
        'Kitchen': ['prepare_food', 'cleanup', 'organize_cabinet'],
        'Tabletop': ['insert', 'sort', 'assembly'],
        'Mobile': ['mobile_manipulation'],
    }
    return tasks_by_category


__all__ = [
    # Basic
    'PickPlaceTask',
    'OpenDrawerTask',
    'PushTask',
    'StackTask',
    # Kitchen
    'PrepareFoodTask',
    'CleanupTask',
    'OrganizeCabinetTask',
    # Tabletop
    'InsertTask',
    'SortTask',
    'AssemblyTask',
    # Mobile
    'MobileManipulationTask',
    # Utils
    'get_task',
    'list_available_tasks',
    'TASK_REGISTRY',
]
