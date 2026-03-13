import genesis as gs
from typing import Any, Dict, List
from engine.constants import ENGINE_MODE, PROJ_DIR
from pathlib import Path
import sys
prompts_root = Path(PROJ_DIR) / 'scripts/prompts'
sys.path.append(prompts_root.as_posix())
from scripts.prompts.sketch_helper import parse_program
from engine.utils.graph_utils import get_root
from scripts.prompts.dsl_utils import set_seed
from pathlib import Path
import numpy as np
from scripts.prompts.helper import *
from scripts.prompts.type_utils import Shape
from scripts.prompts.impl_preset import core
from scripts.prompts.mi_helper import execute_from_preset
import pyquaternion
import re
from transforms3d._gohlketransforms import decompose_matrix

from abc import ABC, abstractmethod


# Adapted from https://github.com/openai/gym/blob/master/gym/core.py
class Env(ABC):
    def reset(self) -> tuple[np.ndarray, dict]:
        """
        Returns:
            observation (object): Observation of the initial state. This will be an element of :attr:`observation_space`
                (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
            info (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: str) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling :meth:`reset` to reset this environment's state.
        Accepts an action and returns either a tuple `(observation, reward, terminated, truncated, info)`.

        Args:
            action (ActType): an action provided by the agent

        Returns:
            observation (object): this will be an element of the environment's :attr:`observation_space`.
                This may, for instance, be a numpy array containing the positions and velocities of certain objects.
            reward (float): The amount of reward returned as a result of taking the action.
            terminated (bool): whether a `terminal state` (as defined under the MDP of the task) is reached.
                In this case further step() calls could return undefined results.
            truncated (bool): whether a truncation condition outside the scope of the MDP is satisfied.
                Typically a timelimit, but could also be used to indicate agent physically going out of bounds.
                Can be used to end the episode prematurely before a `terminal state` is reached.
            info (dictionary): `info` contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                It also can contain information that distinguishes truncation and termination, however this is deprecated in favour
                of returning two booleans, and will be removed in a future version.
        """
        raise NotImplementedError

    @abstractmethod
    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False


def test_genesis():
    gs.init(backend=gs.cpu)
    scene = gs.Scene(gravity=(0, -10, 0))
    cube = scene.add_entity(
        gs.morphs.Box(),
        name='box',
        position=[0, 0, 0],
    )
    scene.build()
    scene.step(n_steps=24)
    print('Genesis simulator test passed!')


PROG_PATH = "resources/assets/jenga.py"

class GenesisEnv(Env):
    def __init__(self):
        super().__init__()
        self.shape_scene: Shape | None = None
        self.shape_leaves: List[Shape] = []
        
        self.gs_scene: gs.Scene | None = None
        self.gs_entities: List[gs.Entity] = []
    
    def reset(self):
        # Initialize Genesis backend
        gs.init(backend=gs.cpu)
        
        library, library_equiv, _ = parse_program(['resources/examples/jenga.py'], roots=None)
        scene_name = get_root(library_equiv)
        with set_seed(0):
            shape = library[scene_name]['__target__']()

        # manually add floor
        floor_x, _, floor_z = compute_shape_center(shape)
        _, floor_y, _ = compute_shape_min(shape)
        shape = concat_shapes(shape, transform_shape(primitive_call('cube', color=(0.2, 0.2, 0.2), shape_kwargs={'scale': (5, 0.1, 5)}), translation_matrix((floor_x, floor_y - .05, floor_z))))

        self.shape_scene = shape
        self.shape_leaves = []
        self.gs_scene = gs.Scene(gravity=(0, -9.81, 0))
        self.gs_entities = []
        
        for eid in range(len(self.shape_scene)):
            if eid in [4, 5]:
                continue  # FIXME HARDCODED
            shape_leaf = self.shape_scene[eid:eid+1]
            rot = shape_leaf[0]['to_world'][:3, :3]
            scale, _, _, translate, _ = decompose_matrix(shape_leaf[0]['to_world'])
            rot = rot @ scale_matrix(1 / scale, (0, 0, 0))[:3, :3]
            
            # Create Genesis entity
            gs_e = self.gs_scene.add_entity(
                gs.morphs.Box(),
                name=f'shape_{eid:03d}',
                position=translate,
                scale=scale,
                rotation=pyquaternion.Quaternion(matrix=rot).yaw_pitch_roll,
                static=eid == len(self.shape_scene) - 1  # Assume the last shape is floor
            )
            self.gs_entities.append(gs_e)
            self.shape_leaves.append(shape_leaf)
        
        self.gs_scene.build()
        self.shape_scene = concat_shapes(*self.shape_leaves)
        return self.shape_scene, {}
    
    def _simulate(self):
        # Run one simulation step
        self.gs_scene.step()

    def _synchronize(self):
        for eid in range(len(self.shape_leaves)):
            shape_leaf = self.shape_leaves[eid]
            gs_e = self.gs_entities[eid]
            
            # Get current transform from Genesis entity
            translate = gs_e.position
            rot = gs_e.rotation_matrix()
            scale = gs_e.scale
            
            # Create transformation matrix
            world_mat = np.eye(4)
            world_mat[:3, :3] = rot @ scale_matrix(scale, (0, 0, 0))[:3, :3]
            world_mat[:3, 3] = translate
            
            # Update shape leaf
            shape_leaf = transform_shape(shape_leaf, world_mat @ np.linalg.inv(shape_leaf[0]['to_world']))
            self.shape_leaves[eid] = shape_leaf
        self.shape_scene = concat_shapes(*self.shape_leaves)

    def step(self, action: str):
        # ignore action for now
        self._simulate()
        self._synchronize()
        return self.shape_scene, 0, False, False, {}

    def render(self):
        # can use `execute_from_preset` to render `self.shape_scene` into pixels
        pass
    
    def close(self):
        # Clean up Genesis resources
        if self.gs_scene is not None:
            # Genesis automatically cleans up when the scene is destroyed
            pass


def main():
    save_dir = Path(PROJ_DIR) / 'logs/simulate_genesis/test'
    save_dir.mkdir(parents=True, exist_ok=True)

    shapes = []

    with GenesisEnv() as env:
        shape, _ = env.reset()
        shapes.append(shape)
        for _ in range(80):
            shape, *_ = env.step(action="")
            shapes.append(shape)

    # if using imports from `scripts.prompts.dsl_utils`, the animation function is not registered for `core` (which imports `dsl_utils` directly)
    sys.path.insert(0, (Path(PROJ_DIR) / 'scripts/prompts').as_posix())
    from dsl_utils import register_animation
    @register_animation()
    def history():
        return shapes
    core([], overwrite=True, save_dir=save_dir.as_posix())


if __name__ == '__main__':
    main()
