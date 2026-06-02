# SPDX-FileCopyrightText: Copyright (c) 2025 Cloud Robotics Sim
# SPDX-License-Identifier: Apache-2.0

"""
Spring Festival (Chinese New Year) Living Room Scene

A complete living room scene with Spring Festival decorations,
adapted from ART's spring-festival genesis_scene.py.

Features:
- Room structure (walls, floor, ceiling)
- Furniture (sofa, coffee table, TV set, carpet)
- Spring Festival decorations (lanterns, Fu character, Chinese knot)
- Proper lighting setup

Usage:
    from genesis_cloud_sim.plugins.scenes.art_scenes.spring_festival import SpringFestivalScene
    
    scene = SpringFestivalScene()
    scene.build()
    
    # Or with custom configuration
    scene = SpringFestivalScene(
        room_config=RoomConfig(width=15, height=8),
        add_decorations=True,
    )
    scene.build(headless=False)

References:
    - Original: ART/spring-festival/genesis_scene.py
"""

from typing import Optional, Dict, Any, List
from pathlib import Path

# Optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

# Genesis imports
try:
    import genesis as gs
    from genesis.engine.scene import Scene
    HAS_GENESIS = True
except ImportError:
    HAS_GENESIS = False
    gs = None
    Scene = None

# Import local modules
try:
    from ..core.scene_builder import SceneBuilder, RoomConfig, LightingConfig
except ImportError:
    from core.scene_builder import SceneBuilder, RoomConfig, LightingConfig

try:
    from ..assets.furniture import Sofa, CoffeeTable, TVSet, Carpet
except ImportError:
    from assets.furniture import Sofa, CoffeeTable, TVSet, Carpet

try:
    from .decorations import SpringFestivalDecorations
except ImportError:
    from decorations import SpringFestivalDecorations


class SpringFestivalScene:
    """
    Complete Spring Festival living room scene.
    
    This class creates a fully furnished living room with Chinese New Year
    decorations, ready for simulation.
    """
    
    def __init__(
        self,
        room_config: Optional[RoomConfig] = None,
        lighting_config: Optional[LightingConfig] = None,
        add_furniture: bool = True,
        add_decorations: bool = True,
        viewer_options: Optional[Dict[str, Any]] = None,
        sim_options: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            room_config: Room configuration. Uses defaults if None.
            lighting_config: Lighting configuration. Uses defaults if None.
            add_furniture: Whether to add furniture (sofa, table, TV, carpet)
            add_decorations: Whether to add Spring Festival decorations
            viewer_options: Genesis viewer options
            sim_options: Genesis simulation options
        """
        self.room_config = room_config or RoomConfig()
        self.lighting_config = lighting_config
        self.add_furniture = add_furniture
        self.add_decorations = add_decorations
        
        # Default viewer options
        if viewer_options is None:
            viewer_options = {
                "camera_pos": (0, 12, 15),
                "camera_lookat": (0, 0, 0),
                "camera_up": (0, 1, 0),
            }
        self.viewer_options = viewer_options
        
        # Default simulation options
        if sim_options is None:
            sim_options = {
                "dt": 0.01,
                "substeps": 10,
            }
        self.sim_options = sim_options
        
        # Scene and builder will be created in build()
        self.scene: Optional[Any] = None
        self.builder: Optional[SceneBuilder] = None
        self.entities: Dict[str, List[Any]] = {}
        
    def build(self, headless: bool = False, device: str = "cuda") -> Any:
        """Build the complete scene.
        
        Args:
            headless: Whether to run without viewer
            device: Device to use ("cuda" or "cpu")
            
        Returns:
            Genesis scene object or None if Genesis not available
        """
        if not HAS_GENESIS:
            print("Warning: Genesis not available. Scene creation failed.")
            return None
        
        # Initialize Genesis
        backend = gs.gpu if device == "cuda" else gs.cpu
        gs.init(backend=backend)
        
        # Create scene
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(**self.viewer_options),
            sim_options=gs.options.SimOptions(**self.sim_options),
            show_viewer=not headless,
        )
        
        # Create builder
        self.builder = SceneBuilder(self.scene)
        
        # Add floor plane
        self.scene.add_entity(
            gs.morphs.Plane(
                pos=(0, 0, 0),
                normal=(0, 1, 0),
            ),
            material=gs.materials.Rigid(friction=0.8),
        )
        
        # Create room structure
        print("Creating room structure...")
        room_entities = self.builder.create_room(self.room_config)
        self.entities['room'] = room_entities
        
        # Setup lighting
        print("Setting up lighting...")
        if self.lighting_config is None:
            # Default Spring Festival lighting
            self.lighting_config = LightingConfig(
                ambient_intensity=0.4,
                directional_intensity=1.0,
                directional_color=(1.0, 0.84, 0.0),  # Warm yellow
                point_lights=[
                    {
                        "color": (1.0, 0.27, 0.0),  # Orange-red for lantern effect
                        "intensity": 0.8,
                        "pos": (np.random.uniform(-10, 10), 5 + np.random.uniform(0, 5), np.random.uniform(-5, 5)) if HAS_NUMPY else (0, 7, 0),
                    }
                    for _ in range(5)
                ] if HAS_NUMPY else [],
            )
        self.builder.setup_lighting(self.lighting_config)
        
        # Add furniture
        if self.add_furniture:
            print("Adding furniture...")
            self._add_furniture()
        
        # Add decorations
        if self.add_decorations:
            print("Adding Spring Festival decorations...")
            self._add_decorations()
        
        # Build the scene
        self.scene.build()
        print("Scene built successfully!")
        
        return self.scene
    
    def _add_furniture(self):
        """Add furniture to the scene."""
        if self.scene is None:
            return
        
        # Sofa
        sofa = Sofa(self.scene)
        sofa_entities = sofa.create_l_shaped_sofa(
            main_position=(0, 0, 7),
            chaise_position=(4.5, 0, 5.5),
            single_position=(-6, 0, 5),
        )
        self.entities['sofa'] = sofa_entities
        
        # Coffee table
        table = CoffeeTable(self.scene)
        table_entities = table.create_table(
            position=(0, 0, 2.5),
        )
        self.entities['table'] = table_entities
        
        # TV set
        tv = TVSet(self.scene)
        tv_entities = tv.create_tv_set(
            position=(0, 0, -8),
        )
        self.entities['tv'] = tv_entities
        
        # Carpet
        carpet = Carpet(self.scene)
        carpet_entity = carpet.create_carpet(
            position=(0, 0, 4),
        )
        self.entities['carpet'] = [carpet_entity] if carpet_entity else []
    
    def _add_decorations(self):
        """Add Spring Festival decorations to the scene."""
        if self.scene is None:
            return
        
        decorations = SpringFestivalDecorations(self.scene)
        
        decoration_entities = decorations.create_full_decorations(
            room_width=self.room_config.width,
            room_depth=self.room_config.depth,
            room_height=self.room_config.height,
        )
        
        self.entities['decorations'] = decoration_entities
    
    def reset(self, seed: Optional[int] = None):
        """Reset the scene.
        
        Args:
            seed: Random seed for reproducibility
        """
        if self.scene is not None:
            if seed is not None and HAS_NUMPY:
                np.random.seed(seed)
            self.scene.reset()
    
    def step(self):
        """Step the simulation."""
        if self.scene is not None:
            self.scene.step()
    
    def run(self, steps: int = 1000, render_interval: int = 1):
        """Run the simulation for a number of steps.
        
        Args:
            steps: Number of simulation steps
            render_interval: Render every N steps
        """
        if self.scene is None:
            print("Warning: Scene not built yet. Call build() first.")
            return
        
        print(f"Running simulation for {steps} steps...")
        for i in range(steps):
            self.step()
            if i % render_interval == 0:
                # Rendering is handled automatically by Genesis viewer
                pass
        print("Simulation complete.")
    
    def export_entities(self) -> Dict[str, List[Any]]:
        """Export all created entities.
        
        Returns:
            Dictionary mapping entity categories to lists of entities
        """
        return self.entities.copy()


def create_spring_festival_scene(
    headless: bool = False,
    device: str = "cuda",
    add_furniture: bool = True,
    add_decorations: bool = True,
    **kwargs
) -> Optional[Any]:
    """Convenience function to create a Spring Festival scene.
    
    Args:
        headless: Whether to run without viewer
        device: Device to use ("cuda" or "cpu")
        add_furniture: Whether to add furniture
        add_decorations: Whether to add decorations
        **kwargs: Additional arguments passed to SpringFestivalScene
        
    Returns:
        Genesis scene object or None
    """
    scene = SpringFestivalScene(
        add_furniture=add_furniture,
        add_decorations=add_decorations,
        **kwargs
    )
    return scene.build(headless=headless, device=device)


# For backward compatibility with ART's original script
def main():
    """Main entry point - replicates ART's genesis_scene.py behavior."""
    scene_obj = create_spring_festival_scene(
        headless=False,
        add_furniture=True,
        add_decorations=True,
    )
    
    if scene_obj is not None:
        # Run simulation (as in original script)
        for i in range(1000):
            scene_obj.step()
    
    return scene_obj


if __name__ == "__main__":
    main()
