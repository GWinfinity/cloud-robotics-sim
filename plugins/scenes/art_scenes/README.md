# ART Scenes Plugin

This plugin provides artistic and culturally-themed scene presets for Genesis physics simulations, adapted from the ART (Artistic Rendering Toolkit) project's spring-festival scene.

## Overview

The ART Scenes plugin enables creation of richly decorated indoor environments with:
- **Room Generation**: Configurable walls, floors, ceilings, and lighting
- **Furniture Library**: Sofas, tables, TV sets, carpets
- **Cultural Decorations**: Chinese New Year (Spring Festival) themed decorations

## Installation

### Prerequisites

```bash
pip install genesis-world>=0.4.0
pip install trimesh>=3.0.0
```

### Plugin Installation

The plugin is automatically available when genesis-cloud-sim is installed.

```python
from genesis_cloud_sim.plugins.scenes.art_scenes import SpringFestivalScene
```

## Quick Start

### Basic Usage

```python
from genesis_cloud_sim.plugins.scenes.art_scenes import SpringFestivalScene

# Create and build scene
scene = SpringFestivalScene()
genesis_scene = scene.build(headless=False)

# Run simulation
scene.run(steps=1000)
```

### Custom Configuration

```python
from genesis_cloud_sim.plugins.scenes.art_scenes import SpringFestivalScene, RoomConfig

# Custom room size
room_config = RoomConfig(
    width=15.0,
    depth=12.0,
    height=8.0,
    wall_color=(0.8, 0.7, 0.6),  # Beige walls
)

scene = SpringFestivalScene(
    room_config=room_config,
    add_furniture=True,
    add_decorations=True,
)
genesis_scene = scene.build(headless=False)
```

### Using Individual Components

```python
from genesis_cloud_sim.plugins.scenes.art_scenes import (
    SceneBuilder,
    RoomConfig,
    LightingConfig,
)
from genesis_cloud_sim.plugins.scenes.art_scenes.assets.furniture import (
    Sofa, CoffeeTable, TVSet
)

# Create scene builder
builder = SceneBuilder(scene)

# Create room
room_config = RoomConfig(width=20, height=10, depth=20)
builder.create_room(room_config)

# Add lighting
lighting = LightingConfig(
    ambient_intensity=0.5,
    directional_color=(1.0, 0.9, 0.8),  # Warm white
)
builder.setup_lighting(lighting)

# Add furniture
sofa = Sofa(scene)
sofa_parts = sofa.create_l_shaped_sofa(
    main_position=(0, 0, 5),
)
```

## Scene Components

### Room Structure

```python
from genesis_cloud_sim.plugins.scenes.art_scenes import RoomConfig

config = RoomConfig(
    width=20.0,        # Room width
    depth=20.0,        # Room depth
    height=10.0,       # Ceiling height
    wall_color=(0.55, 0.27, 0.07),  # RGB color
    floor_color=(0.9, 0.9, 0.9),
    ceiling_color=(1.0, 1.0, 1.0),
    wall_thickness=0.2,
    with_baseboard=True,
    baseboard_color=(1.0, 1.0, 1.0),
    baseboard_height=0.3,
)
```

### Furniture

#### Sofa

```python
from genesis_cloud_sim.plugins.scenes.art_scenes.assets.furniture import Sofa

sofa = Sofa(scene)
parts = sofa.create_l_shaped_sofa(
    main_position=(0, 0, 7),      # Main sofa position
    chaise_position=(4.5, 0, 5.5), # Chaise lounge position
    single_position=(-6, 0, 5),    # Single seat position
    fabric_color=(0.96, 0.96, 0.86),  # Cream white
    pillow_color=(0.83, 0.69, 0.22),  # Light gold
    add_pillows=True,
)
```

#### Coffee Table

```python
from genesis_cloud_sim.plugins.scenes.art_scenes.assets.furniture import CoffeeTable

table = CoffeeTable(scene)
parts = table.create_table(
    position=(0, 0, 2.5),
    wood_color=(0.55, 0.27, 0.07),  # Wood brown
    width=3.6,
    depth=2.1,
    height=1.0,
    with_shelf=True,
)
```

#### TV Set

```python
from genesis_cloud_sim.plugins.scenes.art_scenes.assets.furniture import TVSet

tv = TVSet(scene)
parts = tv.create_tv_set(
    position=(0, 0, -8),
    cabinet_color=(0.29, 0.22, 0.16),  # Dark brown
    screen_color=(0.07, 0.07, 0.07),   # Black
    gold_color=(1.0, 0.84, 0.0),       # Gold handles
)
```

#### Carpet

```python
from genesis_cloud_sim.plugins.scenes.art_scenes.assets.furniture import Carpet

carpet = Carpet(scene)
entity = carpet.create_carpet(
    position=(0, 0, 4),
    color=(0.82, 0.71, 0.55),  # Light brown
    width=8,
    depth=6,
)
```

### Spring Festival Decorations

#### Lanterns

```python
from genesis_cloud_sim.plugins.scenes.art_scenes.spring_festival import Lantern

lantern = Lantern(scene)

# Single lantern
parts = lantern.create_lantern(
    position=(0, 8, 0),
    scale=1.0,
    red_color=(1.0, 0.0, 0.0),   # Bright red
    gold_color=(1.0, 0.84, 0.0), # Gold
)

# String of lanterns
parts = lantern.create_lantern_string(
    start_position=(-6, 8, 0),
    count=4,
    spacing=4.0,
)
```

#### Fu Character (福字)

```python
from genesis_cloud_sim.plugins.scenes.art_scenes.spring_festival import FuCharacter

fu = FuCharacter(scene)
parts = fu.create_fu(
    position=(0, 5, -9.9),
    size=1.5,
    upside_down=True,  # Traditional: upside down = "Fortune arrives"
)
```

#### Chinese Knot (中国结)

```python
from genesis_cloud_sim.plugins.scenes.art_scenes.spring_festival import ChineseKnot

knot = ChineseKnot(scene)
parts = knot.create_knot(
    position=(0, 3, -9.8),
    scale=1.0,
)
```

## API Reference

### SpringFestivalScene

Main scene class for creating complete Spring Festival living rooms.

```python
class SpringFestivalScene:
    def __init__(
        self,
        room_config: Optional[RoomConfig] = None,
        lighting_config: Optional[LightingConfig] = None,
        add_furniture: bool = True,
        add_decorations: bool = True,
    )
    
    def build(self, headless: bool = False, device: str = "cuda") -> Scene
    def reset(self, seed: Optional[int] = None)
    def step(self)
    def run(self, steps: int = 1000, render_interval: int = 1)
```

### SceneBuilder

Generic scene building utilities.

```python
class SceneBuilder:
    def create_box_from_trimesh(...)
    def create_cylinder_from_trimesh(...)
    def create_sphere_from_trimesh(...)
    def create_torus_from_trimesh(...)
    def create_room(self, config: RoomConfig) -> List[Entity]
    def setup_lighting(self, config: LightingConfig)
```

## Cultural Significance

### Spring Festival (春节)

The Spring Festival scene represents a traditional Chinese New Year decorated living room:

- **Red Lanterns (红灯笼)**: Symbolize prosperity and good fortune
- **Fu Character (福字)**: Written on red paper, represents "blessing" or "good fortune". Traditionally hung upside down because "Fu dao" (福倒, upside down Fu) sounds like "Fu dao" (福到, fortune arrives).
- **Chinese Knot (中国结)**: An ancient decorative art representing good luck and prosperity.
- **Colors**: Red and gold are traditional lucky colors for Chinese New Year.

## Integration with Other Plugins

The ART Scenes plugin works seamlessly with:
- **DreamDojo Plugin**: Use ART scenes as environments for data generation
- **ManiSkill Utils**: Use utils for object manipulation within scenes

Example:

```python
from genesis_cloud_sim.plugins.scenes.art_scenes import SpringFestivalScene
from genesis_cloud_sim.plugins.datasets.dreamdojo import GenesisSimulator

# Create scene
scene = SpringFestivalScene()
scene.build(headless=True)

# Use with DreamDojo simulator
simulator = GenesisSimulator(...)
# ... interact with scene objects
```

## Troubleshooting

### Trimesh Not Found

```bash
pip install trimesh
```

### Textures Not Loading

Ensure you have Pillow installed:
```bash
pip install pillow
```

### Scene Looks Dark

Check your lighting configuration. Increase `ambient_intensity` or add more point lights.

## License

Apache License 2.0 - See LICENSE file for details.

## Acknowledgments

- Original ART project: ART/spring-festival/
- Genesis physics engine: https://github.com/Genesis-Embodied-AI/Genesis
- Trimesh library: https://github.com/mikedh/trimesh
