# Core API Reference

## EnvironmentComposer

```python
class EnvironmentComposer:
    """Composes robotic environments from modular components."""
    
    def __init__(self, config: ComposerConfig | None = None) -> None
    
    def compose(
        self,
        scene: Scene,
        robot: RobotEmbodiment,
        task: Task,
        spawn_position: tuple[float, float, float] | None = None,
    ) -> ComposedEnvironment
    
    def compose_from_registry(
        self,
        scene_name: str,
        robot_name: str,
        task_name: str,
        scene_kwargs: dict | None = None,
        robot_kwargs: dict | None = None,
        task_kwargs: dict | None = None,
        registry: Any = None,
    ) -> ComposedEnvironment
```

## ComposerConfig

```python
@dataclass
class ComposerConfig:
    dt: float = 0.01                    # Simulation timestep
    substeps: int = 10                  # Physics substeps
    headless: bool = False              # Run without GUI
    resolution: tuple[int, int] = (640, 480)
    num_envs: int = 1                   # Parallel environments
    domain_randomization: dict = field(default_factory=dict)
```

## ComposedEnvironment

```python
class ComposedEnvironment:
    """A fully composed robotic simulation environment."""
    
    def reset(self, seed: int = 0, options: dict | None = None) -> tuple[dict, dict]
    
    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]
    
    def render(self, mode: str = 'rgb_array') -> np.ndarray | None
    
    def close(self) -> None
    
    @property
    def observation_space(self) -> dict
    
    @property
    def action_space(self) -> dict
```

## Scene

```python
class Scene(ABC):
    """Abstract base class for simulation scenes."""
    
    def __init__(self, config: SceneConfig | None = None) -> None
    
    def add_object(self, spawn: ObjectSpawn) -> Scene
    
    def add_objects(self, spawns: list[ObjectSpawn]) -> Scene
    
    def get_objects_by_tag(self, tag: str) -> list[ObjectSpawn]
    
    def build(self, gs_scene: gs.Scene) -> Scene
    
    def get_spawn_positions(self) -> list[tuple[float, float, float]]
    
    def get_bounds(self) -> tuple[float, float, float, float, float, float]
    
    def reset(self) -> None
```

## RobotEmbodiment

```python
class RobotEmbodiment(ABC):
    """Abstract base class for robot embodiments."""
    
    @abstractmethod
    def spawn(self, scene: gs.Scene, position: tuple | None = None) -> RobotEmbodiment
    
    @abstractmethod
    def reset(self) -> None
    
    @abstractmethod
    def apply_action(self, action: np.ndarray) -> None
    
    @abstractmethod
    def get_observation(self) -> dict
    
    @property
    def obs_dim(self) -> int
    
    @property
    def action_dim(self) -> int
    
    @property
    def action_space(self) -> dict
```

## Task

```python
class Task(ABC):
    """Abstract base class for robotic tasks."""
    
    @abstractmethod
    def reset(self, scene: Scene, robot: RobotEmbodiment, seed: int) -> dict
    
    @abstractmethod
    def step(
        self,
        scene: Scene,
        robot: RobotEmbodiment,
        action: np.ndarray,
    ) -> tuple[float, bool, bool, dict]
```

## Registry

```python
class AssetRegistry:
    """Unified registry for all simulation assets."""
    
    def create_scene(self, name: str, **kwargs) -> Any
    
    def create_robot(self, name: str, **kwargs) -> Any
    
    def create_task(self, name: str, **kwargs) -> Any
```

## VectorizedEnvironment

```python
class VectorizedEnvironment:
    """Base class for vectorized environments."""
    
    def reset(self, seeds: list[int] | None = None) -> tuple[np.ndarray, list[dict]]
    
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]
    
    def close(self) -> None
```
