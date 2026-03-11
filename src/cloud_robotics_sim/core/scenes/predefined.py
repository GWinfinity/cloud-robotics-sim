"""Predefined scene implementations.

Ready-to-use scene configurations for common environments.
"""

from cloud_robotics_sim.core.scene import ObjectLibrary, Scene, SceneConfig


class EmptyRoom(Scene):
    """An empty room with no furniture.

    Useful for testing robot behaviors without obstacles.
    """

    def __init__(self, size: tuple[float, float, float] = (10.0, 10.0, 3.0)):
        config = SceneConfig(
            name="empty_room",
            size=size,
            default_camera_pos=(5.0, 5.0, 5.0),
        )
        super().__init__(config)

    def _build_custom(self):
        """No custom objects in empty room."""
        pass


class LivingRoom(Scene):
    """A furnished living room.

    Contains typical living room furniture: sofa, coffee table, etc.
    """

    def __init__(self):
        config = SceneConfig(
            name="living_room",
            size=(6.0, 5.0, 3.0),
            default_camera_pos=(3.0, 4.0, 3.0),
        )
        super().__init__(config)

    def _build_custom(self):
        """Add living room furniture."""
        self.add_object(ObjectLibrary.sofa_three_seat(position=(0, -1.5, 0)))
        self.add_object(ObjectLibrary.coffee_table(position=(0.5, 0, 0)))


class Kitchen(Scene):
    """A kitchen environment.

    Contains kitchen appliances and counter space.
    """

    def __init__(self):
        config = SceneConfig(
            name="kitchen",
            size=(4.0, 5.0, 3.0),
            default_camera_pos=(2.0, 4.0, 3.0),
        )
        super().__init__(config)

    def _build_custom(self):
        """Add kitchen appliances."""
        self.add_object(ObjectLibrary.refrigerator(position=(-1.5, -2.0, 0)))


class Office(Scene):
    """An office environment.

    Contains desk and office furniture.
    """

    def __init__(self):
        config = SceneConfig(
            name="office",
            size=(5.0, 4.0, 3.0),
            default_camera_pos=(2.5, 3.0, 3.0),
        )
        super().__init__(config)

    def _build_custom(self):
        """Add office furniture."""
        pass
