"""Interactive Genesis simulation of Wholly Aromatic Polycarbonamide Filaments (US 3,671,542).

Liquid-crystalline aramid polymer chain alignment (Kevlar)
"""

from __future__ import annotations

from cloud_robotics_sim.patents.base import (
    PatentSimConfig,
    SimState,
    StubPatentSimulation,
)
from cloud_robotics_sim.patents.registry import register_patent


@register_patent(
    "US3671542",
    {
        "title": "Wholly Aromatic Polycarbonamide Filaments",
        "inventors": ["Stephanie L. Kwolek"],
        "grant_date": "1972-06-20",
        "breakthrough": "Liquid-crystalline aramid polymer chain alignment (Kevlar)",
    },
)
class WhollyAromaticPolycarbonamideFilamentsSimulation(StubPatentSimulation):
    """Stub simulation for Wholly Aromatic Polycarbonamide Filaments (US 3,671,542)."""

    def __init__(self, config: PatentSimConfig | None = None) -> None:
        super().__init__(config)
        self.config.patent_id = self.config.patent_id or "US3671542"
        self._init_parameters(
            {
                "interactive_param": 0.0,
            }
        )

    @property
    def patent_title(self) -> str:
        return "Wholly Aromatic Polycarbonamide Filaments"

    def reset(self) -> SimState:
        """Reset the stub simulation."""
        return super().reset()

    def step(self) -> SimState:
        """Advance the stub simulation."""
        return super().step()
