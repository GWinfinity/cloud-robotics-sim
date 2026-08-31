"""Interactive Genesis simulation of Semiconductor Device-and-Lead Structure (US 2,981,877).

Monolithic planar silicon IC with aluminum leads
"""

from __future__ import annotations

from cloud_robotics_sim.patents.base import (
    PatentSimConfig,
    SimState,
    StubPatentSimulation,
)
from cloud_robotics_sim.patents.registry import register_patent


@register_patent(
    "US2981877",
    {
        "title": "Semiconductor Device-and-Lead Structure",
        "inventors": ["Robert N. Noyce"],
        "grant_date": "1961-04-25",
        "breakthrough": "Monolithic planar silicon IC with aluminum leads",
    },
)
class SemiconductorDeviceandleadStructureSimulation(StubPatentSimulation):
    """Stub simulation for Semiconductor Device-and-Lead Structure (US 2,981,877)."""

    def __init__(self, config: PatentSimConfig | None = None) -> None:
        super().__init__(config)
        self.config.patent_id = self.config.patent_id or "US2981877"
        self._init_parameters(
            {
                "interactive_param": 0.0,
            }
        )

    @property
    def patent_title(self) -> str:
        return "Semiconductor Device-and-Lead Structure"

    def reset(self) -> SimState:
        """Reset the stub simulation."""
        return super().reset()

    def step(self) -> SimState:
        """Advance the stub simulation."""
        return super().step()
