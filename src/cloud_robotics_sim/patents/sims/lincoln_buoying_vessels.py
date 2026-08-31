"""Interactive Genesis simulation of Buoying Vessels Over Shoals (US 6,469).

Synchronized expandable buoyant air chambers
"""

from __future__ import annotations

from cloud_robotics_sim.patents.base import (
    PatentSimConfig,
    SimState,
    StubPatentSimulation,
)
from cloud_robotics_sim.patents.registry import register_patent


@register_patent(
    "US6469",
    {
        "title": "Buoying Vessels Over Shoals",
        "inventors": ["Abraham Lincoln"],
        "grant_date": "1849-05-22",
        "breakthrough": "Synchronized expandable buoyant air chambers",
    },
)
class BuoyingVesselsOverShoalsSimulation(StubPatentSimulation):
    """Stub simulation for Buoying Vessels Over Shoals (US 6,469)."""

    def __init__(self, config: PatentSimConfig | None = None) -> None:
        super().__init__(config)
        self.config.patent_id = self.config.patent_id or "US6469"
        self._init_parameters(
            {
                "interactive_param": 0.0,
            }
        )

    @property
    def patent_title(self) -> str:
        return "Buoying Vessels Over Shoals"

    def reset(self) -> SimState:
        """Reset the stub simulation."""
        return super().reset()

    def step(self) -> SimState:
        """Advance the stub simulation."""
        return super().step()
