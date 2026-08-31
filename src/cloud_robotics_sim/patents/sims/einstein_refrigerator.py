"""Interactive Genesis simulation of Refrigeration (US 1,781,541).

Zero-moving-parts hermetic Dalton partial pressure cooling
"""

from __future__ import annotations

from cloud_robotics_sim.patents.base import (
    PatentSimConfig,
    SimState,
    StubPatentSimulation,
)
from cloud_robotics_sim.patents.registry import register_patent


@register_patent(
    "US1781541",
    {
        "title": "Refrigeration",
        "inventors": ["Albert Einstein", "Leo Szilard"],
        "grant_date": "1930-11-11",
        "breakthrough": "Zero-moving-parts hermetic Dalton partial pressure cooling",
    },
)
class RefrigerationSimulation(StubPatentSimulation):
    """Stub simulation for Refrigeration (US 1,781,541)."""

    def __init__(self, config: PatentSimConfig | None = None) -> None:
        super().__init__(config)
        self.config.patent_id = self.config.patent_id or "US1781541"
        self._init_parameters(
            {
                "interactive_param": 0.0,
            }
        )

    @property
    def patent_title(self) -> str:
        return "Refrigeration"

    def reset(self) -> SimState:
        """Reset the stub simulation."""
        return super().reset()

    def step(self) -> SimState:
        """Advance the stub simulation."""
        return super().step()
