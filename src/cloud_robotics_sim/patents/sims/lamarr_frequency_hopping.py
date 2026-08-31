"""Interactive Genesis simulation of Secret Communication System (US 2,292,387).

88-frequency piano-roll spread-spectrum carrier hopping
"""

from __future__ import annotations

from cloud_robotics_sim.patents.base import (
    PatentSimConfig,
    SimState,
    StubPatentSimulation,
)
from cloud_robotics_sim.patents.registry import register_patent


@register_patent(
    "US2292387",
    {
        "title": "Secret Communication System",
        "inventors": ["Hedy Lamarr", "George Antheil"],
        "grant_date": "1942-08-11",
        "breakthrough": "88-frequency piano-roll spread-spectrum carrier hopping",
    },
)
class SecretCommunicationSystemSimulation(StubPatentSimulation):
    """Stub simulation for Secret Communication System (US 2,292,387)."""

    def __init__(self, config: PatentSimConfig | None = None) -> None:
        super().__init__(config)
        self.config.patent_id = self.config.patent_id or "US2292387"
        self._init_parameters(
            {
                "interactive_param": 0.0,
            }
        )

    @property
    def patent_title(self) -> str:
        return "Secret Communication System"

    def reset(self) -> SimState:
        """Reset the stub simulation."""
        return super().reset()

    def step(self) -> SimState:
        """Advance the stub simulation."""
        return super().step()
