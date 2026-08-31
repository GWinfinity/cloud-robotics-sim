"""Interactive Genesis simulation of India-Rubber Fabrics (Vulcanization) (US 3,633).

Disulfide polymer cross-linking under heat
"""

from __future__ import annotations

from cloud_robotics_sim.patents.base import (
    PatentSimConfig,
    SimState,
    StubPatentSimulation,
)
from cloud_robotics_sim.patents.registry import register_patent


@register_patent(
    "US3633",
    {
        "title": "India-Rubber Fabrics (Vulcanization)",
        "inventors": ["Charles Goodyear"],
        "grant_date": "1844-06-15",
        "breakthrough": "Disulfide polymer cross-linking under heat",
    },
)
class IndiarubberFabricsVulcanizationSimulation(StubPatentSimulation):
    """Stub simulation for India-Rubber Fabrics (Vulcanization) (US 3,633)."""

    def __init__(self, config: PatentSimConfig | None = None) -> None:
        super().__init__(config)
        self.config.patent_id = self.config.patent_id or "US3633"
        self._init_parameters(
            {
                "interactive_param": 0.0,
            }
        )

    @property
    def patent_title(self) -> str:
        return "India-Rubber Fabrics (Vulcanization)"

    def reset(self) -> SimState:
        """Reset the stub simulation."""
        return super().reset()

    def step(self) -> SimState:
        """Advance the stub simulation."""
        return super().step()
