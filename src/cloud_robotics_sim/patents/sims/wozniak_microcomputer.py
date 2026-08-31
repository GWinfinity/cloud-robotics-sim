"""Interactive Genesis simulation of Microcomputer for Use with Video Display (US 4,136,359).

Two-phase shared-bus time-multiplexed DRAM
"""

from __future__ import annotations

from cloud_robotics_sim.patents.base import (
    PatentSimConfig,
    SimState,
    StubPatentSimulation,
)
from cloud_robotics_sim.patents.registry import register_patent


@register_patent(
    "US4136359",
    {
        "title": "Microcomputer for Use with Video Display",
        "inventors": ["Steve Wozniak"],
        "grant_date": "1979-01-23",
        "breakthrough": "Two-phase shared-bus time-multiplexed DRAM",
    },
)
class MicrocomputerForUseWithVideoDisplaySimulation(StubPatentSimulation):
    """Stub simulation for Microcomputer for Use with Video Display (US 4,136,359)."""

    def __init__(self, config: PatentSimConfig | None = None) -> None:
        super().__init__(config)
        self.config.patent_id = self.config.patent_id or "US4136359"
        self._init_parameters(
            {
                "interactive_param": 0.0,
            }
        )

    @property
    def patent_title(self) -> str:
        return "Microcomputer for Use with Video Display"

    def reset(self) -> SimState:
        """Reset the stub simulation."""
        return super().reset()

    def step(self) -> SimState:
        """Advance the stub simulation."""
        return super().step()
