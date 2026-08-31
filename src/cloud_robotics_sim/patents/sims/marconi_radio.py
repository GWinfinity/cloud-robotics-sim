"""Interactive Genesis simulation of Transmitting Electrical Signals (US 586,193).

Elevated monopole aerial and earth-grounded spark system
"""

from __future__ import annotations

from cloud_robotics_sim.patents.base import (
    PatentSimConfig,
    SimState,
    StubPatentSimulation,
)
from cloud_robotics_sim.patents.registry import register_patent


@register_patent(
    "US586193",
    {
        "title": "Transmitting Electrical Signals",
        "inventors": ["Guglielmo Marconi"],
        "grant_date": "1897-07-13",
        "breakthrough": "Elevated monopole aerial and earth-grounded spark system",
    },
)
class TransmittingElectricalSignalsSimulation(StubPatentSimulation):
    """Stub simulation for Transmitting Electrical Signals (US 586,193)."""

    def __init__(self, config: PatentSimConfig | None = None) -> None:
        super().__init__(config)
        self.config.patent_id = self.config.patent_id or "US586193"
        self._init_parameters(
            {
                "interactive_param": 0.0,
            }
        )

    @property
    def patent_title(self) -> str:
        return "Transmitting Electrical Signals"

    def reset(self) -> SimState:
        """Reset the stub simulation."""
        return super().reset()

    def step(self) -> SimState:
        """Advance the stub simulation."""
        return super().step()
