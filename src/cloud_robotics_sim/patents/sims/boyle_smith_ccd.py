"""Interactive Genesis simulation of 3-Phase Charge-Coupled Device (US 3,923,554).

3-phase MOS potential well charge packets
"""

from __future__ import annotations

from cloud_robotics_sim.patents.base import (
    PatentSimConfig,
    SimState,
    StubPatentSimulation,
)
from cloud_robotics_sim.patents.registry import register_patent


@register_patent(
    "US3923554",
    {
        "title": "3-Phase Charge-Coupled Device",
        "inventors": ["Willard Boyle", "George Smith"],
        "grant_date": "1975-12-02",
        "breakthrough": "3-phase MOS potential well charge packets",
    },
)
class ThreePhaseChargeCoupledDeviceSimulation(StubPatentSimulation):
    """Stub simulation for 3-Phase Charge-Coupled Device (US 3,923,554)."""

    def __init__(self, config: PatentSimConfig | None = None) -> None:
        super().__init__(config)
        self.config.patent_id = self.config.patent_id or "US3923554"
        self._init_parameters(
            {
                "interactive_param": 0.0,
            }
        )

    @property
    def patent_title(self) -> str:
        return "3-Phase Charge-Coupled Device"

    def reset(self) -> SimState:
        """Reset the stub simulation."""
        return super().reset()

    def step(self) -> SimState:
        """Advance the stub simulation."""
        return super().step()
