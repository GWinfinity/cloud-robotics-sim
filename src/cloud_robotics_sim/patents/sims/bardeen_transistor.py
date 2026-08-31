"""Interactive Genesis simulation of Three-Electrode Circuit Element Utilizing Semiconductive Materials (US 2,524,035).

Point-contact germanium semiconductor circuit element
"""

from __future__ import annotations

from cloud_robotics_sim.patents.base import (
    PatentSimConfig,
    SimState,
    StubPatentSimulation,
)
from cloud_robotics_sim.patents.registry import register_patent


@register_patent(
    "US2524035",
    {
        "title": "Three-Electrode Circuit Element Utilizing Semiconductive Materials",
        "inventors": ["John Bardeen", "Walter Brattain"],
        "grant_date": "1950-10-03",
        "breakthrough": "Point-contact germanium semiconductor circuit element",
    },
)
class ThreeelectrodeCircuitElementUtilizingSemiconductiveMaterialsSimulation(
    StubPatentSimulation
):
    """Stub simulation for Three-Electrode Circuit Element Utilizing Semiconductive Materials (US 2,524,035)."""

    def __init__(self, config: PatentSimConfig | None = None) -> None:
        super().__init__(config)
        self.config.patent_id = self.config.patent_id or "US2524035"
        self._init_parameters(
            {
                "interactive_param": 0.0,
            }
        )

    @property
    def patent_title(self) -> str:
        return "Three-Electrode Circuit Element Utilizing Semiconductive Materials"

    def reset(self) -> SimState:
        """Reset the stub simulation."""
        return super().reset()

    def step(self) -> SimState:
        """Advance the stub simulation."""
        return super().step()
