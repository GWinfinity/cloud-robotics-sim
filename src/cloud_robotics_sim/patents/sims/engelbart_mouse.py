"""Interactive Genesis simulation of X-Y Position Indicator (US 3,541,541).

Orthogonal dual-wheel coordinate encoder mouse
"""

from __future__ import annotations

from cloud_robotics_sim.patents.base import (
    PatentSimConfig,
    SimState,
    StubPatentSimulation,
)
from cloud_robotics_sim.patents.registry import register_patent


@register_patent(
    "US3541541",
    {
        "title": "X-Y Position Indicator",
        "inventors": ["Douglas C. Engelbart"],
        "grant_date": "1970-11-17",
        "breakthrough": "Orthogonal dual-wheel coordinate encoder mouse",
    },
)
class XyPositionIndicatorSimulation(StubPatentSimulation):
    """Stub simulation for X-Y Position Indicator (US 3,541,541)."""

    def __init__(self, config: PatentSimConfig | None = None) -> None:
        super().__init__(config)
        self.config.patent_id = self.config.patent_id or "US3541541"
        self._init_parameters(
            {
                "interactive_param": 0.0,
            }
        )

    @property
    def patent_title(self) -> str:
        return "X-Y Position Indicator"

    def reset(self) -> SimState:
        """Reset the stub simulation."""
        return super().reset()

    def step(self) -> SimState:
        """Advance the stub simulation."""
        return super().step()
