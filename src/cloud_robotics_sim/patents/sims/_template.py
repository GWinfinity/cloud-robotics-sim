"""Template for a new patent simulation.

Copy this file, replace PATENT_ID and the metadata, and implement ``build()``,
``reset()``, and ``step()`` to create a fully functional demonstrator.
"""

from __future__ import annotations

from cloud_robotics_sim.patents.base import PatentSimConfig, PatentSimulation, SimState
from cloud_robotics_sim.patents.registry import register_patent


@register_patent(
    "PATENT_ID",
    {
        "title": "Patent Title",
        "inventors": ["Inventor Name"],
        "grant_date": "YYYY-MM-DD",
        "breakthrough": "One-line description of the breakthrough.",
    },
)
class PatentTitleSimulation(PatentSimulation):
    """Interactive Genesis simulation of PATENT_ID."""

    def __init__(self, config: PatentSimConfig | None = None) -> None:
        super().__init__(config)
        self.config.patent_id = self.config.patent_id or "PATENT_ID"
        self._init_parameters(
            {
                "param1": 0.0,
                "param2": 1.0,
            }
        )

    @property
    def patent_title(self) -> str:
        return "Patent Title"

    def build(self) -> None:
        """Build the Genesis scene for this patent."""
        raise NotImplementedError("build() must be implemented for this simulation.")

    def reset(self) -> SimState:
        """Reset the simulation."""
        self._time = 0.0
        return self.get_state()

    def step(self) -> SimState:
        """Advance the simulation."""
        raise NotImplementedError("step() must be implemented for this simulation.")
