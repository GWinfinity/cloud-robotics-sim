"""Classic patent simulations implemented with Genesis World.

This package ports the interactive demonstrators from classic-patents.com into
Genesis physics. All 22 historical patents are registered and runnable; the
Wright Flyer (US 821,393) is implemented as a fully interactive pilot, while
the remaining patents ship as runnable stubs that can be incrementally
enhanced.

Example:
    >>> from cloud_robotics_sim.patents import create_simulation, list_patents
    >>> print(list_patents())
    >>> sim = create_simulation("US821393", headless=True)
    >>> sim.build()
    >>> sim.reset()
    >>> for _ in range(100):
    ...     sim.step()
    >>> frame = sim.render()
    >>> sim.close()
"""

from __future__ import annotations

# Import all simulation modules so @register_patent decorators execute.
# This must happen after the registry is defined.
from cloud_robotics_sim.patents import sims  # noqa: F401
from cloud_robotics_sim.patents.base import (
    PatentSimConfig,
    PatentSimulation,
    SimState,
    StubPatentSimulation,
)
from cloud_robotics_sim.patents.registry import (
    PatentRegistry,
    create_simulation,
    default_registry,
    list_patents,
    register_patent,
)
from cloud_robotics_sim.patents.runner import run_patent_simulation

__all__ = [
    # Base classes
    "PatentSimConfig",
    "PatentSimulation",
    "SimState",
    "StubPatentSimulation",
    # Registry
    "PatentRegistry",
    "create_simulation",
    "default_registry",
    "list_patents",
    "register_patent",
    # Runner
    "run_patent_simulation",
]
