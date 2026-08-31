"""Patent simulation implementations.

Importing this module registers all classic patent simulations in the global
``PatentRegistry``. Simulations can be created by ID:

    >>> from cloud_robotics_sim.patents import create_simulation
    >>> sim = create_simulation("US821393", headless=True)
"""

from __future__ import annotations

# Import each simulation module so that @register_patent decorators execute.
from cloud_robotics_sim.patents.sims import (
    bardeen_transistor,
    bell_telephone,
    boyle_smith_ccd,
    edison_lamp,
    einstein_refrigerator,
    engelbart_mouse,
    farnsworth_television,
    fermi_reactor,
    goddard_rocket,
    goodyear_vulcanization,
    howe_sewing_machine,
    kwolek_kevlar,
    lamarr_frequency_hopping,
    lincoln_buoying_vessels,
    marconi_radio,
    morse_telegraph,
    noyce_integrated_circuit,
    spencer_microwave,
    tesla_motor,
    tesla_transformer,
    wozniak_microcomputer,
    wright_flyer,
)

__all__ = [
    "bardeen_transistor",
    "bell_telephone",
    "boyle_smith_ccd",
    "edison_lamp",
    "einstein_refrigerator",
    "engelbart_mouse",
    "farnsworth_television",
    "fermi_reactor",
    "goddard_rocket",
    "goodyear_vulcanization",
    "howe_sewing_machine",
    "kwolek_kevlar",
    "lamarr_frequency_hopping",
    "lincoln_buoying_vessels",
    "marconi_radio",
    "morse_telegraph",
    "noyce_integrated_circuit",
    "spencer_microwave",
    "tesla_motor",
    "tesla_transformer",
    "wozniak_microcomputer",
    "wright_flyer",
]
