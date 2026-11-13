"""Synthetic data generation on genesis-world.

Genesis-side replacement for the reference pipeline's Isaac-Sim render
step: stage an :class:`EpisodeSpec`, randomize the domain, render every
declared camera, and write a resume-safe dataset layout.
"""

from .episodes import EpisodeGenerator, GenerationConfig
from .render import EpisodeResult, GenesisEpisodeRenderer
from .specs import (
    CameraSpec,
    DrSample,
    EpisodeSpec,
    LightSample,
    ObjectPlacement,
    RandomizationSpec,
)
from .writer import DatasetWriter

__all__ = [
    "CameraSpec",
    "DatasetWriter",
    "DrSample",
    "EpisodeGenerator",
    "EpisodeResult",
    "EpisodeSpec",
    "GenerationConfig",
    "GenesisEpisodeRenderer",
    "LightSample",
    "ObjectPlacement",
    "RandomizationSpec",
]
