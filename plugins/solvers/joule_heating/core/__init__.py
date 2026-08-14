"""Core implementation of the JouleHeatingSolver plugin."""

from __future__ import annotations

from .joule_heating_solver import JouleHeatingSolver
from .options import JouleHeatingOptions

__all__ = ["JouleHeatingOptions", "JouleHeatingSolver"]
