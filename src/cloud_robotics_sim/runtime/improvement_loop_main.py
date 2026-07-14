"""Compatibility shim for the improvement loop main entry point.

This module re-exports the public API from ``cloud_robotics_sim.runtime.main``
so that existing import paths continue to work after the module prefix was
removed.
"""

from __future__ import annotations

from .main import ImprovementLoop, LoopConfig, make_env_from_config

__all__ = ["ImprovementLoop", "LoopConfig", "make_env_from_config"]
