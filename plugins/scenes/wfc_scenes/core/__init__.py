# SPDX-FileCopyrightText: Copyright (c) 2025 Cloud Robotics Sim
# SPDX-License-Identifier: Apache-2.0

"""Core WFC algorithm and Chinese-home layout rules."""

from .rules_cn import (
    TILES,
    Layout,
    TileSpec,
    build_variants,
    compatible,
    exterior_ok,
    validate_layout,
)
from .wfc import (
    DIRS,
    TileVariant,
    WFCContradictionError,
    collapse,
    expand_variants,
    rotated_sockets,
)

__all__ = [
    "DIRS",
    "TILES",
    "Layout",
    "TileVariant",
    "TileSpec",
    "WFCContradictionError",
    "build_variants",
    "collapse",
    "compatible",
    "expand_variants",
    "exterior_ok",
    "rotated_sockets",
    "validate_layout",
]
