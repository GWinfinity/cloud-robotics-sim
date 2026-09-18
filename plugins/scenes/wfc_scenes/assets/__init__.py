# SPDX-FileCopyrightText: Copyright (c) 2025 Cloud Robotics Sim
# SPDX-License-Identifier: Apache-2.0

"""Chinese-home furniture primitives and backends."""

from .furniture_cn import (
    Prim,
    PrimBuilder,
    TrimeshBackend,
    build_tile_prims,
    prim_mesh,
)

__all__ = [
    "Prim",
    "PrimBuilder",
    "TrimeshBackend",
    "build_tile_prims",
    "prim_mesh",
]
