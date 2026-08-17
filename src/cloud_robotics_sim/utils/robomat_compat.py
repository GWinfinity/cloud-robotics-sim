"""Compatibility helpers for resolving materials through robomat.

This module isolates the optional robomat dependency. If robomat is not
installed or cannot resolve a material identifier, the helpers return ``None``
and the caller can fall back to simulator defaults.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def resolve_genesis_rigid_material(material_hint: str | None) -> Any | None:
    """Resolve a material hint to a Genesis ``gs.materials.Rigid`` object.

    Args:
        material_hint: A material identifier such as ``"wood"``, ``"fabric"``,
            ``"021_cup"``, or any string accepted by ``robomat.resolve``.
            ``None`` and ``"default"`` are treated as "use simulator default".

    Returns:
        A Genesis ``Rigid`` material instance, or ``None`` if robomat is not
        installed, the hint is empty/default, or resolution fails.
    """
    if not material_hint or material_hint == "default":
        return None

    try:
        import robomat as rm
        from robomat.adapters.genesis import to_genesis
    except ImportError:
        logger.debug("robomat not installed; using default Genesis material")
        return None

    try:
        result = rm.resolve(material_hint)
    except Exception:  # noqa: BLE001 - mapping failure is non-fatal
        logger.debug(
            "robomat could not resolve %r; using default material", material_hint
        )
        return None

    material = result.get("material")
    if material is None:
        return None

    try:
        return to_genesis(material, solver="rigid")
    except Exception as exc:  # noqa: BLE001 - adapter failure is non-fatal
        logger.debug(
            "robomat adapter failed for %r (%s): %s; using default material",
            material_hint,
            material.id,
            exc,
        )
        return None
