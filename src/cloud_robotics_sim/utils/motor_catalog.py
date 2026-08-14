"""Robot motor catalog loader and query utilities.

The catalog is backed by ``src/cloud_robotics_sim/data/robot_motors.yaml`` and
provides typical voltage/current/power/torque ranges for common robot motor
applications. Use it to seed simulation parameters (e.g. for
JouleHeatingSolver or thermal analysis).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "robot_motors.yaml"


def _load_catalog() -> dict[str, Any]:
    """Load the motor catalog YAML file."""
    with open(_CATALOG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_catalog() -> dict[str, Any]:
    """Return the full motor catalog dictionary."""
    return _load_catalog()


def list_categories() -> list[str]:
    """Return the list of motor category keys."""
    catalog = _load_catalog()
    return list(catalog.get("categories", {}).keys())


def list_motors(category: str | None = None) -> list[str]:
    """Return motor names, optionally filtered by category.

    Parameters
    ----------
    category : str | None
        Category key such as ``"humanoid_joint"``, ``"cobot_joint"``,
        ``"industrial_servo"``, ``"quadruped"``, ``"drone"``, ``"agv``,
        or ``"dexterous_hand"``. If ``None``, all motors are returned.

    Returns:
    -------
    list[str]
        Motor entry names.
    """
    catalog = _load_catalog()
    categories = catalog.get("categories", {})
    if category is None:
        names: list[str] = []
        for cat in categories.values():
            names.extend(entry["name"] for entry in cat.get("entries", []))
        return names
    if category not in categories:
        raise ValueError(f"Unknown motor category: {category!r}")
    return [entry["name"] for entry in categories[category].get("entries", [])]


def get_motor(name: str) -> dict[str, Any]:
    """Return a single motor entry by name.

    Parameters
    ----------
    name : str
        Motor entry name, e.g. ``"hip/knee_large"`` or
        ``"yaskawa_sgm7g_55apk"``.

    Returns:
    -------
    dict[str, Any]
        The motor entry dictionary.

    Raises:
    ------
    ValueError
        If no motor with the given name exists.
    """
    catalog = _load_catalog()
    for cat in catalog.get("categories", {}).values():
        for entry in cat.get("entries", []):
            if entry.get("name") == name:
                return dict(entry)
    raise ValueError(f"Unknown motor name: {name!r}")


def get_category(category: str) -> dict[str, Any]:
    """Return a full category dictionary.

    Parameters
    ----------
    category : str
        Category key.

    Returns:
    -------
    dict[str, Any]
        The category dictionary including ``description`` and ``entries``.
    """
    catalog = _load_catalog()
    categories = catalog.get("categories", {})
    if category not in categories:
        raise ValueError(f"Unknown motor category: {category!r}")
    return dict(categories[category])


def get_thermal_defaults() -> dict[str, Any]:
    """Return the default thermal material properties section."""
    catalog = _load_catalog()
    return dict(catalog.get("thermal_defaults", {}))


def estimate_joule_power(current_a: float, resistance_ohm: float) -> float:
    """Estimate resistive (Joule) heating power.

    Parameters
    ----------
    current_a : float
        Motor current in amperes.
    resistance_ohm : float
        Winding resistance in ohms.

    Returns:
    -------
    float
        Heating power in watts.
    """
    return float(current_a) ** 2 * float(resistance_ohm)


def estimate_resistance_from_motor(
    name: str,
    operating_power_w: float | None = None,
    operating_current_a: float | None = None,
) -> float:
    """Estimate an equivalent winding resistance from catalog bounds.

    The catalog stores voltage/current/power ranges, not exact resistances.
    This helper returns a rough resistance estimate using
    ``R ≈ P / I²`` when both bounds are available.

    Parameters
    ----------
    name : str
        Motor entry name.
    operating_power_w : float | None
        Specific operating power. If ``None``, the catalog's typical mid-range
        power is used.
    operating_current_a : float | None
        Specific operating current. If ``None``, the catalog's typical
        mid-range current is used.

    Returns:
    -------
    float
        Estimated resistance in ohms.
    """
    entry = get_motor(name)
    power_range = entry.get("power_w", [1.0, 1.0])
    current_range = entry.get("current_a", [1.0, 1.0])

    power = operating_power_w if operating_power_w is not None else _mid(power_range)
    current = (
        operating_current_a if operating_current_a is not None else _mid(current_range)
    )
    if current == 0:
        raise ValueError("Cannot estimate resistance with zero current")
    return power / (current**2)


def _mid(value_range: list[float]) -> float:
    """Return the midpoint of a numeric range."""
    if not value_range:
        return 0.0
    return (float(value_range[0]) + float(value_range[-1])) / 2.0
