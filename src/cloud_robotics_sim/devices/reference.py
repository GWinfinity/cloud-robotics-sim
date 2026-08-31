"""MHS-style reference-file generation.

Builds a per-device reference document — the machine-readable equivalent of
the paper manual — from three sources:

1. the device's declared read/write primitives and safety limits;
2. its natural-language notes (characteristics not discernible from code);
3. the standards catalogs under ``data/standards/`` (compliance metadata).

Agents consume this file to learn what a device can measure, what can be
adjusted, and which safety limits are enforced, before operating it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    # <root>/src/cloud_robotics_sim/devices/reference.py
    return Path(__file__).resolve().parents[3]


def load_standards_index(catalog_root: Any = None) -> dict[str, dict[str, Any]]:
    """Load every standards catalog under ``data/standards/`` keyed by id."""
    root = Path(catalog_root) if catalog_root is not None else _repo_root()
    index: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "data" / "standards").glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for entry in data.get("standards", []):
            index[entry["id"]] = entry
    return index


def build_reference_file(device: Any, catalog_root: Any = None) -> dict[str, Any]:
    """Assemble the reference file for a :class:`SimDevice` instance."""
    index = load_standards_index(catalog_root)
    compliance = []
    for std_id in device.compliance:
        entry = index.get(std_id)
        compliance.append(
            {
                "id": std_id,
                "name": entry["name"] if entry else "(not in catalog)",
                "level": entry["level"] if entry else None,
                "status": entry["status"] if entry else None,
                "simulatable_params": entry["simulatable_params"] if entry else [],
            }
        )
    return {
        "device_id": device.device_id,
        "device_type": device.device_type,
        "device_class": device.device_class,
        "mounted_on": device.mounted_on(),
        "readable": [
            {"name": p.name, "unit": p.unit, "description": p.description}
            for p in device.reads.values()
        ],
        "writable": [
            {
                "name": p.name,
                "unit": p.unit,
                "description": p.description,
                "minimum": p.minimum,
                "maximum": p.maximum,
                "choices": list(p.choices) if p.choices else None,
            }
            for p in device.writes.values()
        ],
        "safety_limits": device.safety_limits(),
        "compliance": compliance,
        "natural_language_notes": device.natural_language_notes,
    }


def reference_file_yaml(device: Any, catalog_root: Any = None) -> str:
    """Serialize a device's reference file to YAML."""
    return str(
        yaml.safe_dump(
            build_reference_file(device, catalog_root=catalog_root),
            allow_unicode=True,
            sort_keys=False,
        )
    )
