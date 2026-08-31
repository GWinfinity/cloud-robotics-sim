"""Simple YAML config loader with dot-access."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class DotDict(dict):
    """Allow attribute-style access to nested dicts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(path: str | Path) -> DotDict:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return DotDict(raw)
