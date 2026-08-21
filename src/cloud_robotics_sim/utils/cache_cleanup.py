"""Genesis simulation cache cleanup utilities.

After a Genesis simulation finishes, temporary/cache data can either be
handed off to a downstream dataset processing pipeline or cleaned up
immediately.  This module provides the switch and the two code paths.

The behavior is controlled by the ``CRS_DATASET_PIPELINE`` environment
variable (or the equivalent ``dataset_pipeline`` config flag):

* ``true`` / ``1`` / ``on`` / ``yes`` -> move the simulation cache into the
  dataset pipeline staging directory for later processing.
* anything else (default) -> delete the simulation cache and release Genesis
  runtime resources.

Example:
    >>> from cloud_robotics_sim.utils.cache_cleanup import cleanup_after_simulation
    >>> cleanup_after_simulation()
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DATASET_PIPELINE_ENV = "CRS_DATASET_PIPELINE"
CACHE_DIR_ENV = "CRS_SIM_CACHE_DIR"
PIPELINE_DIR_ENV = "CRS_DATASET_PIPELINE_DIR"

DEFAULT_CACHE_DIR = "outputs/sim_cache"
DEFAULT_PIPELINE_DIR = "outputs/dataset_pipeline/staging"

_TRUTHY = {"1", "true", "on", "yes"}


def _resolve_path(value: str | Path | None, default: str) -> Path:
    """Resolve a path-like value relative to the current working directory."""
    if value is None or value == "":
        return Path(default).resolve()
    return Path(value).resolve()


def _is_truthy(value: Any) -> bool:
    """Return ``True`` if *value* looks like an enabled boolean flag."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in _TRUTHY


def _lookup_config(config: Any, key: str) -> Any:
    """Look up *key* in a dict-like or dataclass-like config object."""
    if config is None:
        return None
    if isinstance(config, Mapping):
        return config.get(key)
    return getattr(config, key, None)


def is_dataset_pipeline_enabled(config: Mapping[str, Any] | None = None) -> bool:
    """Return whether the downstream dataset pipeline is enabled.

    Resolution order:

    1. ``CRS_DATASET_PIPELINE`` environment variable, if set.
    2. ``dataset_pipeline`` key in *config*, if provided.
    3. Default: ``False``.

    Args:
        config: Optional configuration dictionary that may contain a
            ``dataset_pipeline`` entry.

    Returns:
        ``True`` when the dataset pipeline should receive the cache.
    """
    env_value = os.environ.get(DATASET_PIPELINE_ENV)
    if env_value is not None:
        return _is_truthy(env_value)
    if config is not None:
        return _is_truthy(_lookup_config(config, "dataset_pipeline"))
    return False


def get_cache_dir(config: Mapping[str, Any] | None = None) -> Path:
    """Return the simulation cache directory."""
    env_value = os.environ.get(CACHE_DIR_ENV)
    if env_value is not None:
        return _resolve_path(env_value, DEFAULT_CACHE_DIR)
    if config is not None:
        return _resolve_path(_lookup_config(config, "cache_dir"), DEFAULT_CACHE_DIR)
    return _resolve_path(None, DEFAULT_CACHE_DIR)


def get_pipeline_dir(config: Mapping[str, Any] | None = None) -> Path:
    """Return the dataset pipeline staging directory."""
    env_value = os.environ.get(PIPELINE_DIR_ENV)
    if env_value is not None:
        return _resolve_path(env_value, DEFAULT_PIPELINE_DIR)
    if config is not None:
        return _resolve_path(
            _lookup_config(config, "dataset_pipeline_dir"), DEFAULT_PIPELINE_DIR
        )
    return _resolve_path(None, DEFAULT_PIPELINE_DIR)


def clean_genesis_runtime() -> None:
    """Release Genesis runtime resources if Genesis is initialized.

    This calls ``gs.destroy()`` when available, which tears down scenes,
    releases GPU memory, and forces caching of compiled kernels.  It also
    invokes ``gs.utils.misc.clear_caches()`` to drop module-level asset
    caches (parsed meshes, baked textures, etc.).
    """
    try:
        import genesis as gs
    except Exception as exc:  # pragma: no cover - genesis may be absent
        logger.debug("Genesis not available for runtime cleanup: %s", exc)
        return

    try:
        if getattr(gs, "_initialized", False):
            gs.destroy()
            logger.info("Genesis runtime destroyed")
    except Exception as exc:
        logger.warning("Failed to destroy Genesis runtime: %s", exc)

    try:
        _clear_caches = getattr(gs.utils.misc, "clear_caches", None)
        if _clear_caches is not None:
            _clear_caches()
            logger.debug("Genesis module-level caches cleared")
    except Exception as exc:
        logger.debug("Failed to clear Genesis caches: %s", exc)


def clean_cache_directory(cache_dir: str | Path | None = None) -> Path:
    """Delete the simulation cache directory and recreate an empty one.

    Args:
        cache_dir: Directory to clean.  Defaults to the resolved simulation
            cache directory.

    Returns:
        The path that was cleaned.
    """
    path = _resolve_path(cache_dir, DEFAULT_CACHE_DIR)
    if path.exists():
        shutil.rmtree(path)
        logger.info("Removed simulation cache directory: %s", path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def stage_cache_for_pipeline(
    cache_dir: str | Path | None = None,
    pipeline_dir: str | Path | None = None,
) -> Path:
    """Move the simulation cache into the dataset pipeline staging area.

    The cache is placed under a timestamped subdirectory so multiple runs do
    not collide.

    Args:
        cache_dir: Directory to stage.  Defaults to the resolved simulation
            cache directory.
        pipeline_dir: Dataset pipeline staging directory.  Defaults to the
            resolved pipeline directory.

    Returns:
        The staging directory that received the cache.
    """
    src = _resolve_path(cache_dir, DEFAULT_CACHE_DIR)
    dst_root = _resolve_path(pipeline_dir, DEFAULT_PIPELINE_DIR)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    dst = dst_root / timestamp

    dst_root.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        logger.warning("No simulation cache to stage at %s", src)
        dst.mkdir(parents=True, exist_ok=True)
        return dst

    shutil.move(str(src), str(dst))
    logger.info("Staged simulation cache for pipeline: %s -> %s", src, dst)
    src.mkdir(parents=True, exist_ok=True)
    return dst


def cleanup_after_simulation(
    cache_dir: str | Path | None = None,
    pipeline_dir: str | Path | None = None,
    dataset_pipeline: bool | None = None,
    config: Mapping[str, Any] | None = None,
) -> None:
    """Clean up Genesis simulation cache after a run.

    If *dataset_pipeline* is ``True`` (or inferred from the environment /
    *config*), the cache is moved into the pipeline staging directory.
    Otherwise the cache directory is deleted and the Genesis runtime is
    released.

    Args:
        cache_dir: Override for the simulation cache directory.
        pipeline_dir: Override for the dataset pipeline staging directory.
        dataset_pipeline: Explicit override.  If ``None``, the value is
            inferred from ``CRS_DATASET_PIPELINE`` or *config*.
        config: Optional configuration mapping that may contain
            ``dataset_pipeline``, ``cache_dir``, and
            ``dataset_pipeline_dir`` entries.
    """
    if dataset_pipeline is None:
        dataset_pipeline = is_dataset_pipeline_enabled(config)

    resolved_cache_dir = (
        _resolve_path(cache_dir, DEFAULT_CACHE_DIR)
        if cache_dir is not None
        else get_cache_dir(config)
    )
    resolved_pipeline_dir = (
        _resolve_path(pipeline_dir, DEFAULT_PIPELINE_DIR)
        if pipeline_dir is not None
        else get_pipeline_dir(config)
    )

    if dataset_pipeline:
        stage_cache_for_pipeline(resolved_cache_dir, resolved_pipeline_dir)
    else:
        clean_genesis_runtime()
        clean_cache_directory(resolved_cache_dir)
