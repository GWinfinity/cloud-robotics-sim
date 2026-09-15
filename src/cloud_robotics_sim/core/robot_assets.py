"""Robot model asset resolution and on-demand downloading.

The robot URDF/MJCF models used by the core embodiments (Franka Panda,
UR5) are not shipped with this repository (they are large; some live in
gitignored ``assets_genesis/``). This module resolves a model path with
the following fallback chain:

1. An explicit ``urdf_path``/``mjcf_path`` that exists on disk.
2. Repo-bundled assets under ``assets_genesis/embodiments/`` (present on
   machines that have the gitignored asset checkout).
3. A shallow clone of ``robot-descriptions/awesome-robot-descriptions``
   (auto-downloaded on first use; the AtomGit mirror is tried first and
   GitHub as fallback). The original absolute-path config that pointed at
   a manual ``awesome-robot-descriptions-main`` checkout is superseded
   by this managed location.
4. ``None`` — the caller falls back to the Genesis built-in asset lookup
   and finally to a procedural placeholder (with a loud warning).

Environment variables:

- ``CRS_ROBOT_DESC_DIR``: override the download root (default
  ``<repo>/assets/robot_descriptions``).
- ``CRS_ROBOT_DESC_AUTO_DOWNLOAD``: set to ``0``/``false``/``no`` to
  disable automatic cloning (missing assets then resolve to ``None``).

Manual prefetch::

    python -m cloud_robotics_sim.core.robot_assets
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENV_DESC_DIR = "CRS_ROBOT_DESC_DIR"
ENV_AUTO_DOWNLOAD = "CRS_ROBOT_DESC_AUTO_DOWNLOAD"

#: Clone sources tried in order (AtomGit mirror is reachable where
#: GitHub is not).
REPO_URLS = [
    "https://atomgit.com/gh_mirrors/aw/awesome-robot-descriptions",
    "https://github.com/robot-descriptions/awesome-robot-descriptions",
]
REPO_DIR_NAME = "awesome-robot-descriptions"


@dataclass(frozen=True)
class RobotModel:
    """A resolved robot model asset."""

    robot: str
    path: Path
    format: str  # "mjcf" or "urdf"


#: Model lookup table: robot -> candidates (repo-bundled relative to
#: assets_genesis/embodiments, then robot-descriptions repo-relative).
_MODELS: dict[str, dict[str, Any]] = {
    "franka_panda": {
        "bundled": ["franka-panda/panda.urdf"],
        "descriptions": ["robot_descriptions/Arms/franka_emika_panda/panda.xml"],
        "format": "mjcf",
    },
    "ur5": {
        "bundled": ["ur5-wsg/ur5_wsg_gripper.urdf"],
        "descriptions": ["robot_descriptions/Arms/ur5/ur5.urdf"],
        "format": "urdf",
    },
}


def repo_root() -> Path:
    """Return the repository root (parent of the ``src`` package dir)."""
    return Path(__file__).resolve().parents[3]


def default_download_root() -> Path:
    """Return the robot-descriptions download root."""
    env = os.environ.get(ENV_DESC_DIR)
    if env:
        return Path(env)
    return repo_root() / "assets" / "robot_descriptions"


def _auto_download_enabled() -> bool:
    return os.environ.get(ENV_AUTO_DOWNLOAD, "1").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def descriptions_repo_dir(root: Path | None = None) -> Path:
    """Return the expected clone directory of awesome-robot-descriptions."""
    return (root or default_download_root()) / REPO_DIR_NAME


def ensure_descriptions_repo(root: Path | None = None) -> Path:
    """Shallow-clone awesome-robot-descriptions if not already present.

    Returns:
        Path to the cloned repository.

    Raises:
        FileNotFoundError: If auto-download is disabled or all clone
            sources failed.
    """
    root = root or default_download_root()
    dest = descriptions_repo_dir(root)
    if dest.exists():
        return dest
    if not _auto_download_enabled():
        raise FileNotFoundError(
            f"{dest} is missing and auto-download is disabled "
            f"({ENV_AUTO_DOWNLOAD}={os.environ.get(ENV_AUTO_DOWNLOAD)}); "
            f"prefetch with: python -m cloud_robotics_sim.core.robot_assets"
        )
    root.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in REPO_URLS:
        logger.info("cloning %s -> %s", url, dest)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(dest)],
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
            logger.info("cloned robot descriptions from %s", url)
            return dest
        except (
            subprocess.CalledProcessError,
            OSError,
            subprocess.TimeoutExpired,
        ) as exc:
            err = getattr(exc, "stderr", None) or str(exc)
            logger.warning("clone from %s failed: %s", url, err.strip()[:200])
            errors.append(f"{url}: {err.strip()[:200]}")
    raise FileNotFoundError(
        "failed to clone awesome-robot-descriptions from all sources: "
        + "; ".join(errors)
    )


def resolve_robot_model(
    robot: str,
    explicit_path: str | None = None,
    *,
    download_root: Path | None = None,
) -> RobotModel | None:
    """Resolve a robot model path through the fallback chain.

    Args:
        robot: Robot key (``"franka_panda"`` or ``"ur5"``).
        explicit_path: User-supplied model path; used if it exists.
        download_root: Override the download root (tests).

    Returns:
        A :class:`RobotModel`, or ``None`` if no source is available
        (caller decides the final fallback and must warn loudly).
    """
    spec = _MODELS.get(robot)
    if spec is None:
        logger.warning("no asset mapping for unknown robot %r", robot)
        return None

    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.exists():
            fmt = "mjcf" if path.suffix == ".xml" else "urdf"
            return RobotModel(robot=robot, path=path, format=fmt)
        logger.warning(
            "explicit model path %s does not exist; trying bundled/downloaded assets",
            path,
        )

    bundled_root = repo_root() / "assets_genesis" / "embodiments"
    for rel in spec["bundled"]:
        candidate = bundled_root / rel
        if candidate.exists():
            return RobotModel(robot=robot, path=candidate, format="urdf")

    try:
        repo = ensure_descriptions_repo(download_root)
    except FileNotFoundError as exc:
        logger.warning("robot descriptions unavailable: %s", exc)
        return None
    for rel in spec["descriptions"]:
        candidate = repo / rel
        if candidate.exists():
            return RobotModel(robot=robot, path=candidate, format=spec["format"])
    logger.warning("robot descriptions repo cloned but %s not found in it", robot)
    return None


def main() -> int:
    """Prefetch the robot descriptions repo."""
    logging.basicConfig(level=logging.INFO)
    try:
        repo = ensure_descriptions_repo()
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 1
    print(f"robot descriptions ready at {repo}")
    for robot in _MODELS:
        model = resolve_robot_model(robot)
        status = f"{model.path} ({model.format})" if model else "NOT FOUND"
        print(f"  {robot}: {status}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
