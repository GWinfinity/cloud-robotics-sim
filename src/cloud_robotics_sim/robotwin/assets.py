"""RoboTwin asset checking and on-demand downloading.

The RoboTwin 2.0 assets (``TianxingChen/RoboTwin2.0`` on Hugging Face) are
not shipped with this repository (~4 GB, see ``.gitignore``). This module
checks whether the required components are present under the asset root and
downloads + extracts them on demand::

    assets/robotwin/
        embodiments.zip                 # downloaded archive (kept)
        embodiments/embodiments/        # extracted: <name>/config.yml + URDFs
        objects.zip
        objects/objects/                # extracted: NNN_class/ object dirs

Usage points (``RoboTwinObjectLibrary``, ``RobotwinEmbodimentConfig.from_yaml``)
call :func:`ensure_for_path` automatically when a missing path points inside
the default asset root. Manual prefetch::

    python -m cloud_robotics_sim.robotwin.assets
    python -m cloud_robotics_sim.robotwin.assets --components objects

Environment variables:

- ``CRS_ROBOTWIN_ASSETS``: override the asset root directory.
- ``CRS_ROBOTWIN_AUTO_DOWNLOAD``: set to ``0``/``false``/``no`` to disable
  automatic downloads (missing assets then raise ``FileNotFoundError``).
- ``HF_ENDPOINT``: Hugging Face endpoint; set to ``https://hf-mirror.com``
  in mainland China.
"""

from __future__ import annotations

import argparse
import logging
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

HF_REPO = "TianxingChen/RoboTwin2.0"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"

ENV_ASSET_ROOT = "CRS_ROBOTWIN_ASSETS"
ENV_AUTO_DOWNLOAD = "CRS_ROBOTWIN_AUTO_DOWNLOAD"

_CHUNK_SIZE = 1 << 20  # 1 MiB
_LOG_EVERY = 64  # log progress every 64 chunks (64 MiB)


@dataclass(frozen=True)
class AssetComponent:
    """One downloadable RoboTwin asset component."""

    name: str
    archive: str  # zip filename in the HF repo
    marker: str  # extracted directory, relative to the asset root
    approx_size: str  # human-readable size hint for messages


COMPONENTS: dict[str, AssetComponent] = {
    "embodiments": AssetComponent(
        name="embodiments",
        archive="embodiments.zip",
        marker="embodiments/embodiments",
        approx_size="~210 MB",
    ),
    "objects": AssetComponent(
        name="objects",
        archive="objects.zip",
        marker="objects/objects",
        approx_size="~3.5 GB",
    ),
}


def default_asset_root() -> Path:
    """Return the asset root: ``CRS_ROBOTWIN_ASSETS`` or ``<repo>/assets/robotwin``."""
    env = os.environ.get(ENV_ASSET_ROOT)
    if env:
        return Path(env)
    # src/cloud_robotics_sim/robotwin/assets.py -> repo root is parents[3].
    return Path(__file__).resolve().parents[3] / "assets" / "robotwin"


def hf_endpoint() -> str:
    """Return the Hugging Face endpoint (``HF_ENDPOINT`` or the default)."""
    return os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT).rstrip("/")


def archive_url(component: AssetComponent, endpoint: str | None = None) -> str:
    """Return the download URL for a component archive."""
    base = (endpoint or hf_endpoint()).rstrip("/")
    return f"{base}/datasets/{HF_REPO}/resolve/main/{component.archive}"


def auto_download_enabled(override: bool | None = None) -> bool:
    """Resolve whether automatic downloads are allowed.

    Priority: explicit ``override`` argument, then the
    ``CRS_ROBOTWIN_AUTO_DOWNLOAD`` environment variable (default: enabled).
    """
    if override is not None:
        return override
    env = os.environ.get(ENV_AUTO_DOWNLOAD, "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    return True


def component_ready(root: Path, name: str) -> bool:
    """Return True if the component's extracted marker directory exists."""
    component = COMPONENTS[name]
    marker = Path(root) / component.marker
    return marker.is_dir() and any(marker.iterdir())


def missing_components(root: Path, components: tuple[str, ...]) -> list[str]:
    """Return the subset of ``components`` whose assets are not extracted."""
    return [name for name in components if not component_ready(root, name)]


def is_under_default_root(path: str | Path) -> bool:
    """Return True if ``path`` points inside the default asset root."""
    try:
        Path(path).resolve().relative_to(default_asset_root().resolve())
        return True
    except ValueError:
        return False


def ensure_for_path(
    component: str,
    path: str | Path,
    auto_download: bool | None = None,
) -> bool:
    """Ensure ``component`` assets when a missing ``path`` needs them.

    Only triggers when ``path`` points inside the default asset root and
    automatic downloads are enabled; arbitrary user paths never trigger a
    download. Returns True if the component is ready afterwards.
    """
    if not is_under_default_root(path):
        return False
    root = default_asset_root()
    if component_ready(root, component):
        return True
    if not auto_download_enabled(auto_download):
        return False
    ensure_robotwin_assets(component, root=root)
    return component_ready(root, component)


def ensure_robotwin_assets(
    *components: str,
    root: str | Path | None = None,
    auto_download: bool | None = None,
) -> Path:
    """Check RoboTwin assets and download the missing components.

    Args:
        components: Component names (``"embodiments"``, ``"objects"``).
            Defaults to all known components when omitted.
        root: Asset root directory (default: :func:`default_asset_root`).
        auto_download: Explicitly allow/deny downloading. When None, the
            ``CRS_ROBOTWIN_AUTO_DOWNLOAD`` environment variable decides
            (default: enabled).

    Returns:
        The asset root path.

    Raises:
        ValueError: Unknown component name.
        FileNotFoundError: Assets missing and downloading is disabled.
        RuntimeError: Download or extraction failed.
    """
    if not components:
        components = tuple(COMPONENTS)
    for name in components:
        if name not in COMPONENTS:
            raise ValueError(
                f"unknown RoboTwin asset component: {name!r} "
                f"(known: {sorted(COMPONENTS)})"
            )
    root_path = Path(root) if root is not None else default_asset_root()

    missing = missing_components(root_path, tuple(components))
    if not missing:
        logger.info("RoboTwin assets ready: %s", root_path)
        return root_path
    if not auto_download_enabled(auto_download):
        raise FileNotFoundError(
            f"RoboTwin assets missing under {root_path}: {', '.join(missing)}. "
            f"Run 'python -m cloud_robotics_sim.robotwin.assets' to download "
            f"them, or set {ENV_AUTO_DOWNLOAD}=1 to allow automatic downloads."
        )
    for name in missing:
        download_component(name, root_path)
    still_missing = missing_components(root_path, tuple(components))
    if still_missing:
        raise RuntimeError(
            f"RoboTwin assets still missing after download: {still_missing}"
        )
    return root_path


def download_component(
    name: str,
    root: str | Path,
    endpoint: str | None = None,
) -> Path:
    """Download, verify, and extract one component archive into ``root``.

    The archive is kept next to the extracted directory (matching the layout
    produced by a manual download) so a broken extraction can be retried
    without re-downloading.
    """
    component = COMPONENTS[name]
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    archive_path = root_path / component.archive

    if not archive_path.is_file():
        url = archive_url(component, endpoint)
        logger.info(
            "Downloading RoboTwin %s (%s) from %s",
            name,
            component.approx_size,
            url,
        )
        _download_file(url, archive_path)
    else:
        logger.info("Archive already present, reusing: %s", archive_path)

    _verify_zip(archive_path)
    _extract_zip(archive_path, root_path / component.name)

    if not component_ready(root_path, name):
        raise RuntimeError(
            f"extracted {archive_path} but marker directory is missing: "
            f"{root_path / component.marker}"
        )
    logger.info("RoboTwin %s ready: %s", name, root_path / component.marker)
    return root_path / component.marker


def _download_file(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` via a ``.part`` file (atomic on success)."""
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            total = int(response.headers.get("Content-Length") or 0)
            received = 0
            with part.open("wb") as fh:
                while True:
                    chunk = response.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    fh.write(chunk)
                    received += len(chunk)
                    if received // _CHUNK_SIZE % _LOG_EVERY == 0:
                        _log_progress(dest, received, total)
        if total and received != total:
            raise RuntimeError(
                f"incomplete download: {dest} ({received}/{total} bytes)"
            )
        part.replace(dest)
    except Exception:
        part.unlink(missing_ok=True)
        raise


def _log_progress(dest: Path, received: int, total: int) -> None:
    """Log download progress in MiB."""
    if total:
        logger.info(
            "  %s: %.0f/%.0f MiB (%.0f%%)",
            dest.name,
            received / (1 << 20),
            total / (1 << 20),
            100.0 * received / total,
        )
    else:
        logger.info("  %s: %.0f MiB", dest.name, received / (1 << 20))


def _verify_zip(archive_path: Path) -> None:
    """Raise RuntimeError if the archive is not a readable, intact zip."""
    if not zipfile.is_zipfile(archive_path):
        raise RuntimeError(f"not a zip archive: {archive_path}")
    with zipfile.ZipFile(archive_path) as zf:
        bad = zf.testzip()
    if bad is not None:
        raise RuntimeError(f"corrupt entry {bad!r} in {archive_path}")


def _extract_zip(archive_path: Path, target_dir: Path) -> None:
    """Extract ``archive_path`` into ``target_dir``.

    Skips macOS metadata entries (``__MACOSX``/``._*``) and guards against
    zip-slip paths escaping ``target_dir``.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    target_resolved = target_dir.resolve()
    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if (
                name.startswith("__MACOSX")
                or "/._" in name
                or name.endswith(".DS_Store")
            ):
                continue
            dest = (target_dir / name).resolve()
            if not str(dest).startswith(str(target_resolved)):
                raise RuntimeError(f"unsafe zip entry: {name!r}")
            zf.extract(info, target_dir)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: check/download RoboTwin assets."""
    parser = argparse.ArgumentParser(
        prog="python -m cloud_robotics_sim.robotwin.assets",
        description="Check and download RoboTwin 2.0 assets from Hugging Face.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=sorted(COMPONENTS),
        default=sorted(COMPONENTS),
        help="Components to ensure (default: all)",
    )
    parser.add_argument(
        "--root",
        default=None,
        help=f"Asset root (default: ${ENV_ASSET_ROOT} or <repo>/assets/robotwin)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only report missing components; never download",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = Path(args.root) if args.root else default_asset_root()
    missing = missing_components(root, tuple(args.components))
    if not missing:
        print(f"All RoboTwin assets present under {root}")
        return 0
    if args.check_only:
        print(f"Missing components under {root}: {', '.join(missing)}")
        return 1
    ensure_robotwin_assets(*args.components, root=root)
    print(f"All RoboTwin assets present under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AssetComponent",
    "COMPONENTS",
    "DEFAULT_HF_ENDPOINT",
    "ENV_ASSET_ROOT",
    "ENV_AUTO_DOWNLOAD",
    "HF_REPO",
    "archive_url",
    "auto_download_enabled",
    "component_ready",
    "default_asset_root",
    "download_component",
    "ensure_for_path",
    "ensure_robotwin_assets",
    "hf_endpoint",
    "is_under_default_root",
    "main",
    "missing_components",
]
