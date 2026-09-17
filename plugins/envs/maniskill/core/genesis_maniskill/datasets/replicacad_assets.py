"""ReplicaCAD scene dataset checking and on-demand downloading.

The ReplicaCAD scene dataset used by ManiSkill's ``ReplicaCAD_SceneManipulation-v1``
task is mirrored on ModelScope as ``jessy888/ManiSkill_replica_cad_dataset``
(Apache-2.0, a single ~289 MB ``replica_cad_dataset.zip``). It is not shipped
with this repository; this module checks whether the dataset has been
extracted under the dataset root and downloads + extracts it on demand::

    assets/maniskill/
        replica_cad_dataset.zip       # downloaded archive (kept)
        replica_cad_dataset/          # extracted: configs/ stages/ objects/ urdf/ ...
            configs/scenes/*.scene_instance.json
            configs/objects/*.object_config.json
            configs/stages/*.stage_config.json
            stages/*.glb
            objects/*.glb
            objects/convex/*_cv_decomp.glb
            urdf/<name>/<name>.urdf

Manual prefetch::

    python -m genesis_maniskill.datasets.replicacad_assets
    python -m genesis_maniskill.datasets.replicacad_assets --check-only

Environment variables:

- ``CRS_REPLICACAD_ASSETS``: override the dataset root directory.
- ``CRS_REPLICACAD_AUTO_DOWNLOAD``: set to ``0``/``false``/``no``/``off`` to
  disable automatic downloads (a missing dataset then raises
  ``FileNotFoundError``).
"""

from __future__ import annotations

import argparse
import logging
import os
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

MODELSCOPE_DATASET = "jessy888/ManiSkill_replica_cad_dataset"
ARCHIVE_NAME = "replica_cad_dataset.zip"
DATASET_DIR_NAME = "replica_cad_dataset"
DEFAULT_URL = (
    "https://www.modelscope.cn/datasets/"
    f"{MODELSCOPE_DATASET}/resolve/master/{ARCHIVE_NAME}"
)

ENV_ASSET_ROOT = "CRS_REPLICACAD_ASSETS"
ENV_AUTO_DOWNLOAD = "CRS_REPLICACAD_AUTO_DOWNLOAD"

_CHUNK_SIZE = 1 << 20  # 1 MiB
_LOG_EVERY = 64  # log progress every 64 chunks (64 MiB)

# Path segments skipped during extraction (packing junk in the ModelScope
# archive, at any depth).
_SKIP_SEGMENTS = (".git", ".cache", "__MACOSX")


def default_dataset_root() -> Path:
    """Return the dataset root: ``CRS_REPLICACAD_ASSETS`` or ``<repo>/assets/maniskill``."""
    env = os.environ.get(ENV_ASSET_ROOT)
    if env:
        return Path(env)
    return _find_repo_root() / "assets" / "maniskill"


def _find_repo_root() -> Path:
    """Locate the repository root by walking up looking for ``pyproject.toml``."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    # plugins/envs/maniskill/core/genesis_maniskill/datasets/replicacad_assets.py
    # -> repo root is parents[6] when pyproject.toml is absent.
    return here.parents[6]


def dataset_url() -> str:
    """Return the download URL of the ReplicaCAD archive on ModelScope."""
    return DEFAULT_URL


def auto_download_enabled(override: bool | None = None) -> bool:
    """Resolve whether automatic downloads are allowed.

    Priority: explicit ``override`` argument, then the
    ``CRS_REPLICACAD_AUTO_DOWNLOAD`` environment variable (default: enabled).
    """
    if override is not None:
        return override
    env = os.environ.get(ENV_AUTO_DOWNLOAD, "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    return True


def dataset_dir(root: str | Path) -> Path:
    """Return the extracted dataset directory for a given root."""
    return Path(root) / DATASET_DIR_NAME


def dataset_ready(root: str | Path) -> bool:
    """Return True if the extracted dataset looks complete under ``root``."""
    dset = dataset_dir(root)
    scenes = dset / "configs" / "scenes"
    return (
        scenes.is_dir()
        and any(scenes.glob("*.scene_instance.json"))
        and (dset / "stages").is_dir()
        and (dset / "objects").is_dir()
    )


def list_scenes(root: str | Path | None = None) -> list[str]:
    """Return the sorted scene names available in the dataset.

    Args:
        root: Dataset root (default: :func:`default_dataset_root`).

    Raises:
        FileNotFoundError: The dataset is not present under ``root``.
    """
    root_path = Path(root) if root is not None else default_dataset_root()
    ensure_dataset(root=root_path)
    scenes_dir = dataset_dir(root_path) / "configs" / "scenes"
    return sorted(
        p.name[: -len(".scene_instance.json")]
        for p in scenes_dir.glob("*.scene_instance.json")
    )


def ensure_dataset(
    root: str | Path | None = None,
    auto_download: bool | None = None,
) -> Path:
    """Ensure the ReplicaCAD dataset is present under ``root``.

    Args:
        root: Dataset root (default: :func:`default_dataset_root`).
        auto_download: Explicitly allow/deny downloading. When None, the
            ``CRS_REPLICACAD_AUTO_DOWNLOAD`` environment variable decides
            (default: enabled).

    Returns:
        The extracted dataset directory (``<root>/replica_cad_dataset``).

    Raises:
        FileNotFoundError: Dataset missing and downloading is disabled.
        RuntimeError: Download or extraction failed.
    """
    root_path = Path(root) if root is not None else default_dataset_root()
    if dataset_ready(root_path):
        return dataset_dir(root_path)
    if not auto_download_enabled(auto_download):
        raise FileNotFoundError(
            f"ReplicaCAD dataset missing under {root_path}. Run "
            f"'python -m genesis_maniskill.datasets.replicacad_assets' to "
            f"download it, or set {ENV_AUTO_DOWNLOAD}=1 to allow automatic "
            f"downloads."
        )
    download_dataset(root_path)
    if not dataset_ready(root_path):
        raise RuntimeError(
            f"ReplicaCAD dataset still incomplete after extraction: "
            f"{dataset_dir(root_path)}"
        )
    return dataset_dir(root_path)


def download_dataset(root: str | Path, url: str | None = None) -> Path:
    """Download, verify, and extract the ReplicaCAD archive into ``root``.

    The archive is kept next to the extracted directory so a broken extraction
    can be retried without re-downloading.
    """
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    archive_path = root_path / ARCHIVE_NAME

    if not archive_path.is_file():
        src = url or dataset_url()
        logger.info(
            "Downloading ReplicaCAD scene dataset (~289 MB) from %s",
            src,
        )
        _download_file(src, archive_path)
    else:
        logger.info("Archive already present, reusing: %s", archive_path)

    _verify_zip(archive_path)
    _extract_zip(archive_path, root_path)

    if not dataset_ready(root_path):
        raise RuntimeError(
            f"extracted {archive_path} but the dataset directory is "
            f"incomplete: {dataset_dir(root_path)}"
        )
    logger.info("ReplicaCAD dataset ready: %s", dataset_dir(root_path))
    return dataset_dir(root_path)


def _download_file(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` via a temp file (atomic on success)."""
    import tempfile

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".part", dir=str(dest.parent))
    tmp = Path(tmp_path)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            total = int(response.headers.get("Content-Length") or 0)
            received = 0
            with os.fdopen(tmp_fd, "wb") as fh:
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
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
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

    Skips macOS metadata entries (``__MACOSX``/``._*``), the ``.git/`` and
    ``.cache/`` packing junk inside the ModelScope archive, and guards
    against zip-slip paths escaping ``target_dir``.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    target_resolved = target_dir.resolve()
    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            name = info.filename
            parts = name.split("/")
            if (
                any(part in _SKIP_SEGMENTS for part in parts)
                or any(part.startswith("._") for part in parts)
                or name.endswith(".DS_Store")
            ):
                continue
            # Reject symlink entries (external_attr bit 0xA0000000).
            if info.external_attr >> 28 == 0xA:
                raise RuntimeError(f"symlink entry not allowed: {name!r}")
            dest = (target_dir / name).resolve()
            if not dest.is_relative_to(target_resolved):
                raise RuntimeError(f"unsafe zip entry: {name!r}")
            zf.extract(info, target_dir)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: check/download the ReplicaCAD scene dataset."""
    parser = argparse.ArgumentParser(
        prog="python -m genesis_maniskill.datasets.replicacad_assets",
        description=(
            "Check and download the ManiSkill ReplicaCAD scene dataset "
            f"({MODELSCOPE_DATASET}) from ModelScope."
        ),
    )
    parser.add_argument(
        "--root",
        default=None,
        help=f"Dataset root (default: ${ENV_ASSET_ROOT} or <repo>/assets/maniskill)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only report whether the dataset is present; never download",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = Path(args.root) if args.root else default_dataset_root()
    if dataset_ready(root):
        print(f"ReplicaCAD dataset present under {root}")
        return 0
    if args.check_only:
        print(f"ReplicaCAD dataset missing under {root}")
        return 1
    ensure_dataset(root=root)
    print(f"ReplicaCAD dataset present under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHIVE_NAME",
    "DATASET_DIR_NAME",
    "DEFAULT_URL",
    "ENV_ASSET_ROOT",
    "ENV_AUTO_DOWNLOAD",
    "MODELSCOPE_DATASET",
    "auto_download_enabled",
    "dataset_dir",
    "dataset_ready",
    "dataset_url",
    "default_dataset_root",
    "download_dataset",
    "ensure_dataset",
    "list_scenes",
    "main",
]
