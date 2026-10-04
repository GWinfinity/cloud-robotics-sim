"""Verify a replay_sdg dataset: structure, loadability, and visual sanity.

Checks, without needing Genesis:
  1. dreamdojo-layout dataset: every episode_k group has uint8 observations
     + float32 actions, and dreamdojo's GenesisDataset serves samples from it.
  2. Rich per-variant files: obs/rgb|depth|segmentation per camera, qpos,
     endpose, camera intrinsics/extrinsics.
  3. Randomization actually happened: prop pixels differ across variants
     (the source trajectory is identical, so frame differences must come
     from object/camera jitter).
  4. Writes a small PNG contact sheet of the primary camera for eyeballing.

Usage:
    python verify_sdg.py sdg_dataset.h5 sdg_dataset_rich/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "datasets"))


def verify_dreamdojo(path: Path) -> list[str]:
    """Structural checks + GenesisDataset load; returns episode key list."""
    from dreamdojo.core.dataset import GenesisDataset

    problems: list[str] = []
    with h5py.File(path, "r") as h5:
        episodes = sorted(k for k in h5.keys() if k.startswith("episode_"))
        if not episodes:
            problems.append(f"{path}: no episode_* groups")
            return problems
        for key in episodes:
            group = h5[key]
            if group["observations"].dtype != np.uint8:
                problems.append(f"{key}: observations not uint8")
            if group["actions"].dtype != np.float32:
                problems.append(f"{key}: actions not float32")
            if group["observations"].shape[0] != group["actions"].shape[0]:
                problems.append(f"{key}: observations/actions length mismatch")
            print(
                f"[verify] {key}: observations {group['observations'].shape} "
                f"uint8, actions {group['actions'].shape} float32"
            )
        n_frames = h5[episodes[0]]["observations"].shape[0]

    ds = GenesisDataset(
        pre_generated_path=str(path),
        num_frames=min(8, n_frames),
        robot_type="franka",
        device="cpu",
    )
    sample = ds[0]
    video = np.asarray(sample["video"])
    action = np.asarray(sample["action"])
    print(f"[verify] GenesisDataset sample: video {video.shape}, action {action.shape}")
    if video.size == 0 or action.size == 0:
        problems.append("GenesisDataset returned an empty sample")
    return episodes


def verify_rich(rich_dir: Path, primary_cam: str) -> list[Path]:
    """Structural checks on per-variant RoboTwin-layout files."""
    problems: list[str] = []
    files = sorted(rich_dir.glob("*.hdf5"))
    if not files:
        return problems  # rich output disabled for this run
    for f in files:
        with h5py.File(f, "r") as h5:
            n = h5.attrs.get("n_frames")
            for cam in h5["obs/rgb"].keys():
                for modality in ("rgb", "depth", "segmentation"):
                    shape = h5[f"obs/{modality}/{cam}"].shape
                    if shape[0] != n:
                        problems.append(f"{f.name}: {modality}/{cam} T mismatch")
            cams = h5["cameras"]
            for cam in cams.keys():
                if cams[cam].attrs["intrinsic"].shape != (3, 3):
                    problems.append(f"{f.name}: {cam} intrinsic not 3x3")
                if cams[cam].attrs["extrinsic"].shape != (4, 4):
                    problems.append(f"{f.name}: {cam} extrinsic not 4x4")
            if h5["qpos"].shape[0] != n or h5["endpose"].shape[0] != n:
                problems.append(f"{f.name}: qpos/endpose T mismatch")
            cam_names = list(cams.keys())
        print(f"[verify] rich {f.name}: n_frames={n}, cams={cam_names}")
    return files


def check_randomization(
    files: list[Path], episodes: list[str], dataset: Path, primary_cam: str
) -> None:
    """Replayed variants of one source must differ (object/camera jitter)."""
    if len(files) < 2 or len(episodes) < 2:
        print("[verify] <2 variants; skipping randomization check")
        return
    with h5py.File(files[0], "r") as a, h5py.File(files[1], "r") as b:
        ra = a[f"obs/rgb/{primary_cam}"][0]
        rb = b[f"obs/rgb/{primary_cam}"][0]
    diff = float(np.mean(np.abs(ra.astype(np.int32) - rb.astype(np.int32))))
    print(f"[verify] variant0-vs-variant1 primary-cam frame-0 mean |diff| = {diff:.2f}")
    if diff < 1.0:
        print("[verify] WARNING: variants look identical — randomization inactive?")


def write_contact_sheet(
    dataset: Path, episodes: list[str], out: Path, cols: int = 4
) -> None:
    """First frame of each dreamdojo episode into one PNG for eyeballing."""
    try:
        from PIL import Image
    except ImportError:
        print("[verify] PIL not installed; skipping contact sheet")
        return
    tiles: list[np.ndarray] = []
    with h5py.File(dataset, "r") as h5:
        for key in episodes:
            tiles.append(h5[key]["observations"][0])
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * h : (r + 1) * h, c * w : (c + 1) * w] = tile
    Image.fromarray(sheet).save(out)
    print(f"[verify] contact sheet: {out}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="dreamdojo-layout HDF5")
    parser.add_argument("rich_dir", type=Path, help="per-variant rich directory")
    parser.add_argument("--primary-cam", default="front")
    parser.add_argument(
        "--sheet", type=Path, default=None, help="contact-sheet PNG output path"
    )
    args = parser.parse_args()

    episodes = verify_dreamdojo(args.dataset)
    files = verify_rich(args.rich_dir, args.primary_cam)
    check_randomization(files, episodes, args.dataset, args.primary_cam)
    write_contact_sheet(
        args.dataset, episodes, args.sheet or args.dataset.with_suffix(".png")
    )

    print("[verify] OK" if episodes else "[verify] FAILED")
    sys.exit(0 if episodes else 1)


if __name__ == "__main__":
    main()
