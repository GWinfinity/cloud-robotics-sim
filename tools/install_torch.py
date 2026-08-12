#!/usr/bin/env python3
"""Install the correct PyTorch wheels for the deployment environment.

Two decisions are automated:

1. **Compute backend** — Moore Threads MUSA (``torch_musa``) is preferred when
   MUSA hardware is detected, then NVIDIA CUDA, then CPU. Override with
   ``--backend`` or the ``CRS_TORCH_BACKEND`` environment variable.
2. **Package mirror** — when the official PyTorch index is unreachable
   (typical inside mainland China), Aliyun mirrors are used instead:

   - PyTorch wheels: ``https://mirrors.aliyun.com/pytorch-wheels``
   - PyPI:           ``https://mirrors.aliyun.com/pypi/simple/``

   Override with ``--mirror`` or the ``CRS_PIP_MIRROR`` environment variable.

Usage:
    python tools/install_torch.py                     # auto-detect everything
    python tools/install_torch.py --backend musa      # force MUSA
    python tools/install_torch.py --mirror aliyun     # force Aliyun mirrors
    python tools/install_torch.py --dry-run           # print the pip command
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import urllib.request

OFFICIAL_PYTORCH_INDEX = "https://download.pytorch.org/whl"
ALIYUN_PYTORCH_INDEX = "https://mirrors.aliyun.com/pytorch-wheels"
OFFICIAL_PYPI_INDEX = "https://pypi.org/simple"
ALIYUN_PYPI_INDEX = "https://mirrors.aliyun.com/pypi/simple/"

DEFAULT_CUDA_VERSION = "cu121"
PROBE_TIMEOUT = 5.0

BACKENDS = ("auto", "cpu", "cuda", "musa")
MIRRORS = ("auto", "official", "aliyun")


def has_musa_device(which=shutil.which, path_glob=glob.glob) -> bool:
    """Return True if Moore Threads MUSA hardware appears to be present."""
    if which("mthreads-gmi"):
        return True
    if path_glob("/dev/mtgpu*"):
        return True
    lspci = which("lspci")
    if lspci:
        try:
            out = subprocess.run(
                [lspci],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            ).stdout.lower()
            # Moore Threads PCI vendor ID is 1ed5.
            if "moore threads" in out or "1ed5:" in out:
                return True
        except Exception:
            pass
    try:
        import torch_musa  # noqa: F401

        return True
    except Exception:
        return False


def has_cuda_device(which=shutil.which) -> bool:
    """Return True if an NVIDIA GPU is visible via ``nvidia-smi``."""
    smi = which("nvidia-smi")
    if not smi:
        return False
    try:
        result = subprocess.run(
            [smi, "-L"],
            capture_output=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def detect_backend() -> str:
    """Auto-detect the compute backend: ``musa`` > ``cuda`` > ``cpu``."""
    if has_musa_device():
        return "musa"
    if has_cuda_device():
        return "cuda"
    return "cpu"


def resolve_backend(requested: str = "auto", env_var: str = "CRS_TORCH_BACKEND") -> str:
    """Resolve the backend from CLI arg, environment, then auto-detection."""
    if requested != "auto":
        return requested
    env = os.environ.get(env_var, "").lower()
    if env in ("cpu", "cuda", "musa"):
        return env
    return detect_backend()


def can_reach(url: str, timeout: float = PROBE_TIMEOUT) -> bool:
    """Return True if ``url`` responds within ``timeout`` seconds."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False


def detect_mirror() -> str:
    """Auto-detect the mirror: Aliyun when the official index is unreachable."""
    if can_reach(f"{OFFICIAL_PYTORCH_INDEX}/"):
        return "official"
    return "aliyun"


def resolve_mirror(requested: str = "auto", env_var: str = "CRS_PIP_MIRROR") -> str:
    """Resolve the mirror from CLI arg, environment, then auto-detection."""
    if requested != "auto":
        return requested
    env = os.environ.get(env_var, "").lower()
    if env in ("official", "aliyun"):
        return env
    return detect_mirror()


def pytorch_index_url(backend: str, mirror: str, cuda_version: str) -> str:
    """Return the PyTorch wheel index URL for CUDA/CPU backends."""
    base = OFFICIAL_PYTORCH_INDEX if mirror == "official" else ALIYUN_PYTORCH_INDEX
    suffix = cuda_version if backend == "cuda" else "cpu"
    return f"{base}/{suffix}"


def pypi_index_url(mirror: str) -> str:
    """Return the PyPI index URL (used for the ``torch_musa`` package)."""
    return OFFICIAL_PYPI_INDEX if mirror == "official" else ALIYUN_PYPI_INDEX


def build_install_command(
    backend: str,
    mirror: str,
    cuda_version: str = DEFAULT_CUDA_VERSION,
    index_url: str | None = None,
    python: str = sys.executable,
) -> list[str]:
    """Build the pip command that installs torch for the given backend.

    Args:
        backend: One of ``cpu``, ``cuda``, ``musa``.
        mirror: One of ``official``, ``aliyun``.
        cuda_version: CUDA wheel tag (e.g. ``cu121``), used for ``cuda``.
        index_url: Explicit wheel index override (skips mirror selection).
        python: Python interpreter whose pip should perform the install.
    """
    cmd = [python, "-m", "pip", "install", "--no-cache-dir"]
    if backend == "musa":
        # torch_musa pins a compatible torch version; the resolver picks it.
        # torch_musa ships on PyPI, not on the PyTorch wheel index.
        cmd += ["torch", "torchvision", "torch_musa"]
        cmd += ["--index-url", index_url or pypi_index_url(mirror)]
    else:
        cmd += ["torch", "torchvision"]
        cmd += [
            "--index-url",
            index_url or pytorch_index_url(backend, mirror, cuda_version),
        ]
    return cmd


def build_verify_command(backend: str, python: str = sys.executable) -> list[str]:
    """Build a python command that verifies the installed torch backend."""
    lines = [
        "import torch",
        "print('torch', torch.__version__)",
    ]
    if backend == "cuda":
        lines += ["print('cuda available:', torch.cuda.is_available())"]
    elif backend == "musa":
        lines += [
            "import torch_musa",
            "print('torch_musa', torch_musa.__version__)",
            "print('musa available:', torch.musa.is_available())",
        ]
    return [python, "-c", "; ".join(lines)]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the subprocess exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="auto",
        help="Compute backend (default: auto-detect; env: CRS_TORCH_BACKEND)",
    )
    parser.add_argument(
        "--mirror",
        choices=MIRRORS,
        default="auto",
        help="Package mirror (default: auto-detect; env: CRS_PIP_MIRROR)",
    )
    parser.add_argument(
        "--cuda-version",
        default=DEFAULT_CUDA_VERSION,
        help="CUDA wheel tag for the cuda backend (default: %(default)s)",
    )
    parser.add_argument(
        "--index-url",
        default=os.environ.get("CRS_PYTORCH_INDEX_URL") or None,
        help="Explicit wheel index URL overriding mirror selection "
        "(env: CRS_PYTORCH_INDEX_URL)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pip command without executing it",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the post-install import verification",
    )
    args = parser.parse_args(argv)

    backend = resolve_backend(args.backend)
    mirror = resolve_mirror(args.mirror)
    cmd = build_install_command(
        backend, mirror, cuda_version=args.cuda_version, index_url=args.index_url
    )

    print(f"[install_torch] backend={backend} mirror={mirror}")
    print(f"[install_torch] {' '.join(cmd)}")
    if args.dry_run:
        return 0

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        return result.returncode

    if not args.no_verify:
        verify = build_verify_command(backend)
        result = subprocess.run(verify, check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
