"""Tests for the PyTorch deployment installer (tools/install_torch.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.install_torch import (  # noqa: E402
    ALIYUN_PYPI_INDEX,
    ALIYUN_PYTORCH_INDEX,
    OFFICIAL_PYPI_INDEX,
    OFFICIAL_PYTORCH_INDEX,
    build_install_command,
    build_verify_command,
    detect_backend,
    has_musa_device,
    main,
    pypi_index_url,
    pytorch_index_url,
    resolve_backend,
    resolve_mirror,
)


def test_pytorch_index_url_official_cuda():
    """CUDA wheels come from the cuXXX path on the official index."""
    url = pytorch_index_url("cuda", "official", "cu121")
    assert url == f"{OFFICIAL_PYTORCH_INDEX}/cu121"


def test_pytorch_index_url_aliyun_cpu():
    """Aliyun mirrors the PyTorch wheel index under pytorch-wheels."""
    url = pytorch_index_url("cpu", "aliyun", "cu121")
    assert url == f"{ALIYUN_PYTORCH_INDEX}/cpu"


def test_pypi_index_url():
    """The PyPI index switches between official and Aliyun."""
    assert pypi_index_url("official") == OFFICIAL_PYPI_INDEX
    assert pypi_index_url("aliyun") == ALIYUN_PYPI_INDEX


def test_build_install_command_cpu_aliyun():
    """CPU installs use the Aliyun cpu wheel path without torch_musa."""
    cmd = build_install_command("cpu", "aliyun", python="python")
    assert cmd[:4] == ["python", "-m", "pip", "install"]
    assert "torch" in cmd and "torchvision" in cmd
    assert "torch_musa" not in cmd
    assert cmd[cmd.index("--index-url") + 1] == f"{ALIYUN_PYTORCH_INDEX}/cpu"


def test_build_install_command_cuda_official():
    """The CUDA version tag selects the wheel directory."""
    cmd = build_install_command(
        "cuda", "official", cuda_version="cu118", python="python"
    )
    assert cmd[cmd.index("--index-url") + 1] == f"{OFFICIAL_PYTORCH_INDEX}/cu118"


def test_build_install_command_musa_uses_pypi_and_torch_musa():
    """MUSA installs torch_musa from PyPI (not the PyTorch wheel index)."""
    cmd = build_install_command("musa", "aliyun", python="python")
    assert "torch_musa" in cmd
    assert cmd[cmd.index("--index-url") + 1] == ALIYUN_PYPI_INDEX


def test_build_install_command_explicit_index_url_wins():
    """An explicit index URL overrides mirror selection."""
    cmd = build_install_command(
        "cuda", "aliyun", index_url="https://example.com/whl", python="python"
    )
    assert cmd[cmd.index("--index-url") + 1] == "https://example.com/whl"


def test_build_verify_command_musa_imports_torch_musa():
    """The MUSA verification snippet imports torch_musa."""
    cmd = build_verify_command("musa", python="python")
    assert "import torch_musa" in cmd[-1]


def test_has_musa_device_via_gmi():
    """mthreads-gmi on PATH indicates Moore Threads hardware."""
    assert has_musa_device(
        which=lambda name: "/usr/bin/mthreads-gmi" if name == "mthreads-gmi" else None
    )


def test_has_musa_device_via_dev_node():
    """/dev/mtgpu* device nodes indicate Moore Threads hardware."""
    assert has_musa_device(
        which=lambda name: None, path_glob=lambda pat: ["/dev/mtgpu0"]
    )


def test_has_musa_device_absent():
    """No MUSA indicators means no MUSA device."""
    assert not has_musa_device(which=lambda name: None, path_glob=lambda pat: [])


def test_detect_backend_prefers_musa(monkeypatch):
    """MUSA wins over CUDA when both kinds of hardware are visible."""
    import tools.install_torch as mod

    monkeypatch.setattr(mod, "has_musa_device", lambda: True)
    monkeypatch.setattr(mod, "has_cuda_device", lambda: True)
    assert detect_backend() == "musa"


def test_resolve_backend_explicit_arg_wins(monkeypatch):
    """An explicit CLI backend overrides the environment variable."""
    monkeypatch.setenv("CRS_TORCH_BACKEND", "cpu")
    assert resolve_backend("musa") == "musa"


def test_resolve_backend_env_over_detection(monkeypatch):
    """CRS_TORCH_BACKEND overrides auto-detection."""
    monkeypatch.setenv("CRS_TORCH_BACKEND", "musa")
    assert resolve_backend("auto") == "musa"


def test_resolve_mirror_env(monkeypatch):
    """CRS_PIP_MIRROR overrides auto-detection."""
    monkeypatch.setenv("CRS_PIP_MIRROR", "aliyun")
    assert resolve_mirror("auto") == "aliyun"


def test_main_dry_run(monkeypatch, capsys):
    """--dry-run prints the resolved command without executing pip."""
    import tools.install_torch as mod

    # Avoid hardware and network probing.
    monkeypatch.setattr(mod, "detect_backend", lambda: "cpu")
    monkeypatch.setattr(mod, "detect_mirror", lambda: "aliyun")
    monkeypatch.delenv("CRS_TORCH_BACKEND", raising=False)
    monkeypatch.delenv("CRS_PIP_MIRROR", raising=False)
    assert main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "backend=cpu" in out and "mirror=aliyun" in out
    assert f"{ALIYUN_PYTORCH_INDEX}/cpu" in out
