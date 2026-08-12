"""Unit tests for ``cloud_robotics_sim.robotwin.assets`` (no network needed)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from cloud_robotics_sim.robotwin import assets
from cloud_robotics_sim.robotwin.assets import (
    COMPONENTS,
    archive_url,
    auto_download_enabled,
    component_ready,
    default_asset_root,
    download_component,
    ensure_for_path,
    ensure_robotwin_assets,
    is_under_default_root,
    missing_components,
)


def _make_marker(root: Path, component: str) -> Path:
    """Create a ready marker directory for ``component`` under ``root``."""
    marker = root / COMPONENTS[component].marker
    marker.mkdir(parents=True, exist_ok=True)
    (marker / "placeholder").write_text("x", encoding="utf-8")
    return marker


def _make_archive(root: Path, component: str) -> Path:
    """Create a fake component archive matching the real zip layout."""
    archive = root / COMPONENTS[component].archive
    inner = Path(COMPONENTS[component].marker).name  # e.g. "embodiments"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(f"{inner}/", "")
        zf.writestr(f"{inner}/demo/config.yml", "name: demo\n")
        zf.writestr("__MACOSX/._junk", "junk")  # must be skipped
    return archive


def test_default_asset_root_env_override(monkeypatch, tmp_path):
    """CRS_ROBOTWIN_ASSETS overrides the repo-relative default."""
    monkeypatch.setenv("CRS_ROBOTWIN_ASSETS", str(tmp_path))
    assert default_asset_root() == tmp_path


def test_component_ready_requires_nonempty_dir(tmp_path):
    """An empty or missing marker directory is not ready."""
    assert not component_ready(tmp_path, "objects")
    (tmp_path / "objects" / "objects").mkdir(parents=True)
    assert not component_ready(tmp_path, "objects")
    _make_marker(tmp_path, "objects")
    assert component_ready(tmp_path, "objects")


def test_missing_components(tmp_path):
    """Only not-ready components are reported missing."""
    _make_marker(tmp_path, "embodiments")
    assert missing_components(tmp_path, ("embodiments", "objects")) == ["objects"]


def test_archive_url_endpoint():
    """The URL points at the HF resolve endpoint (mirror-aware)."""
    url = archive_url(COMPONENTS["objects"], endpoint="https://hf-mirror.com")
    assert url == (
        "https://hf-mirror.com/datasets/TianxingChen/RoboTwin2.0"
        "/resolve/main/objects.zip"
    )


def test_auto_download_enabled_precedence(monkeypatch):
    """Explicit override wins; env var disables; default is enabled."""
    monkeypatch.delenv("CRS_ROBOTWIN_AUTO_DOWNLOAD", raising=False)
    assert auto_download_enabled()
    monkeypatch.setenv("CRS_ROBOTWIN_AUTO_DOWNLOAD", "0")
    assert not auto_download_enabled()
    assert auto_download_enabled(True)


def test_ensure_unknown_component(tmp_path):
    """Unknown component names raise ValueError."""
    with pytest.raises(ValueError, match="unknown RoboTwin asset component"):
        ensure_robotwin_assets("textures", root=tmp_path)


def test_ensure_ready_is_noop(tmp_path):
    """Ready components never trigger a download."""
    _make_marker(tmp_path, "objects")
    assert ensure_robotwin_assets("objects", root=tmp_path) == tmp_path


def test_ensure_missing_download_disabled(tmp_path):
    """Missing assets with downloads disabled raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="RoboTwin assets missing"):
        ensure_robotwin_assets("objects", root=tmp_path, auto_download=False)


def test_ensure_missing_downloads(tmp_path, monkeypatch):
    """Missing components are fetched via download_component."""
    calls = []

    def fake_download(name, root, endpoint=None):
        calls.append(name)
        _make_marker(Path(root), name)
        return Path(root) / COMPONENTS[name].marker

    monkeypatch.setattr(assets, "download_component", fake_download)
    ensure_robotwin_assets(root=tmp_path, auto_download=True)
    assert sorted(calls) == ["embodiments", "objects"]


def test_ensure_raises_when_download_leaves_missing(tmp_path, monkeypatch):
    """A download that does not produce the marker raises RuntimeError."""
    monkeypatch.setattr(assets, "download_component", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="still missing"):
        ensure_robotwin_assets("objects", root=tmp_path, auto_download=True)


def test_download_component_extracts_existing_archive(tmp_path):
    """A pre-seeded archive is verified and extracted without downloading."""
    _make_archive(tmp_path, "embodiments")
    marker = download_component("embodiments", tmp_path)
    assert marker == tmp_path / "embodiments" / "embodiments"
    assert (marker / "demo" / "config.yml").is_file()
    assert not (tmp_path / "embodiments" / "__MACOSX").exists()


def test_download_component_rejects_zip_slip(tmp_path):
    """Archives with escaping paths are rejected."""
    archive = tmp_path / "objects.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with pytest.raises(RuntimeError, match="unsafe zip entry"):
        download_component("objects", tmp_path)


def test_download_component_rejects_corrupt_archive(tmp_path):
    """Non-zip archives fail verification."""
    (tmp_path / "objects.zip").write_bytes(b"not a zip")
    with pytest.raises(RuntimeError, match="not a zip archive"):
        download_component("objects", tmp_path)


def test_is_under_default_root(monkeypatch, tmp_path):
    """Paths are classified relative to the configured asset root."""
    monkeypatch.setenv("CRS_ROBOTWIN_ASSETS", str(tmp_path))
    assert is_under_default_root(tmp_path / "objects" / "objects")
    assert not is_under_default_root(Path("/elsewhere/objects"))


def test_ensure_for_path_ignores_foreign_paths(monkeypatch, tmp_path):
    """Arbitrary user paths never trigger a download."""
    monkeypatch.setenv("CRS_ROBOTWIN_ASSETS", str(tmp_path))
    assert not ensure_for_path("objects", tmp_path.parent / "other" / "objects")


def test_ensure_for_path_ready(monkeypatch, tmp_path):
    """Ready components under the default root short-circuit."""
    monkeypatch.setenv("CRS_ROBOTWIN_ASSETS", str(tmp_path))
    _make_marker(tmp_path, "objects")
    assert ensure_for_path("objects", tmp_path / "objects" / "objects")


def test_ensure_for_path_disabled(monkeypatch, tmp_path):
    """Downloads disabled means ensure_for_path returns False."""
    monkeypatch.setenv("CRS_ROBOTWIN_ASSETS", str(tmp_path))
    monkeypatch.setenv("CRS_ROBOTWIN_AUTO_DOWNLOAD", "0")
    assert not ensure_for_path("objects", tmp_path / "objects" / "objects")


def test_main_check_only(monkeypatch, tmp_path, capsys):
    """--check-only reports missing components with exit code 1."""
    monkeypatch.setenv("CRS_ROBOTWIN_ASSETS", str(tmp_path))
    assert assets.main(["--check-only"]) == 1
    assert "Missing components" in capsys.readouterr().out
    _make_marker(tmp_path, "objects")
    _make_marker(tmp_path, "embodiments")
    assert assets.main(["--check-only"]) == 0


def test_object_library_auto_download_hook(monkeypatch, tmp_path):
    """A missing default-root objects dir triggers the download hook."""
    import cloud_robotics_sim.robotwin.object_library as ol

    monkeypatch.setenv("CRS_ROBOTWIN_ASSETS", str(tmp_path))
    objects_dir = tmp_path / "objects" / "objects"

    def fake_ensure(component, path, auto_download=None):
        assert component == "objects"
        objects_dir.mkdir(parents=True)
        (objects_dir / "001_bottle").mkdir()
        return True

    monkeypatch.setattr(ol, "ensure_for_path", fake_ensure)
    lib = ol.RoboTwinObjectLibrary(objects_dir)
    assert lib.objects_dir == objects_dir


def test_object_library_foreign_path_still_raises(tmp_path):
    """Arbitrary missing paths raise without any download attempt."""
    from cloud_robotics_sim.robotwin.object_library import RoboTwinObjectLibrary

    with pytest.raises(FileNotFoundError, match="objects dir not found"):
        RoboTwinObjectLibrary(tmp_path / "nope")


def test_embodiment_config_auto_download_hook(monkeypatch, tmp_path):
    """A missing default-root config.yml triggers the download hook."""
    import cloud_robotics_sim.robotwin.embodiment_config as ec

    monkeypatch.setenv("CRS_ROBOTWIN_ASSETS", str(tmp_path))
    cfg_path = tmp_path / "embodiments" / "embodiments" / "demo" / "config.yml"

    def fake_ensure(component, path, auto_download=None):
        assert component == "embodiments"
        cfg_path.parent.mkdir(parents=True)
        cfg_path.write_text("name: demo\nurdf_path: ./urdf/robot.urdf\n")
        return True

    monkeypatch.setattr(ec, "ensure_for_path", fake_ensure)
    config = ec.RobotwinEmbodimentConfig.from_yaml(cfg_path)
    assert config.name == "demo"
    assert config.urdf_path == "./urdf/robot.urdf"
