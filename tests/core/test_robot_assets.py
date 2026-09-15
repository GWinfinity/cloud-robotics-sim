"""Tests for robot model asset resolution (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cloud_robotics_sim.core import robot_assets
from cloud_robotics_sim.core.robot_assets import (
    RobotModel,
    descriptions_repo_dir,
    ensure_descriptions_repo,
    resolve_robot_model,
)


@pytest.fixture()
def fake_repo_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point repo_root() at an empty tmp dir so bundled assets are absent."""
    monkeypatch.setattr(robot_assets, "repo_root", lambda: tmp_path)
    return tmp_path


class TestResolveRobotModel:
    """Tests for the fallback chain in resolve_robot_model."""

    def test_explicit_existing_path(self, tmp_path: Path, fake_repo_root: Path) -> None:
        model_file = tmp_path / "custom" / "robot.urdf"
        model_file.parent.mkdir(parents=True)
        model_file.write_text("<robot/>")
        model = resolve_robot_model("franka_panda", str(model_file))
        assert model is not None
        assert model.path == model_file
        assert model.format == "urdf"

    def test_explicit_mjcf_suffix(self, tmp_path: Path, fake_repo_root: Path) -> None:
        model_file = tmp_path / "panda.xml"
        model_file.write_text("<mujoco/>")
        model = resolve_robot_model("franka_panda", str(model_file))
        assert model is not None
        assert model.format == "mjcf"

    def test_explicit_missing_path_falls_through(
        self,
        tmp_path: Path,
        fake_repo_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(robot_assets.ENV_AUTO_DOWNLOAD, "0")
        model = resolve_robot_model("franka_panda", str(tmp_path / "gone.urdf"))
        assert model is None

    def test_bundled_asset_preferred(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bundled = fake_repo_root / "assets_genesis" / "embodiments" / "franka-panda"
        bundled.mkdir(parents=True)
        (bundled / "panda.urdf").write_text("<robot/>")
        monkeypatch.setenv(robot_assets.ENV_AUTO_DOWNLOAD, "0")
        model = resolve_robot_model("franka_panda")
        assert model is not None
        assert model.format == "urdf"
        assert model.path == bundled / "panda.urdf"

    def test_descriptions_repo_used_when_cloned(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(robot_assets.ENV_AUTO_DOWNLOAD, "0")
        repo = descriptions_repo_dir(robot_assets.default_download_root())
        mjcf = repo / "robot_descriptions" / "Arms" / "franka_emika_panda"
        mjcf.mkdir(parents=True)
        (mjcf / "panda.xml").write_text("<mujoco/>")
        model = resolve_robot_model("franka_panda")
        assert model is not None
        assert model.format == "mjcf"
        assert model.path == mjcf / "panda.xml"

    def test_auto_download_disabled_returns_none(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(robot_assets.ENV_AUTO_DOWNLOAD, "0")
        assert resolve_robot_model("franka_panda") is None
        assert resolve_robot_model("ur5") is None

    def test_clone_failure_returns_none(
        self, fake_repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess

        def _fail(*_args: object, **_kwargs: object) -> None:
            raise subprocess.CalledProcessError(1, "git clone", stderr="nope")

        monkeypatch.setattr(robot_assets.subprocess, "run", _fail)
        assert resolve_robot_model("franka_panda") is None

    def test_unknown_robot_returns_none(self, fake_repo_root: Path) -> None:
        assert resolve_robot_model("atlas") is None


class TestEnsureDescriptionsRepo:
    """Tests for the clone-or-reuse logic."""

    def test_existing_repo_returned(self, tmp_path: Path) -> None:
        dest = descriptions_repo_dir(tmp_path)
        dest.mkdir(parents=True)
        assert ensure_descriptions_repo(tmp_path) == dest

    def test_disabled_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(robot_assets.ENV_AUTO_DOWNLOAD, "0")
        with pytest.raises(FileNotFoundError, match="auto-download is disabled"):
            ensure_descriptions_repo(tmp_path)

    def test_clone_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dest = descriptions_repo_dir(tmp_path)

        def _fake_run(cmd: list[str], **kwargs: object) -> object:
            assert cmd[:3] == ["git", "clone", "--depth"]
            assert kwargs["check"] is True
            dest.mkdir(parents=True)
            return object()

        monkeypatch.setattr(robot_assets.subprocess, "run", _fake_run)
        assert ensure_descriptions_repo(tmp_path) == dest

    def test_clone_tries_all_sources(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess

        urls: list[str] = []

        def _failing_run(cmd: list[str], **kwargs: object) -> object:
            urls.append(cmd[4])
            raise subprocess.CalledProcessError(1, cmd, stderr="failed")

        monkeypatch.setattr(robot_assets.subprocess, "run", _failing_run)
        with pytest.raises(FileNotFoundError, match="all sources"):
            ensure_descriptions_repo(tmp_path)
        assert urls == robot_assets.REPO_URLS


class TestRobotModel:
    """Tests for the RobotModel dataclass."""

    def test_fields(self, tmp_path: Path) -> None:
        model = RobotModel(robot="ur5", path=tmp_path / "ur5.urdf", format="urdf")
        assert model.robot == "ur5"
        assert model.format == "urdf"
