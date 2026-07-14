"""Tests for camera utilities."""

from types import SimpleNamespace

import numpy as np
import pytest

from cloud_robotics_sim.utils import camera
from cloud_robotics_sim.utils.genesis_compat import Pose


class TestPoseConversion:
    """Tests for pose conversion helpers."""

    def test_genesis_pose_to_opencv_extrinsic_identity(self):
        """Converting an identity pose should produce a valid 4x4 matrix."""
        identity = np.eye(4, dtype=np.float32)
        extrinsic = camera.genesis_pose_to_opencv_extrinsic(identity)
        assert extrinsic is not None
        assert extrinsic.shape == (4, 4)


class TestLookAt:
    """Tests for the look_at camera pose helper."""

    def test_look_at_default(self):
        """look_at should return a Pose with the requested eye position."""
        pose = camera.look_at([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
        assert isinstance(pose, Pose)

    def test_look_at_with_up(self):
        """look_at should accept a custom up vector."""
        pose = camera.look_at(
            eye=[2.0, 0.0, 2.0],
            target=[0.0, 0.0, 0.0],
            up=[0.0, 0.0, 1.0],
        )
        assert isinstance(pose, Pose)


class TestColorConversion:
    """Tests for color conversion helpers."""

    def test_hex2rgba_red(self):
        """hex2rgba should convert a red hex string to RGBA."""
        rgba = camera.hex2rgba("#FF0000")
        assert rgba is not None
        assert rgba.shape == (4,)
        assert rgba[0] > rgba[1]
        assert rgba[3] == pytest.approx(1.0)

    def test_hex2rgba_no_correction(self):
        """hex2rgba without gamma correction should return normalized values."""
        rgba = camera.hex2rgba("#808080", correction=False)
        assert rgba is not None
        assert rgba[0] == pytest.approx(128 / 255)

    def test_rgba2hex_normalized(self):
        """rgba2hex should convert normalized RGBA to a hex string."""
        assert camera.rgba2hex([1.0, 0.0, 0.0, 1.0]).lower() == "#ff0000"

    def test_rgba2hex_integer(self):
        """rgba2hex should convert integer RGBA to a hex string."""
        assert camera.rgba2hex([0, 255, 0, 255]).lower() == "#00ff00"


class TestSphericalAndFov:
    """Tests for spherical coordinates and FOV helpers."""

    def test_spherical_to_cartesian_default(self):
        """spherical_to_cartesian with zero angles should place camera on x-axis."""
        pos = camera.spherical_to_cartesian(radius=2.0, azimuth=0.0, elevation=0.0)
        assert pos is not None
        assert pos[0] == pytest.approx(2.0)
        assert pos[1] == pytest.approx(0.0)
        assert pos[2] == pytest.approx(0.0)

    def test_compute_fovy(self):
        """compute_fovy should produce a positive angle in degrees."""
        fovy = camera.compute_fovy(focal_length=35.0, sensor_height=24.0)
        assert fovy > 0.0
        assert fovy < 180.0


class TestCameraRays:
    """Tests for camera ray generation."""

    def test_get_camera_rays_shape(self):
        """get_camera_rays should return arrays of shape (H, W, 3)."""
        pose = np.eye(4, dtype=np.float32)
        intrinsics = np.array(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        origins, directions = camera.get_camera_rays(pose, intrinsics, (480, 640))
        assert origins is not None
        assert directions is not None
        assert origins.shape == (480, 640, 3)
        assert directions.shape == (480, 640, 3)


class TestCreateViewer:
    """Tests for the viewer creation helper."""

    def test_create_viewer_no_genesis(self, monkeypatch):
        """If Genesis is unavailable, create_viewer should return None."""
        monkeypatch.setattr(camera, "HAS_GENESIS", False)
        assert camera.create_viewer(None) is None

    def test_create_viewer_render_system_1_0(self, monkeypatch):
        """create_viewer should build a viewer for render system 1.0."""

        class FakeViewer:
            def __init__(self, resolutions):
                self.resolutions = resolutions

        class FakeRender:
            def set_viewer_shader_dir(self, _pack):
                pass

        class FakeGs:
            Viewer = FakeViewer
            render = FakeRender()

        monkeypatch.setattr(camera, "HAS_GENESIS", True)
        monkeypatch.setattr(camera, "GENESIS_RENDER_SYSTEM", "1.0")
        monkeypatch.setattr(camera, "gs", FakeGs())

        class ShaderConfig:
            shader_pack = "default"
            shader_pack_config = {}

        class ViewerCameraConfig:
            width = 640
            height = 480
            shader_config = ShaderConfig()

        viewer = camera.create_viewer(ViewerCameraConfig())
        assert viewer is not None
        assert viewer.resolutions == (640, 480)

    def test_create_viewer_render_system_1_1(self, monkeypatch):
        """create_viewer should build a viewer for render system 1.1."""

        class FakeShaderPack:
            pass

        class FakeRender:
            def get_shader_pack(self, pack):
                return FakeShaderPack()

        class FakeViewer:
            def __init__(self, resolutions, shader_pack):
                self.resolutions = resolutions
                self.shader_pack = shader_pack

        class FakeGs:
            Viewer = FakeViewer
            render = FakeRender()

        monkeypatch.setattr(camera, "HAS_GENESIS", True)
        monkeypatch.setattr(camera, "GENESIS_RENDER_SYSTEM", "1.1")
        monkeypatch.setattr(camera, "gs", FakeGs())

        class ShaderConfig:
            shader_pack = "rt"

        class ViewerCameraConfig:
            width = 640
            height = 480
            shader_config = ShaderConfig()

        viewer = camera.create_viewer(ViewerCameraConfig())
        assert viewer is not None
        assert viewer.resolutions == (640, 480)
        assert isinstance(viewer.shader_pack, FakeShaderPack)

    def test_create_viewer_render_system_unknown(self, monkeypatch):
        """create_viewer should return None for an unsupported render system."""
        monkeypatch.setattr(camera, "HAS_GENESIS", True)
        monkeypatch.setattr(camera, "GENESIS_RENDER_SYSTEM", "2.0")
        assert camera.create_viewer(None) is None

    def test_create_viewer_macos_scaling(self, monkeypatch):
        """create_viewer should apply macOS content scaling when available."""
        import sys

        class FakeWindow:
            def set_content_scale(self, scale):
                self.scale = scale

        class FakeViewer:
            def __init__(self, resolutions):
                self.resolutions = resolutions
                self.window = FakeWindow()

        class FakeGs:
            Viewer = FakeViewer

        monkeypatch.setattr(camera, "HAS_GENESIS", True)
        monkeypatch.setattr(camera, "GENESIS_RENDER_SYSTEM", "1.0")
        monkeypatch.setattr(camera, "gs", FakeGs())
        monkeypatch.setattr(sys, "platform", "darwin")

        class ViewerCameraConfig:
            width = 640
            height = 480
            shader_config = SimpleNamespace(shader_pack="default")

        viewer = camera.create_viewer(ViewerCameraConfig())
        assert viewer is not None
        assert viewer.window.scale == 1


class TestLookAtFallbacks:
    """Tests for look_at fallback behaviour."""

    def test_look_at_without_torch(self, monkeypatch):
        """look_at should return a Pose when torch is unavailable."""
        monkeypatch.setattr(camera, "HAS_TORCH", False)
        pose = camera.look_at([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
        assert isinstance(pose, Pose)


class TestColorConversionFallbacks:
    """Tests for color conversion fallback paths."""

    def test_hex2rgba_invalid_hex(self):
        """hex2rgba should raise for invalid hex input."""
        with pytest.raises(ValueError):
            camera.hex2rgba("not-a-color")

    def test_hex2rgba_too_short(self):
        """hex2rgba should raise when the hex string is too short."""
        with pytest.raises((ValueError, IndexError)):
            camera.hex2rgba("#FF00")

    def test_rgba2hex_without_numpy(self, monkeypatch):
        """rgba2hex should convert without using numpy."""
        monkeypatch.setattr(camera, "HAS_NUMPY", False)
        assert camera.rgba2hex([1.0, 0.0, 0.0, 1.0]).lower() == "#ff0000"

    def test_rgba2hex_without_numpy_integer(self, monkeypatch):
        """rgba2hex should convert integer values without numpy."""
        monkeypatch.setattr(camera, "HAS_NUMPY", False)
        assert camera.rgba2hex([0, 255, 0, 255]).lower() == "#00ff00"


class TestSphericalAndFovFallbacks:
    """Tests for spherical coordinates and FOV fallback paths."""

    def test_spherical_to_cartesian_without_numpy(self, monkeypatch):
        """spherical_to_cartesian should return a list without numpy."""
        monkeypatch.setattr(camera, "HAS_NUMPY", False)
        pos = camera.spherical_to_cartesian(
            radius=2.0, azimuth=0.0, elevation=0.0, target=(1.0, 2.0, 3.0)
        )
        assert isinstance(pos, list)
        assert pos[0] == pytest.approx(3.0)
        assert pos[1] == pytest.approx(2.0)
        assert pos[2] == pytest.approx(3.0)

    def test_compute_fovy_without_numpy(self, monkeypatch):
        """compute_fovy should compute FOV without numpy."""
        monkeypatch.setattr(camera, "HAS_NUMPY", False)
        fovy = camera.compute_fovy(focal_length=35.0, sensor_height=24.0)
        assert fovy > 0.0
        assert fovy < 180.0


class TestCameraRaysFallbacks:
    """Tests for camera ray fallback behaviour."""

    def test_get_camera_rays_without_numpy(self, monkeypatch):
        """get_camera_rays should return (None, None) without numpy."""
        monkeypatch.setattr(camera, "HAS_NUMPY", False)
        origins, directions = camera.get_camera_rays(None, None, (480, 640))
        assert origins is None
        assert directions is None
