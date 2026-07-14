"""Tests for rendering utilities."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from cloud_robotics_sim.utils import rendering


class TestSetRenderMaterial:
    """Tests for set_render_material."""

    def test_set_base_color_hex(self):
        """set_render_material should convert a hex color and call set_base_color."""
        material = MagicMock()
        rendering.set_render_material(material, color="#00FF00")
        material.set_base_color.assert_called_once()

    def test_set_metallic_and_roughness(self):
        """set_render_material should set metallic and roughness properties."""
        material = MagicMock()
        rendering.set_render_material(material, metallic=0.5, roughness=0.2)
        material.set_metallic.assert_called_once_with(0.5)
        material.set_roughness.assert_called_once_with(0.2)

    def test_set_unknown_attribute(self):
        """set_render_material should set arbitrary attributes."""
        material = MagicMock()
        rendering.set_render_material(material, custom_prop=42)
        assert material.custom_prop == 42


class TestSetEntityColor:
    """Tests for set_entity_color."""

    def test_set_entity_color_string(self):
        """set_entity_color should handle a hex color string."""
        entity = MagicMock()
        entity.render_shapes = []
        entity.get_children.return_value = []
        rendering.set_entity_color(entity, "#FF0000")
        entity.set_color.assert_called_once()

    def test_set_entity_color_recursive(self):
        """set_entity_color should apply recursively to children."""
        child = MagicMock()
        child.render_shapes = []
        child.get_children.return_value = []
        parent = MagicMock()
        parent.render_shapes = []
        parent.get_children.return_value = [child]
        rendering.set_entity_color(parent, [1.0, 0.0, 0.0, 1.0])
        parent.set_color.assert_called_once()
        child.set_color.assert_called_once()


class TestShaderConfig:
    """Tests for ShaderConfig."""

    def test_default_values(self):
        """ShaderConfig should store default values."""
        cfg = rendering.ShaderConfig()
        assert cfg.shader_pack == "default"
        assert cfg.shader_pack_config["ray_tracing_denoiser"] == "optix"

    def test_custom_values(self):
        """ShaderConfig should accept custom values."""
        cfg = rendering.ShaderConfig(shader_pack="rt", ray_tracing_path_depth=8)
        assert cfg.shader_pack == "rt"
        assert cfg.shader_pack_config["ray_tracing_path_depth"] == 8


class TestConfigureRendering:
    """Tests for configure_rendering."""

    def test_configure_no_genesis(self, monkeypatch):
        """configure_rendering should return False when Genesis is unavailable."""
        monkeypatch.setattr(rendering, "HAS_GENESIS", False)
        assert rendering.configure_rendering() is False

    def test_configure_success(self, monkeypatch):
        """configure_rendering should return True on success."""

        class FakeRender:
            def set_viewer_shader_dir(self, pack):
                self.pack = pack

        class FakeGs:
            render = FakeRender()

        monkeypatch.setattr(rendering, "HAS_GENESIS", True)
        monkeypatch.setattr(rendering, "gs", FakeGs())
        assert rendering.configure_rendering(shader_pack="rt") is True


class TestTextureUtilities:
    """Tests for texture utilities."""

    def test_load_texture_missing_file(self):
        """load_texture should return None for a missing file."""
        assert rendering.load_texture("/nonexistent/texture.png") is None

    def test_load_texture_with_genesis(self, monkeypatch, tmp_path):
        """load_texture should use gs.Texture when available."""
        texture_path = tmp_path / "fake.png"
        texture_path.write_text("fake image")

        class FakeGs:
            class Texture:
                def __init__(self, path, **_kwargs):
                    self.path = path

        monkeypatch.setattr(rendering, "HAS_GENESIS", True)
        monkeypatch.setattr(rendering, "gs", FakeGs())
        texture = rendering.load_texture(texture_path)
        assert texture is not None
        assert texture.path == str(texture_path)

    def test_create_checkerboard_texture(self):
        """create_checkerboard_texture should return a square RGB array."""
        texture = rendering.create_checkerboard_texture(size=128, check_size=32)
        assert texture is not None
        assert texture.shape == (128, 128, 3)


class TestScreenshotAndRecording:
    """Tests for screenshot and recording helpers."""

    def test_save_screenshot(self, tmp_path):
        """save_screenshot should write an image file when a render method is available."""

        class FakeCamera:
            def render(self, rgb=True):
                return np.zeros((4, 4, 3), dtype=np.uint8)

        output = tmp_path / "shot.png"
        rendering.save_screenshot(FakeCamera(), output)
        assert output.exists() or output.with_suffix(".npy").exists()

    def test_start_recording(self, tmp_path):
        """start_recording should return a handle when the viewer supports it."""

        class FakeViewer:
            def start_recording(self, path, fps=30):
                return MagicMock()

        handle = rendering.start_recording(FakeViewer(), tmp_path / "vid.mp4")
        assert handle is not None

    def test_stop_recording(self):
        """stop_recording should call stop on the handle."""
        handle = MagicMock()
        rendering.stop_recording(handle)
        handle.stop.assert_called_once()


class TestSetRenderMaterialBranches:
    """Tests covering setter vs property assignment branches."""

    def test_set_base_color_property(self):
        """set_render_material should assign base_color when setter missing."""
        material = SimpleNamespace(base_color=None)
        rendering.set_render_material(material, color=[1.0, 0.0, 0.0, 1.0])
        assert material.base_color == [1.0, 0.0, 0.0, 1.0]

    def test_set_metallic_property(self):
        """set_render_material should assign metallic when setter missing."""
        material = SimpleNamespace(metallic=None)
        rendering.set_render_material(material, metallic=0.8)
        assert material.metallic == 0.8

    def test_set_roughness_property(self):
        """set_render_material should assign roughness when setter missing."""
        material = SimpleNamespace(roughness=None)
        rendering.set_render_material(material, roughness=0.3)
        assert material.roughness == 0.3

    def test_set_specular_setter_and_property(self):
        """set_render_material should set specular via setter or property."""
        setter_mat = MagicMock()
        rendering.set_render_material(setter_mat, specular=0.4)
        setter_mat.set_specular.assert_called_once_with(0.4)

        prop_mat = SimpleNamespace(specular=None)
        rendering.set_render_material(prop_mat, specular=0.4)
        assert prop_mat.specular == 0.4

    def test_set_emission_setter_and_property(self):
        """set_render_material should set emission via setter or property."""
        setter_mat = MagicMock()
        rendering.set_render_material(setter_mat, emission=[0.1, 0.1, 0.1])
        setter_mat.set_emission.assert_called_once_with([0.1, 0.1, 0.1])

        prop_mat = SimpleNamespace(emission=None)
        rendering.set_render_material(prop_mat, emission=[0.1, 0.1, 0.1])
        assert prop_mat.emission == [0.1, 0.1, 0.1]


class TestSetArticulationRenderMaterial:
    """Tests for articulation material assignment."""

    def _make_shape(self, *, has_parts=True):
        material = MagicMock()
        if has_parts:
            part = SimpleNamespace(material=material)
            return SimpleNamespace(parts=[part])
        return SimpleNamespace(material=material)

    def test_no_get_links(self):
        """set_articulation_render_material should return early if no get_links."""
        articulation = SimpleNamespace()
        rendering.set_articulation_render_material(articulation, color="#FF0000")

    def test_link_without_entity(self):
        """set_articulation_render_material should skip links without entity."""
        articulation = MagicMock()
        articulation.get_links.return_value = [SimpleNamespace()]
        rendering.set_articulation_render_material(articulation, color="#FF0000")

    def test_render_shapes_with_parts(self, monkeypatch):
        """set_articulation_render_material should set materials on shape parts."""

        class FakeGs:
            RenderBodyComponent = object()

        monkeypatch.setattr(rendering, "gs", FakeGs())
        articulation = MagicMock()
        render_component = SimpleNamespace(
            render_shapes=[self._make_shape(has_parts=True)]
        )
        entity = SimpleNamespace(find_component_by_type=lambda _type: render_component)
        link = SimpleNamespace(entity=entity)
        articulation.get_links.return_value = [link]
        rendering.set_articulation_render_material(articulation, color="#FF0000")
        render_component.render_shapes[0].parts[
            0
        ].material.set_base_color.assert_called_once()

    def test_render_shapes_without_parts(self, monkeypatch):
        """set_articulation_render_material should set materials directly on shapes."""

        class FakeGs:
            RenderBodyComponent = object()

        monkeypatch.setattr(rendering, "gs", FakeGs())
        articulation = MagicMock()
        render_component = SimpleNamespace(
            render_shapes=[self._make_shape(has_parts=False)]
        )
        entity = SimpleNamespace(find_component_by_type=lambda _type: render_component)
        link = SimpleNamespace(entity=entity)
        articulation.get_links.return_value = [link]
        rendering.set_articulation_render_material(articulation, color="#FF0000")
        render_component.render_shapes[0].material.set_base_color.assert_called_once()

    def test_find_component_raises(self):
        """set_articulation_render_material should swallow exceptions."""
        articulation = MagicMock()

        def raise_exception(_type):
            raise RuntimeError("boom")

        entity = SimpleNamespace(find_component_by_type=raise_exception)
        link = SimpleNamespace(entity=entity)
        articulation.get_links.return_value = [link]
        rendering.set_articulation_render_material(articulation, color="#FF0000")


class TestSetEntityColorBranches:
    """Tests for set_entity_color branches."""

    def test_color_property_branch(self):
        """set_entity_color should assign color property when setter missing."""
        entity = SimpleNamespace(color=None, render_shapes=[], get_children=lambda: [])
        rendering.set_entity_color(entity, [1.0, 0.0, 0.0, 1.0])
        assert entity.color == [1.0, 0.0, 0.0, 1.0]

    def test_render_shapes_branch(self):
        """set_entity_color should set materials on render shapes."""
        material = MagicMock()
        shape = SimpleNamespace(material=material)
        entity = SimpleNamespace(render_shapes=[shape], get_children=lambda: [])
        rendering.set_entity_color(entity, [0.0, 1.0, 0.0, 1.0])
        material.set_base_color.assert_called_once()

    def test_non_recursive(self):
        """set_entity_color should not recurse when recursive=False."""
        child = MagicMock()
        child.render_shapes = []
        child.get_children.return_value = []
        parent = MagicMock()
        parent.render_shapes = []
        parent.get_children.return_value = [child]
        rendering.set_entity_color(parent, [1.0, 0.0, 0.0, 1.0], recursive=False)
        child.set_color.assert_not_called()


class TestConfigureRenderingBranches:
    """Tests for configure_rendering edge cases."""

    def test_configure_ray_tracing_enabled(self, monkeypatch):
        """configure_rendering should set ray tracing options when requested."""

        class FakeRender:
            pack = None
            denoiser = None
            depth = None
            samples = None

            def set_viewer_shader_dir(self, pack):
                self.pack = pack

            def set_ray_tracing_denoiser(self, denoiser):
                self.denoiser = denoiser

            def set_ray_tracing_path_depth(self, depth):
                self.depth = depth

            def set_ray_tracing_samples_per_pixel(self, samples):
                self.samples = samples

        class FakeGs:
            render = FakeRender()

        monkeypatch.setattr(rendering, "HAS_GENESIS", True)
        monkeypatch.setattr(rendering, "gs", FakeGs())
        assert rendering.configure_rendering(
            shader_pack="rt",
            enable_ray_tracing=True,
            denoiser="oidn",
            path_depth=8,
            samples_per_pixel=64,
        )
        assert FakeGs.render.pack == "rt"
        assert FakeGs.render.denoiser == "oidn"
        assert FakeGs.render.depth == 8
        assert FakeGs.render.samples == 64

    def test_configure_failure(self, monkeypatch):
        """configure_rendering should return False when rendering setup fails."""

        class FakeRender:
            def set_viewer_shader_dir(self, _pack):
                raise RuntimeError("fail")

        class FakeGs:
            render = FakeRender()

        monkeypatch.setattr(rendering, "HAS_GENESIS", True)
        monkeypatch.setattr(rendering, "gs", FakeGs())
        assert rendering.configure_rendering() is False


class TestTextureBranches:
    """Tests for texture utility branches."""

    def test_load_texture_exception(self, monkeypatch, tmp_path):
        """load_texture should return None when gs.Texture raises."""

        class FakeGs:
            class Texture:
                def __init__(self, _path, **_kwargs):
                    raise RuntimeError("bad texture")

        texture_path = tmp_path / "bad.png"
        texture_path.write_text("not an image")

        monkeypatch.setattr(rendering, "HAS_GENESIS", True)
        monkeypatch.setattr(rendering, "gs", FakeGs())
        assert rendering.load_texture(texture_path) is None


class TestScreenshotAndRecordingBranches:
    """Tests for screenshot and recording fallback/exception branches."""

    def test_save_screenshot_float_image(self, tmp_path):
        """save_screenshot should convert float images to uint8."""

        class FakeCamera:
            def render(self, rgb=True):
                return np.ones((4, 4, 3), dtype=np.float32) * 0.5

        output = tmp_path / "shot.png"
        rendering.save_screenshot(FakeCamera(), output)
        assert output.exists()

    def test_save_screenshot_pil_fallback(self, monkeypatch, tmp_path):
        """save_screenshot should fall back to npy when PIL is unavailable."""
        import sys

        class FakeCamera:
            def render(self, rgb=True):
                return np.zeros((4, 4, 3), dtype=np.uint8)

        # Block PIL import so save_screenshot falls back to np.save.
        monkeypatch.setitem(sys.modules, "PIL", None)

        output = tmp_path / "shot.png"
        rendering.save_screenshot(FakeCamera(), output)
        assert output.with_suffix(".npy").exists()

    def test_start_recording_exception(self):
        """start_recording should return None when start_recording raises."""

        class FakeViewer:
            def start_recording(self, _path, fps=30):
                raise RuntimeError("recorder failed")

        assert rendering.start_recording(FakeViewer(), "vid.mp4") is None

    def test_stop_recording_exception(self):
        """stop_recording should swallow exceptions from handle.stop."""

        class BadHandle:
            def stop(self):
                raise RuntimeError("stop failed")

        rendering.stop_recording(BadHandle())
