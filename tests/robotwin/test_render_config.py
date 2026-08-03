"""Tests for the three-tier render configuration (doc section 6)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from cloud_robotics_sim.robotwin.render_config import (
    RenderConfig,
    compare_render_pair,
    create_scene_with_render_config,
    load_render_config,
    make_genesis_renderer,
)

RENDER_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "render"


class TestRenderConfig:
    """Dataclass validation."""

    def test_defaults(self) -> None:
        config = RenderConfig()
        assert config.mode == "rasterizer"
        assert config.resolution == (640, 480)
        assert config.spp == 32
        assert config.denoise is True

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            RenderConfig(mode="path_tracing")  # type: ignore[arg-type]

    def test_invalid_resolution(self) -> None:
        with pytest.raises(ValueError, match="resolution"):
            RenderConfig(resolution=(640,))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="resolution"):
            RenderConfig(resolution=(0, 480))

    def test_invalid_spp(self) -> None:
        with pytest.raises(ValueError, match="spp"):
            RenderConfig(spp=0)


class TestLoadRenderConfig:
    """YAML loading, including the shipped tier configs."""

    @pytest.mark.parametrize(
        "filename,expected_mode",
        [
            ("rasterizer.yaml", "rasterizer"),
            ("raytracer.yaml", "raytracer"),
            ("batch_madrona.yaml", "batch"),
        ],
    )
    def test_shipped_configs(self, filename: str, expected_mode: str) -> None:
        config = load_render_config(RENDER_CONFIG_DIR / filename)
        assert config.mode == expected_mode
        if expected_mode == "raytracer":
            assert config.spp == 32  # SAPIEN RT alignment (doc 6.1)
        if expected_mode == "batch":
            assert config.resolution == (256, 256)  # Madrona sweet spot

    def test_unknown_keys_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("mode: rasterizer\nbogus: 1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="bogus"):
            load_render_config(path)

    def test_non_mapping_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(TypeError):
            load_render_config(path)


class TestMakeGenesisRenderer:
    """Renderer instantiation per tier (genesis-world is a core dep)."""

    def test_rasterizer(self) -> None:
        gs = pytest.importorskip("genesis")
        renderer = make_genesis_renderer(RenderConfig(mode="rasterizer"))
        assert isinstance(renderer, gs.renderers.Rasterizer)

    def test_raytracer(self) -> None:
        gs = pytest.importorskip("genesis")
        renderer = make_genesis_renderer(RenderConfig(mode="raytracer"))
        assert isinstance(renderer, gs.renderers.RayTracer)

    def test_batch(self) -> None:
        gs = pytest.importorskip("genesis")
        renderer = make_genesis_renderer(RenderConfig(mode="batch"))
        assert isinstance(renderer, gs.renderers.BatchRenderer)


class TestCreateSceneWithRenderConfig:
    """Scene wiring passes the renderer through the backend interface."""

    def test_renderer_forwarded(self) -> None:
        backend = MagicMock()
        config = RenderConfig(mode="rasterizer", resolution=(320, 240))
        create_scene_with_render_config(backend, config, dt=0.02, substeps=2)

        call = backend.create_scene.call_args
        assert call.kwargs["dt"] == 0.02
        assert call.kwargs["substeps"] == 2
        assert call.kwargs["headless"] is True
        gs = pytest.importorskip("genesis")
        assert isinstance(call.kwargs["renderer"], gs.renderers.Rasterizer)


class TestCompareRenderPair:
    """Image-pair alignment metrics (doc section 6.2, gate V4)."""

    def test_identical_images(self) -> None:
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        result = compare_render_pair(img, img.copy())
        assert result.is_perfect
        assert result.psnr == float("inf")
        assert result.mae == 0.0

    def test_known_difference(self) -> None:
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = np.full((4, 4, 3), 255, dtype=np.uint8)
        result = compare_render_pair(a, b)
        assert result.psnr == pytest.approx(0.0, abs=1e-9)
        assert result.mae == pytest.approx(255.0)

    def test_psnr_value(self) -> None:
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = np.full((4, 4, 3), 25, dtype=np.uint8)  # mse = 625 -> psnr ~ 20.2dB
        result = compare_render_pair(a, b)
        assert result.psnr == pytest.approx(20.17, abs=0.01)

    def test_shape_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="shapes"):
            compare_render_pair(np.zeros((4, 4, 3)), np.zeros((5, 5, 3)))
