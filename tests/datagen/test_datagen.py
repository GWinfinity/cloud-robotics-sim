"""Tests for the genesis-world datagen slice (no GPU, no gs.init)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloud_robotics_sim.datagen import (  # noqa: E402
    CameraSpec,
    DatasetWriter,
    EpisodeResult,
    EpisodeSpec,
    GenesisEpisodeRenderer,
    ObjectPlacement,
    RandomizationSpec,
)
from cloud_robotics_sim.datagen import render as render_module  # noqa: E402


def _spec(**overrides) -> EpisodeSpec:
    base = {
        "episode_id": "ep0",
        "objects": (ObjectPlacement("glass_cup", pos=(0.0, 0.0, 0.1)),),
        "cameras": (
            CameraSpec(name="tpv", pos=(1.0, 0.0, 1.0), lookat=(0.0, 0.0, 0.0)),
        ),
        "steps": 2,
        "randomization": RandomizationSpec(seed=7),
    }
    base.update(overrides)
    return EpisodeSpec(**base)


def test_spec_rejects_invalid_configs() -> None:
    """Names must be unique, follows must reference placed objects, ranges sane."""
    with pytest.raises(ValueError, match="unique"):
        _spec(
            cameras=(
                CameraSpec(name="cam", pos=(0, 0, 1), lookat=(0, 0, 0)),
                CameraSpec(name="cam", pos=(0, 0, 2), lookat=(0, 0, 0)),
            )
        )
    with pytest.raises(ValueError, match="unknown objects"):
        _spec(
            cameras=(
                CameraSpec(
                    name="fpv", pos=(0, 0, 1), lookat=(0, 0, 0), follow="sponge"
                ),
            )
        )
    with pytest.raises(ValueError, match="fov"):
        CameraSpec(name="cam", pos=(0, 0, 1), lookat=(0, 0, 0), fov=400.0)
    with pytest.raises(ValueError, match="background"):
        RandomizationSpec(background_colors=())
    with pytest.raises(ValueError, match="resolution"):
        CameraSpec(name="cam", pos=(0, 0, 1), lookat=(0, 0, 0), res=(8, 480))


def test_randomization_sample_is_seeded() -> None:
    """Same seed reproduces the episode's DR draw; different seeds diverge."""
    sample_a = RandomizationSpec(seed=3).sample()
    sample_b = RandomizationSpec(seed=3).sample()
    sample_c = RandomizationSpec(seed=4).sample()
    assert sample_a == sample_b
    assert sample_a != sample_c
    meta = sample_a.to_metadata()
    assert len(meta["lights"]) == RandomizationSpec().n_lights
    for key in ("background", "floor"):
        assert len(meta[key]) == 3
        assert all(0.0 <= c <= 1.0 for c in meta[key])


def test_writer_roundtrip_and_resume(tmp_path: Path) -> None:
    """Frames land as PNGs, metadata is JSON, reruns skip existing episodes."""
    writer = DatasetWriter(tmp_path)
    frames = {
        "tpv": np.arange(2 * 4 * 6 * 3, dtype=np.uint8).reshape(2, 4, 6, 3),
        "fpv": np.full((1, 2, 2, 3), 255, dtype=np.uint8),
    }
    result = EpisodeResult(
        episode_id="ep_000001", frames=frames, metadata={"cameras": []}
    )
    episode_dir = writer.write_episode(result)
    assert (episode_dir / "tpv" / "frame_000000.png").is_file()
    assert (episode_dir / "fpv" / "frame_000000.png").is_file()
    assert len(list((episode_dir / "tpv").glob("*.png"))) == 2
    meta = json.loads((episode_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["cameras_frames"]["tpv"]["frames"] == 2
    assert meta["cameras_frames"]["fpv"]["res"] == [2, 2]
    assert writer.written_episode_ids() == ["ep_000001"]
    assert writer.is_done("ep_000001")

    png_before = (episode_dir / "tpv" / "frame_000000.png").read_bytes()
    writer.write_episode(
        EpisodeResult(
            episode_id="ep_000001",
            frames={"tpv": np.zeros((1, 4, 6, 3), dtype=np.uint8)},
            metadata={},
        )
    )
    assert (episode_dir / "tpv" / "frame_000000.png").read_bytes() == png_before
    assert writer.written_episode_ids() == ["ep_000001"]


def test_writer_rejects_path_traversal(tmp_path: Path) -> None:
    """Episode and camera names are directory names; traversal must fail."""
    writer = DatasetWriter(tmp_path)
    with pytest.raises(ValueError, match="episode_id"):
        writer.write_episode(
            EpisodeResult(episode_id="../escape", frames={}, metadata={})
        )
    with pytest.raises(ValueError, match="episode_id"):
        writer.write_episode(
            EpisodeResult(
                episode_id="ok",
                frames={"../escape": np.zeros((1, 2, 2, 3), dtype=np.uint8)},
                metadata={},
            )
        )


class _FakeCamera:
    def __init__(self, res: tuple[int, int]) -> None:
        self.res = res
        self.poses: list[tuple] = []

    def render(self):
        return np.zeros((self.res[1], self.res[0], 3), dtype=np.uint8)

    def set_pose(self, pos=None, lookat=None) -> None:
        self.poses.append((tuple(pos), tuple(lookat)))


class _FakeScene:
    def __init__(self, scenes: list, **kwargs) -> None:
        scenes.append(self)
        self.kwargs = kwargs
        self.cameras: list[tuple[dict, _FakeCamera]] = []
        self.lights: list[dict] = []
        self.planes = 0
        self.steps = 0
        self.destroyed = False

    def add_camera(self, **kwargs):
        camera = _FakeCamera(kwargs["res"])
        self.cameras.append((kwargs, camera))
        return camera

    def add_entity(self, *args, **kwargs):
        self.planes += 1
        return SimpleNamespace(get_pos=lambda: np.array([0.5, 0.0, 0.1]))

    def add_light(self, **kwargs) -> None:
        self.lights.append(kwargs)

    def build(self) -> None:
        pass

    def step(self) -> None:
        self.steps += 1

    def destroy(self) -> None:
        self.destroyed = True


def _fake_genesis_module(scenes: list) -> SimpleNamespace:
    return SimpleNamespace(
        Scene=lambda **kwargs: _FakeScene(scenes, **kwargs),
        morphs=SimpleNamespace(Plane=lambda: object()),
        surfaces=SimpleNamespace(Default=lambda **kwargs: SimpleNamespace(**kwargs)),
        options=SimpleNamespace(
            SimOptions=lambda **kwargs: kwargs,
            VisOptions=lambda **kwargs: kwargs,
        ),
    )


class _FakeLibrary:
    """Drop-in for RoboTwinObjectLibrary resolving every placement."""

    def __init__(self, objects_dir) -> None:
        self.objects_dir = objects_dir
        self.spawned: list[tuple] = []

    def list_classes(self) -> list[str]:
        return ["001_glass_cup", "002_sponge"]

    def spawn_in_scene(self, scene, class_name, index=0, pos=None, quat=None):
        self.spawned.append((class_name, index))
        return SimpleNamespace(get_pos=lambda: np.asarray([0.5, 0.0, 0.1]))


def _render_with_fake_genesis(spec: EpisodeSpec, scenes: list) -> EpisodeResult:
    fake_gs = _fake_genesis_module(scenes)
    with (
        mock.patch.dict(sys.modules, {"genesis": fake_gs}),
        mock.patch.object(render_module, "genesis_init", lambda **kwargs: None),
        mock.patch.object(render_module, "RoboTwinObjectLibrary", _FakeLibrary),
    ):
        renderer = GenesisEpisodeRenderer("unused_objects_dir", backend="cpu")
        return renderer.render_episode(spec)


def test_renderer_stages_episode_and_captures_frames() -> None:
    """The renderer wires scene, cameras, DR, and returns stacked frames."""
    spec = _spec(
        cameras=(
            CameraSpec(name="tpv", pos=(1.0, 0.0, 1.0), lookat=(0.0, 0.0, 0.0)),
            CameraSpec(
                name="fpv",
                pos=(0.0, 0.0, 0.3),
                lookat=(0.0, 0.0, -1.0),
                res=(320, 240),
                follow="glass_cup",
            ),
        ),
        steps=3,
    )
    scenes: list[_FakeScene] = []
    result = _render_with_fake_genesis(spec, scenes)

    assert len(scenes) == 1
    scene = scenes[0]
    assert scene.destroyed
    assert scene.steps == 3
    assert scene.planes == 1
    assert len(scene.lights) == spec.randomization.n_lights
    vis_options = scene.kwargs["vis_options"]
    expected_dr = spec.randomization.sample()
    assert tuple(vis_options["background_color"]) == expected_dr.background

    assert result.frames["tpv"].shape == (3, 480, 640, 3)
    assert result.frames["fpv"].shape == (3, 240, 320, 3)
    assert result.frames["tpv"].dtype == np.uint8
    assert result.steps == 3
    camera_kwargs, fpv_camera = scene.cameras[1]
    assert camera_kwargs["model"] == "pinhole"
    assert fpv_camera.poses, "follow camera must be re-posed every step"
    first_pos, first_lookat = fpv_camera.poses[0]
    assert first_pos == pytest.approx((0.5, 0.0, 0.4))
    assert first_lookat == pytest.approx((0.5, 0.0, -0.9))

    metadata = result.metadata
    assert metadata["episode_id"] == "ep0"
    assert [c["name"] for c in metadata["cameras"]] == ["tpv", "fpv"]
    # bare recipe ids resolve to the library's NNN_name class dirs
    assert (
        render_module.GenesisEpisodeRenderer._resolve_class(
            _FakeLibrary("x"), "glass_cup"
        )
        == "001_glass_cup"
    )
    with pytest.raises(ValueError, match="resolves to"):
        render_module.GenesisEpisodeRenderer._resolve_class(_FakeLibrary("x"), "nope")
    assert metadata["domain_randomization"]["lights"] == [
        {"intensity": light.intensity, "color": list(light.color)}
        for light in expected_dr.lights
    ]


def test_renderer_handles_zero_steps_and_light_failure() -> None:
    """Empty episodes still return correctly shaped buffers; light errors degrade."""
    scenes: list[_FakeScene] = []
    empty = _render_with_fake_genesis(_spec(steps=0), scenes)
    assert empty.frames["tpv"].shape == (0, 480, 640, 3)
    assert empty.steps == 0

    class _NoLightScene(_FakeScene):
        def add_light(self, **kwargs) -> None:
            raise RuntimeError("renderer has no lights")

    no_light_scenes: list[_FakeScene] = []

    def scene_factory(**kwargs):
        return _NoLightScene(no_light_scenes, **kwargs)

    fake_gs = SimpleNamespace(
        Scene=scene_factory,
        morphs=SimpleNamespace(Plane=lambda: object()),
        surfaces=SimpleNamespace(Default=lambda **kwargs: SimpleNamespace(**kwargs)),
        options=SimpleNamespace(
            SimOptions=lambda **kwargs: kwargs,
            VisOptions=lambda **kwargs: kwargs,
        ),
    )
    with (
        mock.patch.dict(sys.modules, {"genesis": fake_gs}),
        mock.patch.object(render_module, "genesis_init", lambda **kwargs: None),
        mock.patch.object(render_module, "RoboTwinObjectLibrary", _FakeLibrary),
    ):
        renderer = GenesisEpisodeRenderer(backend="cpu")
        result = renderer.render_episode(_spec(steps=1))
    assert result.frames["tpv"].shape == (1, 480, 640, 3)
    assert no_light_scenes[0].destroyed


# ---------------------------------------------------------------------------
# Episode generator + shard runner
# ---------------------------------------------------------------------------

import tools.run_datagen as run_datagen_module  # noqa: E402
from cloud_robotics_sim.datagen import EpisodeGenerator, GenerationConfig  # noqa: E402


def _gen_config(**overrides) -> GenerationConfig:
    base = {
        "episodes": 8,
        "steps": 3,
        "available_classes": ("glass_cup", "sponge", "hanger"),
        "seed": 11,
    }
    base.update(overrides)
    return GenerationConfig(**base)


def test_generator_is_deterministic_and_in_bounds() -> None:
    """Same config regenerates identical specs; placements stay in the arena."""
    config = _gen_config()
    generator = EpisodeGenerator(config)
    spec = generator.spec(2)
    again = EpisodeGenerator(config).spec(2)
    assert spec == again
    lo, hi = config.objects_per_episode
    assert lo <= len(spec.objects) <= hi
    half_x, half_y = config.arena
    follow_targets = {p.class_name for p in spec.objects}
    for placement in spec.objects:
        x, y, z = placement.pos
        assert abs(x) <= half_x and abs(y) <= half_y
        assert 0.08 <= z <= 0.25
        assert placement.quat is not None
    fpv = next(c for c in spec.cameras if c.name == "fpv")
    assert fpv.follow in follow_targets
    assert spec.randomization.seed == config.seed + 2


def test_generator_rejects_bad_config_and_shards() -> None:
    """Invalid ranges and out-of-range shards fail loudly."""
    with pytest.raises(ValueError, match="available class"):
        _gen_config(available_classes=())
    with pytest.raises(ValueError, match="min <= max"):
        _gen_config(objects_per_episode=(3, 2))
    config = _gen_config()
    with pytest.raises(ValueError, match="shard"):
        config.shard_indices(3, 3)
    with pytest.raises(ValueError, match="outside"):
        EpisodeGenerator(config).spec(8)


def test_shard_indices_cover_dataset_disjointly() -> None:
    """Strided shards partition the index space for several shard counts."""
    config = _gen_config(episodes=17)
    for num_shards in (1, 2, 3, 5, 20):
        shards = [set(config.shard_indices(i, num_shards)) for i in range(num_shards)]
        assert shards[0].union(*shards[1:]) == set(range(17))
        for i in range(num_shards):
            for j in range(i + 1, num_shards):
                assert not (shards[i] & shards[j])


class _FakeBatchRenderer:
    """Renderer stand-in returning tiny frames, optionally failing one episode."""

    def __init__(self, objects_dir, **kwargs) -> None:
        self.objects_dir = objects_dir
        self.calls: list[str] = []
        self.fail_on: str | None = None

    def render_episode(self, spec):
        self.calls.append(spec.episode_id)
        if spec.episode_id == self.fail_on:
            raise RuntimeError("boom")
        return EpisodeResult(
            episode_id=spec.episode_id,
            frames={"tpv": np.zeros((1, 4, 4, 3), dtype=np.uint8)},
            metadata={"objects": []},
        )


def _run_tool(
    monkeypatch, tmp_path: Path, renderer: _FakeBatchRenderer, extra: list[str]
) -> int:
    objects_dir = tmp_path / "objects"
    objects_dir.mkdir(exist_ok=True)
    for name in ("001_glass_cup", "002_sponge"):
        (objects_dir / name).mkdir(exist_ok=True)
    fake_gs = SimpleNamespace(gpu=object(), cpu=object(), init=lambda **kwargs: None)
    monkeypatch.setattr(
        run_datagen_module, "GenesisEpisodeRenderer", lambda *a, **k: renderer
    )
    monkeypatch.setattr(
        run_datagen_module, "get_genesis_backend", lambda name: object()
    )
    monkeypatch.setitem(sys.modules, "genesis", fake_gs)
    argv = [
        "--objects-dir",
        str(objects_dir),
        "--output-dir",
        str(tmp_path / "dataset"),
        "--episodes",
        "4",
        "--steps",
        "1",
        *extra,
    ]
    return run_datagen_module.main(argv)


def test_run_datagen_shard_resume_and_failure(monkeypatch, tmp_path, capsys) -> None:
    """Shards render their strided subset; failures are reported and retried."""
    renderer = _FakeBatchRenderer("objects")
    renderer.fail_on = "ep_000002"

    exit_code = _run_tool(
        monkeypatch, tmp_path, renderer, ["--shard", "0", "--num-shards", "2"]
    )
    assert exit_code == 1
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["done"] == 1  # ep_000000 rendered, ep_000002 failed
    assert summary["failed"] == [{"episode_id": "ep_000002", "error": "boom"}]
    assert renderer.calls == ["ep_000000", "ep_000002"]

    # rerun the same shard: the done episode is skipped, the failed one retried
    renderer.fail_on = None
    exit_code = _run_tool(
        monkeypatch, tmp_path, renderer, ["--shard", "0", "--num-shards", "2"]
    )
    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["done"] == 1
    assert summary["skipped"] == 1
    assert renderer.calls == ["ep_000000", "ep_000002", "ep_000002"]

    writer = DatasetWriter(tmp_path / "dataset")
    assert writer.written_episode_ids() == ["ep_000000", "ep_000002"]


def test_run_datagen_rejects_unknown_classes(monkeypatch, tmp_path) -> None:
    """A typo in --classes fails fast listing what the library has."""
    renderer = _FakeBatchRenderer("objects")
    exit_code = _run_tool(
        monkeypatch, tmp_path, renderer, ["--classes", "glass_cup", "nope"]
    )
    assert exit_code == 2
    assert renderer.calls == []
