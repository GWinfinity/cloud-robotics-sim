"""Tests for Genesis simulation cache cleanup utilities."""

import sys
from unittest.mock import MagicMock

from cloud_robotics_sim.utils import cache_cleanup


class TestIsDatasetPipelineEnabled:
    """Tests for the dataset pipeline flag resolution."""

    def test_env_var_true(self, monkeypatch):
        """Environment variable set to 'true' enables the pipeline."""
        monkeypatch.setenv("CRS_DATASET_PIPELINE", "true")
        assert cache_cleanup.is_dataset_pipeline_enabled() is True

    def test_env_var_false(self, monkeypatch):
        """Environment variable set to 'false' disables the pipeline."""
        monkeypatch.setenv("CRS_DATASET_PIPELINE", "false")
        assert cache_cleanup.is_dataset_pipeline_enabled() is False

    def test_env_var_overrides_config(self, monkeypatch):
        """The environment variable takes precedence over config."""
        monkeypatch.setenv("CRS_DATASET_PIPELINE", "0")
        assert (
            cache_cleanup.is_dataset_pipeline_enabled({"dataset_pipeline": True})
            is False
        )

    def test_config_true(self):
        """Config flag can enable the pipeline when env is unset."""
        assert (
            cache_cleanup.is_dataset_pipeline_enabled({"dataset_pipeline": True})
            is True
        )

    def test_default_false(self):
        """Default is disabled when neither env nor config is set."""
        assert cache_cleanup.is_dataset_pipeline_enabled() is False

    def test_dataclass_like_config(self):
        """Config can be a dataclass-like object with attributes."""
        config = type("Config", (), {"dataset_pipeline": True})()
        assert cache_cleanup.is_dataset_pipeline_enabled(config) is True


class TestPathResolution:
    """Tests for cache / pipeline directory resolution."""

    def test_default_cache_dir(self, monkeypatch, tmp_path):
        """Default cache dir resolves relative to cwd."""
        monkeypatch.chdir(tmp_path)
        assert cache_cleanup.get_cache_dir() == tmp_path / "outputs" / "sim_cache"

    def test_env_cache_dir(self, monkeypatch, tmp_path):
        """CRS_SIM_CACHE_DIR overrides the default."""
        monkeypatch.setenv("CRS_SIM_CACHE_DIR", str(tmp_path / "custom_cache"))
        assert cache_cleanup.get_cache_dir() == tmp_path / "custom_cache"

    def test_config_pipeline_dir(self, monkeypatch, tmp_path):
        """Config provides a fallback pipeline directory."""
        monkeypatch.chdir(tmp_path)
        assert cache_cleanup.get_pipeline_dir(
            {"dataset_pipeline_dir": "pipeline/staging"}
        ) == (tmp_path / "pipeline" / "staging")


class TestCleanCacheDirectory:
    """Tests for deleting the simulation cache directory."""

    def test_removes_existing_contents(self, tmp_path):
        """Existing files under the cache directory are removed."""
        cache = tmp_path / "sim_cache"
        (cache / "sub").mkdir(parents=True)
        (cache / "sub" / "file.txt").write_text("data")

        cleaned = cache_cleanup.clean_cache_directory(cache)

        assert cleaned == cache
        assert cache.exists()
        assert not (cache / "sub").exists()

    def test_creates_missing_directory(self, tmp_path):
        """A missing cache directory is created empty."""
        cache = tmp_path / "new_cache"
        cleaned = cache_cleanup.clean_cache_directory(cache)
        assert cleaned == cache
        assert cache.is_dir()


class TestStageCacheForPipeline:
    """Tests for staging cache into the dataset pipeline."""

    def test_moves_cache_to_timestamped_subdir(self, tmp_path):
        """Cache is moved under a timestamped staging subdirectory."""
        cache = tmp_path / "sim_cache"
        (cache / "sub").mkdir(parents=True)
        (cache / "sub" / "file.txt").write_text("data")
        pipeline = tmp_path / "pipeline"

        staged = cache_cleanup.stage_cache_for_pipeline(cache, pipeline)

        assert staged.parent == pipeline
        assert (staged / "sub" / "file.txt").read_text() == "data"
        assert cache.exists() and cache.is_dir()
        assert not (cache / "sub").exists()

    def test_no_cache_warns_and_creates_dirs(self, tmp_path, caplog):
        """Missing source cache is reported and directories are created."""
        cache = tmp_path / "empty_cache"
        pipeline = tmp_path / "pipeline"

        with caplog.at_level(
            "WARNING", logger="cloud_robotics_sim.utils.cache_cleanup"
        ):
            staged = cache_cleanup.stage_cache_for_pipeline(cache, pipeline)

        assert staged.parent == pipeline
        assert "No simulation cache to stage" in caplog.text


class TestCleanupAfterSimulation:
    """Tests for the top-level cleanup dispatcher."""

    def test_pipeline_false_cleans_runtime_and_cache(self, tmp_path, monkeypatch):
        """When pipeline is disabled, the cache is deleted."""
        cache = tmp_path / "sim_cache"
        (cache / "sub").mkdir(parents=True)
        (cache / "sub" / "file.txt").write_text("data")

        # Genesis is not initialized here, so clean_genesis_runtime is a no-op.
        cache_cleanup.cleanup_after_simulation(
            cache_dir=cache,
            dataset_pipeline=False,
        )

        assert cache.exists()
        assert not (cache / "sub").exists()

    def test_pipeline_true_stages_cache(self, tmp_path):
        """When pipeline is enabled, the cache is staged."""
        cache = tmp_path / "sim_cache"
        pipeline = tmp_path / "pipeline"
        cache.mkdir(parents=True)
        (cache / "file.txt").write_text("data")

        cache_cleanup.cleanup_after_simulation(
            cache_dir=cache,
            pipeline_dir=pipeline,
            dataset_pipeline=True,
        )

        assert cache.exists() and cache.is_dir()
        assert any(pipeline.iterdir())

    def test_pipeline_inferred_from_env(self, tmp_path, monkeypatch):
        """When no explicit flag is given, the env variable decides."""
        monkeypatch.setenv("CRS_DATASET_PIPELINE", "true")
        cache = tmp_path / "sim_cache"
        pipeline = tmp_path / "pipeline"
        cache.mkdir(parents=True)
        (cache / "file.txt").write_text("data")

        cache_cleanup.cleanup_after_simulation(cache_dir=cache, pipeline_dir=pipeline)

        assert any(pipeline.iterdir())


class TestCleanGenesisRuntime:
    """Tests for Genesis runtime teardown."""

    def test_destroy_called_when_initialized(self, monkeypatch):
        """gs.destroy() and clear_caches() are called when initialized."""
        gs = MagicMock()
        gs._initialized = True
        gs.utils.misc.clear_caches = MagicMock()
        monkeypatch.setitem(sys.modules, "genesis", gs)

        cache_cleanup.clean_genesis_runtime()

        gs.destroy.assert_called_once()
        gs.utils.misc.clear_caches.assert_called_once()

    def test_no_destroy_when_not_initialized(self, monkeypatch):
        """gs.destroy() is skipped when Genesis is not initialized."""
        gs = MagicMock()
        gs._initialized = False
        gs.utils.misc.clear_caches = MagicMock()
        monkeypatch.setitem(sys.modules, "genesis", gs)

        cache_cleanup.clean_genesis_runtime()

        gs.destroy.assert_not_called()
        gs.utils.misc.clear_caches.assert_called_once()

    def test_graceful_when_genesis_missing(self, monkeypatch):
        """Cleanup does not crash when genesis is not installed."""
        monkeypatch.delitem("sys.modules", "genesis", raising=False)
        cache_cleanup.clean_genesis_runtime()
