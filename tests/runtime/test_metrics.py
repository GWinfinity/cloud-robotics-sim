"""Tests for runtime metrics collection and storage."""

from types import SimpleNamespace

import pytest

from cloud_robotics_sim.runtime.metrics import (
    EpisodeMetrics,
    MetricCollector,
    MetricSummary,
    default_thresholds,
)


class TestEpisodeMetrics:
    """Tests for EpisodeMetrics."""

    def test_to_dict(self):
        metric = EpisodeMetrics(
            success=True,
            episode_reward=42.0,
            episode_length=100,
            latency_ms=12.5,
            physics_violations=2,
            stability_score=0.9,
            final_distance_to_goal=0.05,
            time_to_success=3.4,
            info={"extra": "data"},
        )
        d = metric.to_dict()
        assert d["success"] is True
        assert d["episode_reward"] == pytest.approx(42.0)
        assert d["episode_length"] == 100
        assert d["latency_ms"] == pytest.approx(12.5)
        assert d["physics_violations"] == 2
        assert d["stability_score"] == pytest.approx(0.9)
        assert d["final_distance_to_goal"] == pytest.approx(0.05)
        assert d["time_to_success"] == pytest.approx(3.4)
        assert d["info"] == {"extra": "data"}


class TestMetricSummary:
    """Tests for MetricSummary."""

    def test_to_dict_defaults(self):
        summary = MetricSummary()
        d = summary.to_dict()
        assert d["episodes"] == 0
        assert d["mean_final_distance"] is None

    def test_from_episodes_empty(self):
        summary = MetricSummary.from_episodes([])
        assert summary.episodes == 0
        assert summary.success_rate == pytest.approx(0.0)
        assert summary.mean_final_distance is None

    def test_from_episodes_without_final_distance(self):
        episodes = [
            EpisodeMetrics(success=True, episode_reward=10.0, episode_length=5),
            EpisodeMetrics(success=False, episode_reward=-5.0, episode_length=10),
        ]
        summary = MetricSummary.from_episodes(episodes)
        assert summary.episodes == 2
        assert summary.success_rate == pytest.approx(0.5)
        assert summary.mean_reward == pytest.approx(2.5)
        assert summary.mean_length == pytest.approx(7.5)
        assert summary.mean_final_distance is None
        assert summary.raw == episodes

    def test_from_episodes_with_final_distance(self):
        episodes = [
            EpisodeMetrics(success=True, final_distance_to_goal=0.1),
            EpisodeMetrics(success=False, final_distance_to_goal=0.2),
            EpisodeMetrics(success=False),
        ]
        summary = MetricSummary.from_episodes(episodes)
        assert summary.mean_final_distance == pytest.approx(0.15)


class TestMetricCollectorCollectEpisode:
    """Tests for MetricCollector.collect_episode action space handling."""

    def _make_env(self, action_space, info_at_end=None, terminated_at=2):
        class Env:
            def __init__(self):
                self.action_space = action_space
                self.step_count = 0

            def reset(self):
                self.step_count = 0
                return None, info_at_end or {}

            def step(self, action):
                self.step_count += 1
                info = info_at_end if self.step_count >= terminated_at else {}
                return None, 1.0, self.step_count >= terminated_at, False, info

        return Env()

    def test_action_space_dict(self):
        collector = MetricCollector()
        env = self._make_env({"low": -0.5, "high": 0.5, "shape": (3,)})
        metric = collector.collect_episode(env, max_steps=5)
        assert metric.episode_reward > 0
        assert metric.episode_length == 2

    def test_action_space_object_with_sample(self):
        collector = MetricCollector()
        action_space = SimpleNamespace()
        action_space.sample = lambda: [0.0, 1.0]
        env = self._make_env(action_space)
        metric = collector.collect_episode(env, max_steps=5)
        assert metric.episode_reward > 0

    def test_action_space_none(self):
        collector = MetricCollector()
        env = self._make_env(None)
        metric = collector.collect_episode(env, max_steps=5)
        assert metric.episode_reward > 0

    def test_info_records_physics_and_stability(self):
        collector = MetricCollector()
        env = self._make_env(
            None, info_at_end={"physics_violations": 2, "stability_score": 0.7}
        )
        metric = collector.collect_episode(env, max_steps=5)
        assert metric.physics_violations == 2
        assert metric.stability_score == pytest.approx(0.7)

    def test_info_records_final_distance_and_time_to_success(self):
        collector = MetricCollector()
        env = self._make_env(
            None,
            info_at_end={
                "success": True,
                "final_distance": 0.05,
                "time_to_success": 4.2,
            },
        )
        metric = collector.collect_episode(env, max_steps=5)
        assert metric.success is True
        assert metric.final_distance_to_goal == pytest.approx(0.05)
        assert metric.time_to_success == pytest.approx(4.2)


class TestMetricCollectorLifecycle:
    """Tests for MetricCollector collection lifecycle."""

    def _make_env(self):
        class Env:
            def __init__(self):
                self.action_space = None

            def reset(self):
                return None, {}

            def step(self, _action):
                return None, 1.0, True, False, {}

        return Env()

    def test_collect_n_episodes(self):
        collector = MetricCollector()
        summary = collector.collect_n_episodes(self._make_env(), n=3, max_steps=5)
        assert summary.episodes == 3
        assert len(collector.episodes) == 3

    def test_reset(self):
        collector = MetricCollector()
        collector.collect_n_episodes(self._make_env(), n=2, max_steps=5)
        collector.reset()
        assert collector.episodes == []

    def test_save_and_load(self, tmp_path):
        collector = MetricCollector()
        collector.collect_n_episodes(self._make_env(), n=2, max_steps=5)
        path = tmp_path / "metrics.json"
        collector.save(path)
        assert path.exists()

        loaded = MetricCollector()
        loaded.load(path)
        assert len(loaded.episodes) == 2
        assert loaded.episodes[0].episode_reward == collector.episodes[0].episode_reward


class TestDefaultThresholds:
    """Tests for metrics default thresholds."""

    def test_default_thresholds(self):
        thresholds = default_thresholds()
        assert thresholds["min_success_rate"] == pytest.approx(0.8)
        assert thresholds["max_latency_ms"] == pytest.approx(100.0)
        assert thresholds["max_physics_violations"] == pytest.approx(5.0)
        assert thresholds["min_stability_score"] == pytest.approx(0.85)
        assert thresholds["min_mean_reward"] == pytest.approx(-50.0)
