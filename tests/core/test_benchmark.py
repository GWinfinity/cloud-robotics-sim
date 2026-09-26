"""Tests for the benchmark evaluation protocol (core/benchmark.py)."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cloud_robotics_sim.core.benchmark import (
    EpisodeRecord,
    EpisodeTrajectory,
    RandomPolicy,
    ZeroPolicy,
    load_suite,
    make_policy,
    run_benchmark,
    run_episode,
    summarize,
    write_report,
)
from cloud_robotics_sim.core.config_loader import ConfigError
from cloud_robotics_sim.core.task_loader import EvaluationSpec, TaskSpec


def _spec(name: str = "fake_task", max_steps: int = 10) -> TaskSpec:
    return TaskSpec(
        name=name,
        scene_name="scene",
        robot_name="robot",
        task_name="task",
        max_episode_steps=max_steps,
        evaluation=EvaluationSpec(seeds=[0, 1], episodes_per_seed=2),
    )


class _FakeEnv:
    """Scripted env: pops (reward, terminated, truncated, info) per step."""

    def __init__(self, script, action_dim: int = 3) -> None:
        self.script = list(script)
        self.robot = SimpleNamespace(action_dim=action_dim)
        self.resets: list[int] = []
        self.closed = False

    def reset(self, seed: int = 0, options=None):
        self.resets.append(seed)
        return {}, {"seed": seed}

    def step(self, action):
        if not self.script:
            return {}, 0.0, False, True, {"success": False}
        reward, terminated, truncated, info = self.script.pop(0)
        return {}, reward, terminated, truncated, info

    def close(self):
        self.closed = True


class _FakeLoaded:
    def __init__(self, env: _FakeEnv, spec: TaskSpec) -> None:
        self.env = env
        self.spec = spec

    def evaluation_episodes(self):
        return self.spec.evaluation.episodes()

    def reset_episode(self, seed: int):
        return self.env.reset(seed=seed)


def _record(task="t", seed=0, episode=0, status="success", steps=5, reward=1.0):
    return EpisodeRecord(
        task=task, seed=seed, episode=episode, status=status, steps=steps, reward=reward
    )


class TestPolicies:
    """Tests for the built-in benchmark policies."""

    def test_zero_policy(self):
        """ZeroPolicy emits zero actions with the right dimension."""
        policy = ZeroPolicy(action_dim=4)
        action = policy.act({})
        assert action.shape == (4,)
        assert np.all(action == 0.0)

    def test_random_policy_deterministic_per_seed(self):
        """The same seed replays the same action sequence."""
        first = RandomPolicy(action_dim=3, seed=42)
        second = RandomPolicy(action_dim=3, seed=42)
        for _ in range(5):
            np.testing.assert_allclose(first.act({}), second.act({}))

    def test_random_policy_differs_across_seeds(self):
        """Different seeds produce different action sequences."""
        first = RandomPolicy(action_dim=3, seed=1)
        second = RandomPolicy(action_dim=3, seed=2)
        actions_first = np.concatenate([first.act({}) for _ in range(4)])
        actions_second = np.concatenate([second.act({}) for _ in range(4)])
        assert not np.allclose(actions_first, actions_second)

    def test_unknown_policy(self):
        """Unknown policy kinds raise ConfigError."""
        with pytest.raises(ConfigError, match="unknown policy"):
            make_policy("oracle", action_dim=1, seed=0)


class TestRunEpisode:
    """Tests for the single-episode rollout loop."""

    def test_successful_episode(self):
        """Success info maps to status 'success' with summed rewards."""
        env = _FakeEnv(
            [
                (0.1, False, False, {"success": False}),
                (1.0, True, False, {"success": True}),
            ]
        )
        loaded = _FakeLoaded(env, _spec())
        record = run_episode(loaded, seed=7, episode=0, policy=ZeroPolicy(3))
        assert record.status == "success"
        assert record.steps == 2
        assert record.reward == pytest.approx(1.1)
        assert env.resets == [7]

    def test_failure_stage_reported(self):
        """A failing episode reports the failure stage from info."""
        env = _FakeEnv(
            [(0.0, True, False, {"success": False, "failure_stage": "place_fail"})]
        )
        loaded = _FakeLoaded(env, _spec())
        record = run_episode(loaded, seed=0, episode=1, policy=ZeroPolicy(3))
        assert record.status == "place_fail"
        assert record.failure_stage == "place_fail"

    def test_error_when_no_failure_stage(self):
        """Failure without a stage maps to 'error'."""
        env = _FakeEnv([(0.0, True, False, {"success": False})])
        loaded = _FakeLoaded(env, _spec())
        record = run_episode(loaded, seed=0, episode=0, policy=ZeroPolicy(3))
        assert record.status == "error"

    def test_step_cap_guards_infinite_rollouts(self):
        """Episodes that never terminate hit the cap and report 'error'."""
        env = _FakeEnv([(0.0, False, False, {})] * 100)
        loaded = _FakeLoaded(env, _spec(max_steps=5))
        record = run_episode(loaded, seed=0, episode=0, policy=ZeroPolicy(3))
        assert record.status == "error"
        assert record.steps == 5

    def test_trajectory_sink_collects_actions_and_rewards(self):
        """A trajectory sink receives one action/reward pair per step."""
        env = _FakeEnv(
            [
                (0.5, False, False, {"success": False}),
                (1.0, True, False, {"success": True}),
            ]
        )
        loaded = _FakeLoaded(env, _spec())
        trajectory = EpisodeTrajectory()
        record = run_episode(
            loaded, seed=0, episode=0, policy=ZeroPolicy(3), trajectory=trajectory
        )
        assert record.steps == 2
        assert len(trajectory.actions) == 2
        assert len(trajectory.rewards) == 2
        assert trajectory.actions[0] == [0.0, 0.0, 0.0]
        assert trajectory.rewards == [0.5, 1.0]


class TestSummarize:
    """Tests for leaderboard aggregation."""

    def test_per_task_and_overall(self):
        """Success rates, failure stages, and overall stats aggregate."""
        records = [
            _record(task="a", status="success"),
            _record(task="a", status="place_fail", steps=10, reward=-0.5),
            _record(task="b", status="grasp_fail"),
        ]
        summary = summarize(records, suite_name="s")

        assert summary["suite"] == "s"
        task_a = summary["tasks"]["a"]
        assert task_a["episodes"] == 2
        assert task_a["success"] == 1
        assert task_a["success_rate"] == 0.5
        assert task_a["failures_by_stage"] == {"place_fail": 1}
        assert task_a["mean_steps"] == 7.5
        task_b = summary["tasks"]["b"]
        assert task_b["failures_by_stage"] == {"grasp_fail": 1}
        assert summary["overall"] == {
            "episodes": 3,
            "success": 1,
            "success_rate": pytest.approx(1 / 3),
        }

    def test_empty_failures_omitted(self):
        """Stages with zero failures are pruned."""
        summary = summarize([_record(status="success")])
        assert summary["tasks"]["t"]["failures_by_stage"] == {}

    def test_no_records(self):
        """An empty run yields a zeroed summary without crashing."""
        summary = summarize([], suite_name="empty")
        assert summary["tasks"] == {}
        assert summary["overall"]["success_rate"] == 0.0


class TestWriteReport:
    """Tests for report writing and byte reproducibility."""

    def test_outputs_written(self, tmp_path):
        """report.json, report.md, and episodes.jsonl are all created."""
        records = [_record(status="success")]
        summary = summarize(records, suite_name="s")
        report_json, report_md, episodes_jsonl = write_report(
            summary, records, tmp_path
        )

        assert (
            report_json.is_file() and report_md.is_file() and episodes_jsonl.is_file()
        )
        content = report_md.read_text(encoding="utf-8")
        assert "t" in content and "100.0%" in content
        lines = episodes_jsonl.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert '"status": "success"' in lines[0]

    def test_report_json_ignores_nondeterministic_fields(self, tmp_path):
        """Records differing only in wall-clock duration give identical report.json."""
        base = [_record(status="success")]
        first_dir, second_dir = tmp_path / "a", tmp_path / "b"
        summary = summarize(base, suite_name="s")
        write_report(summary, base, first_dir)

        varied = [EpisodeRecord(**{**base[0].__dict__, "duration_s": 999.9})]
        write_report(summary, varied, second_dir)

        assert (first_dir / "report.json").read_bytes() == (
            second_dir / "report.json"
        ).read_bytes()
        assert (first_dir / "episodes.jsonl").read_bytes() != (
            second_dir / "episodes.jsonl"
        ).read_bytes()


class TestLoadSuite:
    """Tests for benchmark suite loading."""

    def test_suite_loads_and_resolves_paths(self, tmp_path):
        """Relative task paths resolve against the suite file location."""
        task_file = tmp_path / "task.yaml"
        task_file.write_text("x", encoding="utf-8")
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(
            "version: 1\nname: demo\ntasks:\n  - task.yaml\n", encoding="utf-8"
        )

        suite = load_suite(suite_file)

        assert suite.name == "demo"
        assert suite.task_paths == [task_file]

    def test_missing_suite(self, tmp_path):
        """A missing suite file raises ConfigError."""
        with pytest.raises(ConfigError, match="not found"):
            load_suite(tmp_path / "nope.yaml")

    @pytest.mark.parametrize(
        "content,match",
        [
            ("tasks: [t.yaml]\n", "name"),  # missing name
            ("name: s\n", "tasks"),  # missing tasks
            ("name: s\ntasks: [missing.yaml]\n", "not found"),  # bad task path
            ("name: s\ntasks: [42]\n", "path string"),  # non-string entry
        ],
    )
    def test_invalid_suites(self, tmp_path, content, match):
        """Malformed suites raise descriptive ConfigErrors."""
        (tmp_path / "t.yaml").write_text("x", encoding="utf-8")
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(content, encoding="utf-8")
        with pytest.raises(ConfigError, match=match):
            load_suite(suite_file)


class TestRunBenchmark:
    """End-to-end tests with injected fake tasks."""

    def _suite(self, tmp_path, n_tasks: int = 1) -> Path:
        task_paths = []
        for index in range(n_tasks):
            task_file = tmp_path / f"task_{index}.yaml"
            task_file.write_text("x", encoding="utf-8")
            task_paths.append(f"task_{index}.yaml")
        suite_file = tmp_path / "suite.yaml"
        suite_file.write_text(
            "version: 1\nname: e2e\ntasks:\n"
            + "".join(f"  - {p}\n" for p in task_paths),
            encoding="utf-8",
        )
        return suite_file

    def _factory(self, env: _FakeEnv, spec: TaskSpec):
        return lambda path: _FakeLoaded(env, spec)

    def test_runs_full_grid_and_closes_env(self, tmp_path):
        """Every (seed, episode) pair runs once and the env is closed."""
        spec = _spec()
        env = _FakeEnv([(0.0, True, False, {"success": True})] * 4)
        suite = load_suite(self._suite(tmp_path))

        summary = run_benchmark(
            suite, tmp_path / "out", task_factory=self._factory(env, spec)
        )

        assert env.resets == [0, 0, 1, 1]  # 2 seeds x 2 episodes
        assert env.closed is True
        assert summary["overall"]["episodes"] == 4
        assert summary["overall"]["success"] == 4

    def test_report_json_byte_reproducible(self, tmp_path):
        """Two benchmark runs produce byte-identical report.json."""
        suite = load_suite(self._suite(tmp_path))

        def factory(path):
            return _FakeLoaded(
                _FakeEnv(
                    [
                        (
                            0.0,
                            True,
                            False,
                            {"success": False, "failure_stage": "place_fail"},
                        )
                    ]
                ),
                _spec(),
            )

        run_benchmark(suite, tmp_path / "run_a", task_factory=factory)
        run_benchmark(suite, tmp_path / "run_b", task_factory=factory)

        first = (tmp_path / "run_a" / "report.json").read_bytes()
        second = (tmp_path / "run_b" / "report.json").read_bytes()
        assert first == second

    def test_episodes_limit_truncates_grid(self, tmp_path):
        """The episode cap truncates the evaluation grid."""
        spec = _spec()
        env = _FakeEnv([(0.0, True, False, {"success": True})])
        suite = load_suite(self._suite(tmp_path))

        summary = run_benchmark(
            suite,
            tmp_path / "out",
            episodes_limit=3,
            task_factory=self._factory(env, spec),
        )

        assert env.resets == [0, 0, 1]
        assert summary["overall"]["episodes"] == 3

    def test_save_trajectories_writes_traces_and_paths(self, tmp_path):
        """Trajectory files land in trajectories/ and paths are recorded."""
        suite = load_suite(self._suite(tmp_path))

        def factory(path):
            return _FakeLoaded(
                _FakeEnv([(0.0, True, False, {"success": True})]),
                _spec(),
            )

        run_benchmark(
            suite, tmp_path / "out", save_trajectories=True, task_factory=factory
        )

        records = (tmp_path / "out" / "episodes.jsonl").read_text(encoding="utf-8")
        assert '"trajectory_path": "trajectories/fake_task_seed0_ep0.json"' in records
        trace_file = tmp_path / "out" / "trajectories" / "fake_task_seed0_ep0.json"
        trace = json.loads(trace_file.read_text(encoding="utf-8"))
        assert trace["task"] == "fake_task"
        assert trace["actions"] == [[0.0, 0.0, 0.0]]
        assert trace["rewards"] == [0.0]
