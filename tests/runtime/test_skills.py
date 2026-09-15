"""Tests for the skill registry, task executor, and replay buffer."""

from __future__ import annotations

import pytest

from cloud_robotics_sim.runtime.skills import (
    ExecutionRecord,
    ReplayBuffer,
    Skill,
    SkillError,
    SkillRegistry,
    TaskExecutor,
    default_skill_registry,
    register_skill,
)


class TestSkillRegistry:
    """Tests for SkillRegistry validation and dispatch."""

    def test_register_and_execute(self) -> None:
        registry = SkillRegistry()

        @registry.register_skill(
            "add",
            "Add two numbers.",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        )
        def _add(params: dict) -> dict:
            return {"sum": params["a"] + params["b"]}

        assert registry.execute("add", {"a": 1, "b": 2}) == {"sum": 3}

    def test_missing_required_param(self) -> None:
        registry = SkillRegistry()
        registry.register(
            Skill(
                name="need_arg",
                description="",
                parameters={
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
                handler=lambda p: {"x": p["x"]},
            )
        )
        with pytest.raises(SkillError, match="missing required"):
            registry.execute("need_arg", {})

    def test_unknown_param_rejected(self) -> None:
        registry = SkillRegistry()

        @registry.register_skill("noop", "No parameters.")
        def _noop(_params: dict) -> dict:
            return {}

        with pytest.raises(SkillError, match="unknown parameters"):
            registry.execute("noop", {"hack": 1})

    def test_wrong_type_rejected(self) -> None:
        registry = SkillRegistry()
        registry.register(
            Skill(
                name="typed",
                description="",
                parameters={
                    "type": "object",
                    "properties": {"n": {"type": "integer"}},
                },
                handler=lambda p: {"n": p["n"]},
            )
        )
        with pytest.raises(SkillError, match="must be of type"):
            registry.execute("typed", {"n": "not-a-number"})

    def test_unknown_skill(self) -> None:
        registry = SkillRegistry()
        with pytest.raises(SkillError, match="unknown skill"):
            registry.get("nope")

    def test_list_skills(self) -> None:
        registry = SkillRegistry()

        @registry.register_skill("a", "first")
        def _a(_p: dict) -> dict:
            return {}

        names = [s["name"] for s in registry.list_skills()]
        assert names == ["a"]


class TestReplayBuffer:
    """Tests for ReplayBuffer."""

    def test_append_and_records(self, tmp_path) -> None:
        path = tmp_path / "replay.jsonl"
        replay = ReplayBuffer(path)
        record = ExecutionRecord(
            task_id="t1", skill="s", params={}, status="ok", started_at=1.0
        )
        replay.append(record)
        assert len(replay) == 1
        assert replay.records()[0].task_id == "t1"
        assert '"task_id": "t1"' in path.read_text(encoding="utf-8")

    def test_in_memory_only(self) -> None:
        replay = ReplayBuffer()
        replay.append(ExecutionRecord(task_id="t", skill="s", params={}, status="ok"))
        assert len(replay) == 1
        replay.clear()
        assert len(replay) == 0


class TestTaskExecutor:
    """Tests for TaskExecutor execution recording and goal resolution."""

    def _make_executor(self) -> TaskExecutor:
        registry = SkillRegistry()

        @registry.register_skill(
            "run_patent",
            "Run a patent simulation headlessly.",
            parameters={
                "type": "object",
                "properties": {"run": {"type": "string"}},
            },
        )
        def _run(_p: dict) -> dict:
            return {"ok": True}

        @registry.register_skill("list_patents", "List patents.")
        def _list(_p: dict) -> dict:
            return {"patents": []}

        return TaskExecutor(registry)

    def test_execute_records_success(self) -> None:
        executor = self._make_executor()
        record = executor.execute("run_patent", {"run": "US821393"})
        assert record.status == "ok"
        assert record.result == {"ok": True}
        assert record.error is None
        assert len(executor.replay) == 1

    def test_execute_records_failure(self) -> None:
        executor = self._make_executor()
        record = executor.execute("unknown_skill")
        assert record.status == "error"
        assert record.error is not None
        assert len(executor.replay) == 1

    def test_resolve_goal_prefers_name_match(self) -> None:
        executor = self._make_executor()
        matches = executor.resolve_goal("run the patent simulation")
        assert matches
        assert matches[0][0].name == "run_patent"

    def test_resolve_goal_empty(self) -> None:
        executor = self._make_executor()
        assert executor.resolve_goal("!!!") == []

    def test_execute_goal_runs_best_match(self) -> None:
        executor = self._make_executor()
        record = executor.execute_goal("please run patent US821393")
        assert record.skill == "run_patent"
        assert record.status == "ok"

    def test_execute_goal_no_match_raises(self) -> None:
        executor = self._make_executor()
        with pytest.raises(SkillError, match="no skill matches"):
            executor.execute_goal("zzzqqq")


class TestBuiltinSkills:
    """Tests for the built-in default skill registry."""

    def test_builtin_skills_listed(self) -> None:
        registry = default_skill_registry()
        names = {s["name"] for s in registry.list_skills()}
        assert {
            "list_patents",
            "run_patent",
            "list_scenes",
            "list_robots",
            "list_tasks",
            "submit_sim_task",
        } <= names

    def test_run_patent_skill(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cloud_robotics_sim.patents as patents

        class FakeState:
            time = 3.0
            parameters: dict = {}
            metrics: dict = {"altitude": 1.0}

        calls: dict = {}

        def fake_run(patent_id: str, **kwargs: object) -> FakeState:
            calls["patent_id"] = patent_id
            calls.update(kwargs)
            return FakeState()

        monkeypatch.setattr(patents, "run_patent_simulation", fake_run)
        result = default_skill_registry().execute(
            "run_patent", {"run": "US821393", "steps": 10, "follow": "aircraft"}
        )
        assert calls["patent_id"] == "US821393"
        assert calls["headless"] is True
        assert calls["steps"] == 10
        assert calls["follow_entity"] == "aircraft"
        assert result["time"] == 3.0

    def test_run_patent_missing_id(self) -> None:
        with pytest.raises(SkillError, match="missing required"):
            default_skill_registry().execute("run_patent", {})

    def test_list_components_skills(self) -> None:
        registry = default_skill_registry()
        scenes = registry.execute("list_scenes", {})
        robots = registry.execute("list_robots", {})
        tasks = registry.execute("list_tasks", {})
        assert any(s["name"] == "empty_room" for s in scenes["scenes"])
        assert any(r["name"] == "franka_panda" for r in robots["robots"])
        assert any(t["name"] == "pick_place" for t in tasks["tasks"])

    def test_register_skill_decorator_on_default_registry(self) -> None:
        @register_skill("test_ping", "Return pong.")
        def _ping(_p: dict) -> dict:
            return {"pong": True}

        assert default_skill_registry().execute("test_ping", {}) == {"pong": True}
