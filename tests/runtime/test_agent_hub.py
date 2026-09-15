"""Tests for the SimHub MCP-style tool surface."""

from __future__ import annotations

import json

import pytest

from cloud_robotics_sim.runtime.agent_hub import SimHub, call_tool, list_tools
from cloud_robotics_sim.runtime.skills import SkillRegistry, TaskExecutor


@pytest.fixture()
def hub() -> SimHub:
    """SimHub over a small two-skill registry."""
    registry = SkillRegistry()

    @registry.register_skill("ping", "Return pong.")
    def _ping(_p: dict) -> dict:
        return {"pong": True}

    @registry.register_skill("fail", "Always fails.")
    def _fail(_p: dict) -> dict:
        raise ValueError("boom")

    return SimHub(executor=TaskExecutor(registry))


class TestListTools:
    """Tests for tool descriptor shape."""

    def test_tool_descriptors(self, hub: SimHub) -> None:
        tools = list_tools(hub)
        names = {t["name"] for t in tools}
        assert {
            "sim.skills.list",
            "sim.skill.describe",
            "sim.skill.run",
            "sim.goal.run",
            "sim.replay.list",
            "sim.components.list",
            "sim.queue.length",
        } == names
        for tool in tools:
            schema = tool["inputSchema"]
            assert schema["type"] == "object"
            for req in schema.get("required", []):
                assert req in schema["properties"]


class TestCallTool:
    """Tests for tool dispatch and result envelopes."""

    def test_skills_list(self, hub: SimHub) -> None:
        result = call_tool(hub, "sim.skills.list", {})
        assert result["isError"] is False
        payload = json.loads(result["content"][0]["text"])
        assert {s["name"] for s in payload["skills"]} == {"ping", "fail"}

    def test_skill_describe(self, hub: SimHub) -> None:
        result = call_tool(hub, "sim.skill.describe", {"name": "ping"})
        payload = json.loads(result["content"][0]["text"])
        assert payload["name"] == "ping"
        assert result["isError"] is False

    def test_skill_run(self, hub: SimHub) -> None:
        result = call_tool(hub, "sim.skill.run", {"name": "ping", "params": {}})
        payload = json.loads(result["content"][0]["text"])
        assert payload["status"] == "ok"
        assert payload["result"] == {"pong": True}

    def test_skill_run_records_replay(self, hub: SimHub) -> None:
        call_tool(hub, "sim.skill.run", {"name": "ping"})
        result = call_tool(hub, "sim.replay.list", {})
        payload = json.loads(result["content"][0]["text"])
        assert len(payload["records"]) == 1
        assert payload["records"][0]["skill"] == "ping"

    def test_skill_error_envelope(self, hub: SimHub) -> None:
        result = call_tool(hub, "sim.skill.run", {"name": "fail"})
        assert result["isError"] is False  # execution itself succeeded; the
        payload = json.loads(result["content"][0]["text"])  # skill failed
        assert payload["status"] == "error"
        assert "boom" in payload["error"]

    def test_unknown_skill_error_envelope(self, hub: SimHub) -> None:
        result = call_tool(hub, "sim.skill.run", {"name": "nope"})
        assert result["isError"] is True
        assert "unknown skill" in result["content"][0]["text"]

    def test_unknown_tool(self, hub: SimHub) -> None:
        result = call_tool(hub, "sim.nope", {})
        assert result["isError"] is True
        assert "unknown tool" in result["content"][0]["text"]

    def test_goal_run(self, hub: SimHub) -> None:
        result = call_tool(hub, "sim.goal.run", {"goal": "ping please"})
        assert result["isError"] is False
        payload = json.loads(result["content"][0]["text"])
        assert payload["skill"] == "ping"
        assert payload["status"] == "ok"

    def test_goal_no_match_error(self, hub: SimHub) -> None:
        result = call_tool(hub, "sim.goal.run", {"goal": "zzzqqq"})
        assert result["isError"] is True
        assert "no skill matches" in result["content"][0]["text"]

    def test_queue_length_requires_redis_or_connection(
        self, hub: SimHub, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("redis")
        import redis as redis_pkg

        class FakeClient:
            def __init__(self) -> None:
                self.asked: list[str] = []

            def llen(self, queue: str) -> int:
                self.asked.append(queue)
                return 7

        fake = FakeClient()
        monkeypatch.setattr(
            redis_pkg.Redis, "from_url", classmethod(lambda cls, url: fake)
        )
        result = call_tool(hub, "sim.queue.length", {"queue": "sim-tasks-cpu"})
        assert result["isError"] is False
        payload = json.loads(result["content"][0]["text"])
        assert payload == {"queue": "sim-tasks-cpu", "length": 7}
