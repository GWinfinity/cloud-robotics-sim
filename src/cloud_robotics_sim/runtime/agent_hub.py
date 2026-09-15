"""MCP exposure layer for the simulation environment (agent tool surface).

Transport-agnostic adapter in the same style as
:mod:`cloud_robotics_sim.devices.mcp_adapter`. ``list_tools()`` returns
MCP tool descriptors (JSON Schema inputs) and ``call_tool()`` dispatches
with MCP-compatible result envelopes, so agents can drive the simulation
platform:

* directly from Python (tests, notebooks);
* from any agent harness that speaks MCP, via :func:`serve_stdio` when
  the optional ``mcp`` package is installed;
* from a plain CLI/JSON-RPC shim without any extra dependency.

Tool surface:

* ``sim.skills.list``      — discover agent-callable simulation skills
* ``sim.skill.describe``   — full parameter schema of one skill
* ``sim.skill.run``        — run a skill with parameters
* ``sim.goal.run``         — resolve a natural-language goal to a skill and run it
* ``sim.replay.list``      — inspect recorded executions (replay buffer)
* ``sim.components.list``  — list registered scenes/robots/tasks
* ``sim.queue.length``     — pending task count on a Redis queue (KEDA deployment)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from .skills import SkillError, SkillRegistry, TaskExecutor, default_skill_registry

logger = logging.getLogger(__name__)


class SimHub:
    """Skill registry + executor behind a single MCP-style tool surface."""

    def __init__(
        self,
        registry: SkillRegistry | None = None,
        executor: TaskExecutor | None = None,
    ) -> None:
        if executor is not None:
            self.executor = executor
        else:
            self.executor = TaskExecutor(registry or default_skill_registry())

    @property
    def registry(self) -> SkillRegistry:
        return self.executor.registry


def list_tools(hub: SimHub) -> list[dict[str, Any]]:
    """Return MCP tool descriptors for the hub."""
    return [
        {
            "name": "sim.skills.list",
            "description": "List all agent-callable simulation skills.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "sim.skill.describe",
            "description": "Show the full parameter schema of one skill.",
            "inputSchema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
        {
            "name": "sim.skill.run",
            "description": "Run a simulation skill with the given parameters.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "params": {"type": "object"},
                },
                "required": ["name"],
            },
        },
        {
            "name": "sim.goal.run",
            "description": (
                "Resolve a natural-language goal to the best-matching skill "
                "and run it (deterministic token matching; no LLM required)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"goal": {"type": "string"}},
                "required": ["goal"],
            },
        },
        {
            "name": "sim.replay.list",
            "description": "List recorded skill executions (the replay buffer).",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "sim.components.list",
            "description": "List registered scenes, robots, or tasks.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["scenes", "robots", "tasks"]}
                },
                "required": ["kind"],
            },
        },
        {
            "name": "sim.queue.length",
            "description": (
                "Pending task count on a Redis queue (Kubernetes/KEDA "
                "deployment; requires the 'redis' package and REDIS_URL)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"queue": {"type": "string"}},
                "required": ["queue"],
            },
        },
    ]


def call_tool(hub: SimHub, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a tool call; returns an MCP-compatible result envelope."""
    try:
        result = _dispatch(hub, name, arguments or {})
        return {
            "content": [
                {"type": "text", "text": json.dumps(result, ensure_ascii=False)}
            ],
            "isError": False,
        }
    except (SkillError, KeyError, ValueError, TypeError, RuntimeError) as exc:
        return {
            "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
            "isError": True,
        }


def _dispatch(hub: SimHub, name: str, args: dict[str, Any]) -> Any:
    if name == "sim.skills.list":
        return {"skills": hub.registry.list_skills()}
    if name == "sim.skill.describe":
        skill = hub.registry.get(str(args["name"]))
        return {
            "name": skill.name,
            "description": skill.description,
            "parameters": skill.parameters,
        }
    if name == "sim.skill.run":
        skill_name = str(args["name"])
        hub.registry.get(skill_name)  # unknown skill -> isError envelope
        record = hub.executor.execute(skill_name, dict(args.get("params") or {}))
        return record.to_dict()
    if name == "sim.goal.run":
        record = hub.executor.execute_goal(str(args["goal"]))
        return record.to_dict()
    if name == "sim.replay.list":
        return {"records": [r.to_dict() for r in hub.executor.replay.records()]}
    if name == "sim.components.list":
        skill_name = f"list_{args['kind']}"
        return hub.registry.execute(skill_name, {})
    if name == "sim.queue.length":
        queue = str(args["queue"])
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError(
                "sim.queue.length requires the 'redis' package "
                "(pip install cloud-robotics-sim[k8s])"
            ) from exc
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        client = redis.Redis.from_url(redis_url)
        return {"queue": queue, "length": client.llen(queue)}
    raise SkillError(f"unknown tool {name!r}")


def serve_stdio(hub: SimHub) -> None:
    r"""Serve the hub over MCP stdio (requires the optional ``mcp`` package).

    Install with ``uv add mcp`` and run::

        python -m cloud_robotics_sim.runtime.agent_hub

    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "serve_stdio requires the 'mcp' package (uv add mcp). "
            "The list_tools/call_tool adapter works without it."
        ) from exc

    server = FastMCP("genesis-cloud-sim")

    @server.tool(name="sim.skills.list", description="List simulation skills.")
    def _skills_list() -> str:  # pragma: no cover
        return json.dumps(_dispatch(hub, "sim.skills.list", {}), ensure_ascii=False)

    @server.tool(name="sim.skill.run", description="Run a simulation skill.")
    def _skill_run(
        name: str, params: dict[str, Any] | None = None
    ) -> str:  # pragma: no cover
        return json.dumps(
            _dispatch(hub, "sim.skill.run", {"name": name, "params": params or {}}),
            ensure_ascii=False,
        )

    @server.tool(name="sim.goal.run", description="Run a natural-language goal.")
    def _goal_run(goal: str) -> str:  # pragma: no cover
        return json.dumps(
            _dispatch(hub, "sim.goal.run", {"goal": goal}), ensure_ascii=False
        )

    server.run()  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    serve_stdio(SimHub())
