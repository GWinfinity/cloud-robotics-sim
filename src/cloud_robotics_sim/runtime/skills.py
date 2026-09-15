"""Skill registry, task executor, and replay for agent-facing simulation.

This module makes the simulation platform drivable by agents (LLM or
otherwise):

* :class:`Skill` — a named, self-describing capability with a JSON-Schema
  parameter spec and a handler.
* :class:`SkillRegistry` — discover/validate/execute skills.
* :class:`TaskExecutor` — executes skills and records every execution.
* :class:`ReplayBuffer` — in-memory (optionally JSONL-backed) record of
  executions so agent runs can be inspected and replayed.

Built-in skills cover the currently headless-runnable surface: classic
patent simulations, core component discovery, and queue submission for
the Kubernetes/KEDA deployment.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

SkillHandler = Callable[[dict[str, Any]], dict[str, Any]]

_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
}


class SkillError(Exception):
    """Raised for unknown skills, invalid parameters, or handler failures."""


@dataclass
class Skill:
    """A named agent-callable simulation capability."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: SkillHandler


def _validate_params(schema: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    props: dict[str, Any] = schema.get("properties", {})
    required = schema.get("required", [])
    unknown = [k for k in params if k not in props]
    if unknown:
        raise SkillError(f"unknown parameters: {sorted(unknown)}")
    missing = [k for k in required if k not in params]
    if missing:
        raise SkillError(f"missing required parameters: {sorted(missing)}")
    coerced = dict(params)
    for key, value in coerced.items():
        spec = props.get(key, {})
        expected = _TYPE_MAP.get(spec.get("type"))
        if expected is not None and not isinstance(value, expected):
            raise SkillError(
                f"parameter {key!r} must be of type {spec.get('type')}, "
                f"got {type(value).__name__}"
            )
    return coerced


class SkillRegistry:
    """Registry of :class:`Skill` objects with validation and dispatch."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> Skill:
        if skill.name in self._skills:
            logger.warning("overwriting existing skill: %s", skill.name)
        self._skills[skill.name] = skill
        return skill

    def register_skill(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
    ) -> Callable[[SkillHandler], SkillHandler]:
        """Decorator registering a function as a skill."""

        def decorator(handler: SkillHandler) -> SkillHandler:
            self.register(
                Skill(
                    name=name,
                    description=description,
                    parameters=parameters or {"type": "object", "properties": {}},
                    handler=handler,
                )
            )
            return handler

        return decorator

    def get(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError:
            raise SkillError(
                f"unknown skill {name!r}; available: {sorted(self._skills)}"
            ) from None

    def list_skills(self) -> list[dict[str, Any]]:
        """Return skill descriptors (name/description/parameters)."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters,
            }
            for s in self._skills.values()
        ]

    def execute(
        self, name: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Validate parameters and run a skill handler."""
        skill = self.get(name)
        validated = _validate_params(skill.parameters, dict(params or {}))
        return skill.handler(validated)


@dataclass
class ExecutionRecord:
    """One recorded skill execution (replay buffer entry)."""

    task_id: str
    skill: str
    params: dict[str, Any]
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "skill": self.skill,
            "params": self.params,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class ReplayBuffer:
    """In-memory record of skill executions, optionally mirrored to JSONL."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._records: list[ExecutionRecord] = []

    def append(self, record: ExecutionRecord) -> None:
        self._records.append(record)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def records(self) -> list[ExecutionRecord]:
        return list(self._records)

    def clear(self) -> None:
        self._records.clear()

    def __len__(self) -> int:
        """Number of recorded executions."""
        return len(self._records)


_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class TaskExecutor:
    """Executes skills and records executions to a replay buffer.

    Also provides deterministic goal resolution: natural-language goals are
    matched against skill names/descriptions by token overlap (no LLM
    required; an LLM planner can sit on top via the MCP hub). Registered
    per-skill extractors pull structured parameters (e.g. a patent ID)
    out of the goal text.
    """

    def __init__(
        self,
        registry: SkillRegistry | None = None,
        replay: ReplayBuffer | None = None,
        extractors: dict[str, Callable[[str], dict[str, Any]]] | None = None,
    ) -> None:
        self.registry = registry or default_skill_registry()
        self.replay = replay or ReplayBuffer()
        self.extractors = (
            dict(extractors) if extractors is not None else dict(_DEFAULT_EXTRACTORS)
        )

    def execute(
        self, skill_name: str, params: dict[str, Any] | None = None
    ) -> ExecutionRecord:
        """Execute a skill, recording the outcome."""
        started = time.time()
        record = ExecutionRecord(
            task_id=uuid.uuid4().hex,
            skill=skill_name,
            params=dict(params or {}),
            status="ok",
            started_at=started,
        )
        try:
            record.result = self.registry.execute(skill_name, params)
        except Exception as exc:  # noqa: BLE001 - record, do not crash the agent loop
            record.status = "error"
            record.error = f"{type(exc).__name__}: {exc}"
            logger.exception("skill %s failed", skill_name)
        record.finished_at = time.time()
        self.replay.append(record)
        return record

    def resolve_goal(self, goal: str) -> list[tuple[Skill, int]]:
        """Score all skills against a natural-language goal, best first."""
        goal_tokens = _tokens(goal)
        if not goal_tokens:
            return []
        scored: list[tuple[Skill, int]] = []
        for skill in [self.registry.get(n) for n in self._skill_names()]:
            name_tokens = set(_tokens(skill.name.replace("_", " ")))
            desc_tokens = set(_tokens(skill.description))
            score = sum(
                3 if t in name_tokens else 1 if t in desc_tokens else 0
                for t in goal_tokens
            )
            if score > 0:
                scored.append((skill, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def execute_goal(self, goal: str) -> ExecutionRecord:
        """Resolve a goal to the best-matching skill and execute it."""
        matches = self.resolve_goal(goal)
        if not matches:
            raise SkillError(
                f"no skill matches goal {goal!r}; "
                f"available: {sorted(self._skill_names())}"
            )
        skill, score = matches[0]
        params = self.extractors.get(skill.name, lambda _goal: {})(goal)
        logger.info(
            "goal %r resolved to skill %r (score=%d, params=%s)",
            goal,
            skill.name,
            score,
            params,
        )
        return self.execute(skill.name, params)

    def _skill_names(self) -> list[str]:
        return [s["name"] for s in self.registry.list_skills()]


# ---------------------------------------------------------------------------
# Built-in skills
# ---------------------------------------------------------------------------


def _extract_patent_id(goal: str) -> dict[str, Any]:
    """Pull a US patent ID (e.g. US821393) out of a goal string."""
    match = re.search(r"US\s?\d{4,}", goal, re.IGNORECASE)
    return {"run": match.group(0).replace(" ", "").upper()} if match else {}


_DEFAULT_EXTRACTORS: dict[str, Callable[[str], dict[str, Any]]] = {
    "run_patent": _extract_patent_id,
}


def run_patent_task(params: dict[str, Any]) -> dict[str, Any]:
    """Run a classic patent simulation headlessly (shared with queue worker)."""
    from cloud_robotics_sim.patents import run_patent_simulation

    params = dict(params)
    patent_id = params.pop("run", None) or params.pop("patent_id", None)
    if not isinstance(patent_id, str) or not patent_id:
        raise SkillError("patent task requires params.run (patent id)")
    allowed = {
        "steps",
        "dt",
        "substeps",
        "resolution",
        "device",
        "seed",
        "record_path",
        "follow",
    }
    unknown = set(params) - allowed
    if unknown:
        raise SkillError(f"unknown patent params: {sorted(unknown)}")
    resolution = params.get("resolution")
    if resolution is not None:
        params["resolution"] = tuple(resolution)
    follow = params.pop("follow", None)
    if follow is not None:
        params["follow_entity"] = follow
    state = run_patent_simulation(patent_id, headless=True, **params)
    return {
        "time": state.time,
        "parameters": state.parameters,
        "metrics": state.metrics,
    }


def _component_names(kind: str) -> dict[str, Any]:
    """List registered core components, ensuring defaults are registered."""
    import cloud_robotics_sim.runtime.main  # noqa: F401 - triggers registration
    from cloud_robotics_sim.core.registry import default_registry

    registry = default_registry()
    reg = {
        "scenes": registry.scenes,
        "robots": registry.robots,
        "tasks": registry.tasks,
    }[kind]
    return {
        kind: [
            {"name": name, "metadata": reg.get_metadata(name)}
            for name in reg.list_components()
        ]
    }


def _submit_sim_task(params: dict[str, Any]) -> dict[str, Any]:
    """Push a task onto a Redis queue (Kubernetes/KEDA deployment)."""
    try:
        import redis
    except ImportError as exc:
        raise SkillError(
            "submit_sim_task requires the 'redis' package "
            "(pip install cloud-robotics-sim[k8s])"
        ) from exc

    import os

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    task_type = params.get("type")
    queue = params.get("queue", "sim-tasks-cpu")
    task_params = params.get("params", {})
    task = {
        "task_id": uuid.uuid4().hex,
        "type": task_type,
        "params": task_params,
    }
    client = redis.Redis.from_url(str(redis_url))
    client.lpush(str(queue), json.dumps(task))
    return {"task_id": task["task_id"], "queue": queue}


_default_skills: SkillRegistry | None = None


def _list_patents(_params: dict[str, Any]) -> dict[str, Any]:
    from cloud_robotics_sim.patents import list_patents

    return {"patents": list_patents()}


def register_skill(
    name: str,
    description: str,
    parameters: dict[str, Any] | None = None,
) -> Callable[[SkillHandler], SkillHandler]:
    """Decorator registering a function as a skill on the default registry."""
    return default_skill_registry().register_skill(name, description, parameters)


def _make_component_handler(kind: str) -> SkillHandler:
    def _handler(_params: dict[str, Any]) -> dict[str, Any]:
        return _component_names(kind)

    return _handler


def default_skill_registry() -> SkillRegistry:
    """Get the global skill registry with built-in skills registered."""
    global _default_skills
    if _default_skills is None:
        registry = SkillRegistry()
        registry.register(
            Skill(
                name="list_patents",
                description=(
                    "List all registered classic patent simulations "
                    "(e.g. US821393 Wright Flyer, US223898 Edison lamp)."
                ),
                parameters={"type": "object", "properties": {}},
                handler=_list_patents,
            )
        )
        registry.register(
            Skill(
                name="run_patent",
                description=(
                    "Run a classic patent simulation headlessly and return "
                    "its final state metrics."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "run": {
                            "type": "string",
                            "description": "Patent ID, e.g. US821393",
                        },
                        "steps": {"type": "integer"},
                        "dt": {"type": "number"},
                        "device": {"type": "string"},
                        "seed": {"type": "integer"},
                        "record_path": {"type": "string"},
                        "follow": {"type": "string"},
                    },
                    "required": ["run"],
                },
                handler=run_patent_task,
            )
        )
        for kind in ("scenes", "robots", "tasks"):
            registry.register(
                Skill(
                    name=f"list_{kind}",
                    description=f"List registered simulation {kind} components.",
                    parameters={"type": "object", "properties": {}},
                    handler=_make_component_handler(kind),
                )
            )
        registry.register(
            Skill(
                name="submit_sim_task",
                description=(
                    "Submit an asynchronous simulation task to a Redis queue "
                    "consumed by the Kubernetes/KEDA worker pool."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "description": "Task type, e.g. patent",
                        },
                        "queue": {"type": "string"},
                        "params": {"type": "object"},
                    },
                    "required": ["type"],
                },
                handler=_submit_sim_task,
            )
        )
        _default_skills = registry
    return _default_skills
