"""Redis queue worker for cloud (Kubernetes/KEDA) deployment.

The worker consumes JSON task specs from a Redis list and writes results
back to a Redis results list. It pairs with the KEDA redis scaler
(see ``deploy/kubernetes/``) so worker Deployments scale from zero when
tasks are queued and back to zero when idle.

Task spec format::

    {"task_id": "uuid", "type": "patent", "params": {"run": "US821393"}}

Results are appended as JSON to the ``sim-results`` list::

    {"task_id": ..., "status": "ok"|"error", "result": ..., "error": ...,
     "worker_id": ..., "queue": ..., "finished_at": ...}

The ``redis`` package is an optional dependency; install it with
``pip install cloud-robotics-sim[k8s]``.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_QUEUE = "sim-tasks-cpu"
RESULTS_KEY = "sim-results"
DEFAULT_POLL_INTERVAL = 2.0


class TaskError(Exception):
    """Raised when a task spec is invalid or its execution fails."""


@dataclass
class TaskSpec:
    """Parsed simulation task."""

    task_id: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)


def parse_task(raw: bytes | str) -> TaskSpec:
    """Parse and validate a raw task payload.

    Raises:
        TaskError: If the payload is not valid JSON or misses required
            fields.
    """
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TaskError(f"invalid JSON task payload: {exc}") from exc
    if not isinstance(data, dict):
        raise TaskError("task payload must be a JSON object")
    task_type = data.get("type")
    if not isinstance(task_type, str) or not task_type:
        raise TaskError("task payload missing 'type'")
    params = data.get("params", {})
    if not isinstance(params, dict):
        raise TaskError("'params' must be a JSON object")
    task_id = data.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        task_id = uuid.uuid4().hex
    return TaskSpec(task_id=task_id, type=task_type, params=params)


def _run_patent(params: dict[str, Any]) -> dict[str, Any]:
    """Run a classic patent simulation headlessly."""
    from cloud_robotics_sim.patents import run_patent_simulation

    params = dict(params)
    patent_id = params.pop("run", None) or params.pop("patent_id", None)
    if not isinstance(patent_id, str) or not patent_id:
        raise TaskError("patent task requires params.run (patent id)")
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
        raise TaskError(f"unknown patent params: {sorted(unknown)}")
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


_TASK_HANDLERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "patent": _run_patent,
}


def execute_task(spec: TaskSpec) -> dict[str, Any]:
    """Execute a parsed task and return its result payload.

    Raises:
        TaskError: For unknown task types or invalid parameters.
    """
    handler = _TASK_HANDLERS.get(spec.type)
    if handler is None:
        known = ", ".join(sorted(_TASK_HANDLERS))
        raise TaskError(f"unknown task type: {spec.type!r} (known: {known})")
    return handler(spec.params)


def _get_client(redis_url: str) -> Any:
    try:
        import redis
    except ImportError as exc:
        raise RuntimeError(
            "the 'redis' package is required for the queue worker; "
            "install it with: pip install cloud-robotics-sim[k8s]"
        ) from exc
    return redis.Redis.from_url(redis_url)


def _record_result(
    client: Any,
    *,
    spec: TaskSpec | None,
    status: str,
    worker_id: str,
    queue: str,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    payload = {
        "task_id": spec.task_id if spec else None,
        "type": spec.type if spec else None,
        "status": status,
        "result": result,
        "error": error,
        "worker_id": worker_id,
        "queue": queue,
        "finished_at": time.time(),
    }
    client.rpush(RESULTS_KEY, json.dumps(payload))
    logger.info(
        "task %s finished with status=%s", spec.task_id if spec else "?", status
    )


def run_worker(
    redis_url: str | None = None,
    queue: str | None = None,
    poll_interval: float | None = None,
    worker_id: str | None = None,
    once: bool = False,
    client: Any | None = None,
) -> int:
    """Run the blocking worker loop.

    Args:
        redis_url: Redis connection URL (default: ``REDIS_URL`` env).
        queue: Queue (Redis list) to consume (default: ``QUEUE_NAME`` env
            or ``sim-tasks-cpu``).
        poll_interval: Seconds between polls when the queue is empty
            (default: ``POLL_INTERVAL`` env or 2.0).
        worker_id: Identifier recorded with results (default:
            ``WORKER_ID`` env or hostname).
        once: Process at most one task then return (used by tests).
        client: Pre-built Redis client (tests); a new client is created
            from ``redis_url`` otherwise.

    Returns:
        Process exit code (0 on clean shutdown).
    """
    redis_url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    queue = queue or os.environ.get("QUEUE_NAME", DEFAULT_QUEUE)
    poll_interval = (
        poll_interval
        if poll_interval is not None
        else float(os.environ.get("POLL_INTERVAL", DEFAULT_POLL_INTERVAL))
    )
    worker_id = (
        worker_id or os.environ.get("WORKER_ID") or os.environ.get("HOSTNAME", "worker")
    )

    if client is None:
        client = _get_client(redis_url)

    stopping = False

    def _handle_signal(signum: int, _frame: Any) -> None:
        nonlocal stopping
        logger.info("received signal %s, shutting down", signum)
        stopping = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info(
        "worker %s listening on queue=%s (redis=%s)", worker_id, queue, redis_url
    )
    processed = 0
    while not stopping:
        item = client.brpop(queue, timeout=int(poll_interval))
        if item is None:
            if once:
                break
            continue
        _queue, raw = item
        try:
            spec = parse_task(raw)
        except TaskError as exc:
            logger.error("dropping malformed task: %s", exc)
            _record_result(
                client,
                spec=None,
                status="error",
                worker_id=worker_id,
                queue=queue,
                error=str(exc),
            )
            continue
        try:
            result = execute_task(spec)
        except Exception as exc:  # noqa: BLE001 - worker must not crash on task errors
            logger.exception("task %s failed", spec.task_id)
            _record_result(
                client,
                spec=spec,
                status="error",
                worker_id=worker_id,
                queue=queue,
                error=str(exc),
            )
        else:
            _record_result(
                client,
                spec=spec,
                status="ok",
                worker_id=worker_id,
                queue=queue,
                result=result,
            )
        processed += 1
        if once:
            break

    logger.info("worker %s stopped (processed=%d)", worker_id, processed)
    return 0
