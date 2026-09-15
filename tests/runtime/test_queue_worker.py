"""Tests for the Redis queue worker (mocked redis, no network)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from cloud_robotics_sim.runtime.queue_worker import (
    RESULTS_KEY,
    TaskError,
    TaskSpec,
    execute_task,
    parse_task,
    run_worker,
)


class FakeRedis:
    """Minimal in-memory stand-in for a redis client."""

    def __init__(self, pops: list[Any] | None = None) -> None:
        self.pops = list(pops or [])
        self.pushed: list[tuple[str, str]] = []

    def brpop(self, queue: str, timeout: int) -> Any:
        if not self.pops:
            return None
        return (queue, self.pops.pop(0))

    def rpush(self, key: str, value: str) -> None:
        self.pushed.append((key, value))


def _task_payload(**overrides: Any) -> str:
    payload: dict[str, Any] = {
        "task_id": "t-1",
        "type": "patent",
        "params": {"run": "US821393"},
    }
    payload.update(overrides)
    return json.dumps(payload)


class TestParseTask:
    """Tests for parse_task validation."""

    def test_valid_payload(self) -> None:
        spec = parse_task(_task_payload())
        assert spec == TaskSpec(
            task_id="t-1", type="patent", params={"run": "US821393"}
        )

    def test_bytes_payload(self) -> None:
        spec = parse_task(_task_payload().encode("utf-8"))
        assert spec.type == "patent"

    def test_missing_task_id_gets_uuid(self) -> None:
        spec = parse_task(json.dumps({"type": "patent"}))
        assert spec.task_id
        assert spec.params == {}

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(TaskError, match="invalid JSON"):
            parse_task("{not json")

    def test_non_object_raises(self) -> None:
        with pytest.raises(TaskError, match="JSON object"):
            parse_task("[1, 2]")

    def test_missing_type_raises(self) -> None:
        with pytest.raises(TaskError, match="'type'"):
            parse_task(json.dumps({"params": {}}))

    def test_non_dict_params_raises(self) -> None:
        with pytest.raises(TaskError, match="'params'"):
            parse_task(json.dumps({"type": "patent", "params": [1]}))


class TestExecuteTask:
    """Tests for execute_task dispatch."""

    def test_patent_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import cloud_robotics_sim.patents as patents

        class FakeState:
            time = 5.0
            parameters: dict = {"thrust": 0.8}
            metrics: dict = {"altitude": 42.0}

        calls: dict[str, Any] = {}

        def fake_run(patent_id: str, **kwargs: Any) -> FakeState:
            calls["patent_id"] = patent_id
            calls.update(kwargs)
            return FakeState()

        monkeypatch.setattr(patents, "run_patent_simulation", fake_run)
        spec = parse_task(
            _task_payload(
                params={"run": "US821393", "steps": 100, "follow": "aircraft"}
            )
        )
        result = execute_task(spec)
        assert calls["patent_id"] == "US821393"
        assert calls["headless"] is True
        assert calls["steps"] == 100
        assert calls["follow_entity"] == "aircraft"
        assert result["time"] == 5.0
        assert result["metrics"] == {"altitude": 42.0}

    def test_patent_task_missing_run_raises(self) -> None:
        with pytest.raises(TaskError, match="params.run"):
            execute_task(TaskSpec(task_id="t", type="patent", params={}))

    def test_patent_task_unknown_param_raises(self) -> None:
        with pytest.raises(TaskError, match="unknown patent params"):
            execute_task(
                TaskSpec(
                    task_id="t", type="patent", params={"run": "US821393", "hack": 1}
                )
            )

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(TaskError, match="unknown task type"):
            execute_task(TaskSpec(task_id="t", type="nope", params={}))


class TestRunWorker:
    """Tests for the worker main loop with a fake redis client."""

    def test_processes_task_and_records_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import cloud_robotics_sim.patents as patents

        class FakeState:
            time = 1.0
            parameters: dict = {}
            metrics: dict = {}

        monkeypatch.setattr(
            patents,
            "run_patent_simulation",
            lambda *a, **k: FakeState(),
        )
        client = FakeRedis(pops=[_task_payload()])
        rc = run_worker(
            redis_url="redis://unused:6379/0",
            queue="sim-tasks-cpu",
            poll_interval=0.01,
            worker_id="w-1",
            once=True,
            client=client,
        )
        assert rc == 0
        assert len(client.pushed) == 1
        key, raw = client.pushed[0]
        assert key == RESULTS_KEY
        record = json.loads(raw)
        assert record["status"] == "ok"
        assert record["task_id"] == "t-1"
        assert record["worker_id"] == "w-1"
        assert record["queue"] == "sim-tasks-cpu"
        assert record["result"]["time"] == 1.0

    def test_task_error_recorded_not_raised(self) -> None:
        client = FakeRedis(pops=[json.dumps({"task_id": "bad", "type": "unknown"})])
        rc = run_worker(
            redis_url="redis://unused:6379/0",
            queue="q",
            poll_interval=0.01,
            once=True,
            client=client,
        )
        assert rc == 0
        assert len(client.pushed) == 1
        record = json.loads(client.pushed[0][1])
        assert record["status"] == "error"
        assert record["task_id"] == "bad"
        assert "unknown task type" in record["error"]

    def test_malformed_payload_recorded(self) -> None:
        client = FakeRedis(pops=["{not json"])
        run_worker(
            redis_url="redis://unused:6379/0",
            queue="q",
            poll_interval=0.01,
            once=True,
            client=client,
        )
        record = json.loads(client.pushed[0][1])
        assert record["status"] == "error"
        assert record["task_id"] is None
        assert "invalid JSON" in record["error"]

    def test_empty_queue_returns_when_once(self) -> None:
        client = FakeRedis(pops=[])
        rc = run_worker(
            redis_url="redis://unused:6379/0",
            queue="q",
            poll_interval=0.01,
            once=True,
            client=client,
        )
        assert rc == 0
        assert client.pushed == []
