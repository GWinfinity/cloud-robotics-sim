"""环境服务端到端测试：真实 EnvironmentService + HTTP 路由 + 官方客户端。

场景用 fake（无 genesis）；HTTP 只走 127.0.0.1 loopback，无外部网络。
覆盖 episode 协议的关键行为：info/capture/step 时序/陈旧版本拒绝/finish。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.optional_deps import uni_robot_agent_only

pytestmark = uni_robot_agent_only

_TESTS_DIR = Path(__file__).resolve().parent
_CORE_DIR = _TESTS_DIR.parent / "core"
for _dir in (str(_TESTS_DIR), str(_CORE_DIR)):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)


from uni_robot_agent.artifacts.store import ArtifactStore  # noqa: E402
from uni_robot_agent.services.environment.client import (  # noqa: E402
    EnvironmentClient,
    EnvironmentRequestError,
)
from uni_robot_agent.services.environment.contracts import (  # noqa: E402
    EpisodeInfo,
    FinishRequest,
    JointCommand,
    JointTarget,
    StepRequest,
)
from uni_robot_agent.services.environment.routes import build_router  # noqa: E402
from uni_robot_agent.services.environment.service import (  # noqa: E402
    EnvironmentService,
)
from uni_robot_agent.services.hosting import HttpServer, create_app  # noqa: E402
from uni_robot_genesis.backend import GenesisEnvironmentBackend  # noqa: E402


@pytest.fixture()
def service(built):
    """组装并 initialize 的 EnvironmentService（临时 artifact 目录）。"""
    backend = GenesisEnvironmentBackend(built)
    backend.start()
    info = EpisodeInfo(
        episode_id=built.spec.episode_id,
        task=built.spec.instruction,
        environment=backend.description,
        integration="genesis",
        metadata={"instruction": built.spec.instruction},
    )
    # ArtifactStore root 用 pytest tmp 目录之外会写盘；用内存目录隔离
    import tempfile

    tmp = tempfile.mkdtemp(prefix="uni-robot-genesis-test-")
    svc = EnvironmentService(backend, info, ArtifactStore(tmp))
    svc.initialize()
    return svc


def _joint_step(version: int) -> StepRequest:
    return StepRequest(
        command=JointCommand(
            arms={"arm": JointTarget(joints=(0.0, 0.0, 0.0), gripper=0.5)}
        ),
        expected_state_version=version,
    )


class TestEpisodeProtocol:
    """HTTP 端到端：connect/step/capture/finish 全流。"""

    def test_connect_and_snapshot_flow(self, service):
        server = HttpServer(create_app(build_router(service), on_close=service.close))
        url = server.start_background()
        client = EnvironmentClient(url)
        try:
            info = client.connect()
            assert info.integration == "genesis"
            assert info.episode_id == service.info().episode_id
            assert info.environment.arms[0].name == "arm"
            assert info.metadata["instruction"]

            # initialize() 已发布第 0 号记录；record 可读回
            record = client.record(0)
            assert record["episode_id"] == info.episode_id

            # 一次合法的 joint step：版本 0 -> 1
            reply = client.step(_joint_step(version=0))
            assert reply.execution_status == "completed"
            assert reply.state_after.native_steps == 1

            # 陈旧版本（expected 0）必须被拒，且执行状态为 not_started
            with pytest.raises(EnvironmentRequestError) as excinfo:
                client.step(_joint_step(version=0))
            assert excinfo.value.error.execution_status == "not_started"
            assert "Stale" in str(excinfo.value)

            # 捕获当前快照：manifest 带相机与标定
            from uni_robot_agent.services.environment.contracts import CaptureRequest

            manifest = client.capture(CaptureRequest(high_resolution=True))
            camera = manifest["cameras"]["head"]
            assert camera["intrinsic_K"] is not None
            assert camera["T_world_from_optical"] is not None

            # finish -> outcome（无 judge，task_success 为 None）
            outcome = client.finish(FinishRequest(reason="test_done"))
            assert outcome["outcome"]["task_success"] is None
            assert client.outcome()["outcome"]["stop_reason"] == "test_done"
        finally:
            client.close()
            server.stop()


class TestHostOwnerThread:
    """owner 线程：执行、异常传递、同线程直执行。"""

    def test_owner_thread_marshals_and_stops(self):
        from uni_robot_genesis.host import _OwnerThread

        owner = _OwnerThread()
        owner.start()
        try:
            assert owner.submit(lambda: "hello").result(timeout=5) == "hello"

            def boom():
                raise ValueError("native failure")

            with pytest.raises(ValueError, match="native failure"):
                owner.submit(boom).result(timeout=5)

            # owner 线程内直接执行不经过队列（防死锁路径）
            def nested():
                return owner.submit(lambda: 42).result(timeout=5)

            assert owner.submit(nested).result(timeout=5) == 42
        finally:
            owner.stop()
