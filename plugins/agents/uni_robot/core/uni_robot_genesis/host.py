"""把一个 Genesis 场景装配成 uni-robot-agent 的环境服务（HTTP）。

Genesis/Taichi 对调用线程敏感：所有原生操作（构建、步进、渲染、状态
读取）都收敛到一条 owner 线程执行，请求线程通过 Future 等待结果。
这与 uni-robot-agent 自带的 Isaac/RoboDojo 宿主同款模式。

用法（编程方式）::

    host = GenesisEpisodeHost(spec, session_dir="runs/demo")
    url = host.start()          # 打印/传给 agent 侧 --env-url
    ...                          # agent 执行（见 integration.py）
    host.stop()

用法（CLI）::

    python -m uni_robot_genesis serve --spec examples/franka_table.yaml --port 18760
"""

from __future__ import annotations

import argparse
import logging
import threading
from concurrent.futures import Future
from pathlib import Path
from queue import Queue
from typing import Any

logger = logging.getLogger(__name__)

_SENTINEL = object()


class _OwnerThread:
    """单线程执行器：原生句柄的全部操作在同一所有者线程上运行。"""

    def __init__(self, name: str = "genesis-owner") -> None:
        self._queue: Queue[Any] = Queue()
        self._thread: threading.Thread | None = None
        self._name = name
        self._owner_id: int | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name=self._name, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        self._owner_id = threading.get_ident()
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                return
            future, operation, args, kwargs = item
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(operation(*args, **kwargs))
                except BaseException as error:  # noqa: BLE001 - 原样传递给等待方
                    future.set_exception(error)

    def submit(self, operation: Any, *args: Any, **kwargs: Any) -> Future:
        if self._owner_id == threading.get_ident():
            # 已在 owner 线程（如 lifespan 清理）：直接执行避免死锁。
            future: Future = Future()
            try:
                future.set_result(operation(*args, **kwargs))
            except BaseException as error:  # noqa: BLE001
                future.set_exception(error)
            return future
        future = Future()
        self._queue.put((future, operation, args, kwargs))
        return future

    def stop(self, *, timeout_s: float = 10.0) -> None:
        if self._thread is None:
            return
        self._queue.put(_SENTINEL)
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            raise TimeoutError("Genesis owner thread did not stop")
        self._thread = None
        self._owner_id = None


class _OwnedBackend:
    """协议方法的每个调用都被编排到 owner 线程。

    EnvironmentService 直接持有本包装器；快照读取因此也不会与运动竞争
    同一条原生队列之外的锁。
    """

    def __init__(self, owner: _OwnerThread, backend: Any) -> None:
        self._owner, self._backend = owner, backend
        self.description = owner.submit(lambda: backend.description).result()

    def read_state(self) -> Any:
        return self._owner.submit(self._backend.read_state).result()

    def step_pose(self, command: Any) -> Any:
        return self._owner.submit(self._backend.step_pose, command).result()

    def step_joints(self, command: Any) -> Any:
        return self._owner.submit(self._backend.step_joints, command).result()

    def step_native(self, command: Any) -> Any:
        return self._owner.submit(self._backend.step_native, command).result()

    def validate_commands(self, commands: Any, mode: str) -> None:
        return self._owner.submit(
            self._backend.validate_commands, commands, mode
        ).result()

    def preview_joints(self, commands: Any) -> Any:
        return self._owner.submit(self._backend.preview_joints, commands).result()

    def capture(self, cameras: Any = None, *, high_resolution: bool = False) -> Any:
        return self._owner.submit(
            self._backend.capture, cameras, high_resolution=high_resolution
        ).result()

    def finish(self, reason: str, status: str | None = None) -> Any:
        return self._owner.submit(self._backend.finish, reason, status).result()

    def close(self) -> None:
        return self._owner.submit(self._backend.close).result()


class GenesisEpisodeHost:
    """组装 backend + EnvironmentService + HTTP 路由的完整宿主。

    start() 返回环境服务的 base URL；agent 侧用
    ``uni-robot-agent run --integration genesis --env-url <url>`` 连接。
    """

    def __init__(
        self,
        spec: Any,
        *,
        session_dir: str | Path,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.spec = spec
        self.session_dir = Path(session_dir)
        self.host, self.port = host, port
        self._owner = _OwnerThread()
        self._server: Any = None
        self._service: Any = None
        self._owned: Any = None
        self._started = False

    @property
    def url(self) -> str | None:
        return None if self._server is None else self._server.url

    @property
    def service(self) -> Any:
        """底层 EnvironmentService（测试/内嵌用途）。"""
        return self._service

    def start(self) -> str:
        if self._started:
            raise RuntimeError("host is already started")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._owner.start()
        service, owned = self._owner.submit(self._assemble).result()
        self._service = service
        self._owned = owned
        from uni_robot_agent.services.environment.routes import build_router
        from uni_robot_agent.services.hosting import HttpServer, create_app

        router = build_router(service)
        app = create_app(router, on_close=service.close)
        self._server = HttpServer(app, host=self.host, port=self.port)
        url = self._server.start_background()
        self._started = True
        logger.info("genesis environment service ready at %s", url)
        return url

    def _assemble(self) -> tuple[Any, Any]:
        """在 owner 线程上构建场景、backend 与 service。"""
        from uni_robot_agent.artifacts.store import ArtifactStore
        from uni_robot_agent.services.environment.contracts import EpisodeInfo
        from uni_robot_agent.services.environment.service import EnvironmentService

        from .backend import GenesisEnvironmentBackend
        from .builder import build_scene

        built = build_scene(self.spec)
        backend = GenesisEnvironmentBackend(built, episode_id=self.spec.episode_id)
        backend.start()
        info = EpisodeInfo(
            episode_id=self.spec.episode_id,
            task=self.spec.instruction,
            environment=backend.description,
            integration="genesis",
            metadata={"instruction": self.spec.instruction},
        )
        service = EnvironmentService(backend, info, ArtifactStore(self.session_dir))
        service.initialize()
        return service, backend

    def stop(self) -> None:
        try:
            if self._server is not None:
                try:
                    self._server.stop()
                finally:
                    self._server = None
        finally:
            self._owner.stop()
            self._started = False


def serve_main(argv: list[str] | None = None) -> int:
    """宿主端 CLI：加载 spec、起服务、阻塞到 Ctrl+C。"""
    parser = argparse.ArgumentParser(
        prog="uni_robot_genesis.serve",
        description="Host a Genesis scene as a uni-robot-agent environment service.",
    )
    parser.add_argument("--spec", required=True, help="SceneSpec YAML path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--session-dir",
        default=None,
        help="artifact/snapshot output directory (default: ./runs/<spec stem>)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    from .scene_spec import SceneSpec

    spec = SceneSpec.from_yaml(args.spec)
    session_dir = args.session_dir or str(Path("runs") / Path(args.spec).stem)
    host = GenesisEpisodeHost(
        spec, session_dir=session_dir, host=args.host, port=args.port
    )
    url = host.start()
    print(f"environment service: {url}")
    print("agent side, e.g.:")
    print(
        "  UNI_ROBOT_AGENT_API_KEY=... uni-robot-agent run --integration genesis "
        f"--env-url {url} --task {spec.episode_id} --model YOUR_MODEL_ID "
        "--provider-url https://provider.example/v1 --no-sam3"
    )
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        host.stop()
    return 0
