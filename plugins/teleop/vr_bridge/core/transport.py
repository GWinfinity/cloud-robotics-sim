"""L1 transport layer: UDP state receiver + TCP control channel.

Pure stdlib (``socket`` + ``threading``) so the plugin adds no new
dependencies to the project lockfile. Genesis steps synchronously, so all
network I/O lives on daemon threads and hands data to the control loop
through the mailbox / event queue (see ``sync_buffer``).
"""

from __future__ import annotations

import logging
import socket
import threading

from .messages import (
    ProtocolError,
    decode_control,
    decode_state_datagram,
    encode_message,
)
from .session import SessionManager
from .sync_buffer import EventQueue, StateMailbox, now_ms

logger = logging.getLogger(__name__)

MAX_DGRAM = 65535
MAX_LINE = 65536


class TransportServer:
    """Dual-channel server driven by background daemon threads.

    Args:
        host: Bind address (``"0.0.0.0"`` for LAN clients).
        state_port: UDP port for the state stream (0 = ephemeral).
        control_port: TCP port for the control channel (0 = ephemeral).
        mailbox: Shared latest-value store for controller states.
        events: Shared queue for button events.
        session: Session manager (handshake / single-client mutex).
    """

    def __init__(
        self,
        host: str,
        state_port: int,
        control_port: int,
        mailbox: StateMailbox,
        events: EventQueue,
        session: SessionManager,
    ) -> None:
        self.host = host
        self.mailbox = mailbox
        self.events = events
        self.session = session

        self._udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._udp.bind((host, state_port))
        self._udp.settimeout(0.2)

        self._tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._tcp.bind((host, control_port))
        self._tcp.listen(1)
        self._tcp.settimeout(0.2)

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._client: socket.socket | None = None
        self._client_addr: tuple[str, int] | None = None
        self._client_lock = threading.Lock()

    @property
    def state_port(self) -> int:
        return int(self._udp.getsockname()[1])

    @property
    def control_port(self) -> int:
        return int(self._tcp.getsockname()[1])

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._udp_loop, name="vrbridge-udp", daemon=True),
            threading.Thread(
                target=self._tcp_accept_loop, name="vrbridge-tcp", daemon=True
            ),
        ]
        for t in self._threads:
            t.start()
        logger.info(
            "vr_bridge transport up: udp=%d tcp=%d",
            self.state_port,
            self.control_port,
        )

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)
        self._close_client()
        self._udp.close()
        self._tcp.close()
        self.session.end_session()

    # ------------------------------------------------------------------
    # UDP state stream
    # ------------------------------------------------------------------

    def _udp_loop(self) -> None:
        while not self._stop.is_set():
            try:
                data, addr = self._udp.recvfrom(MAX_DGRAM)
            except socket.timeout:
                continue
            except OSError:
                break
            if not self.session.is_active():
                continue  # ignore state before handshake completes
            with self._client_lock:
                if self._client_addr is not None and addr[0] != self._client_addr[0]:
                    continue  # only the session client may drive the robot
            try:
                state = decode_state_datagram(data)
            except ProtocolError as exc:
                logger.warning("dropping bad state datagram: %s", exc)
                continue
            self.mailbox.put(state, server_recv_ms=now_ms())

    # ------------------------------------------------------------------
    # TCP control channel
    # ------------------------------------------------------------------

    def _tcp_accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, addr = self._tcp.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with self._client_lock:
                busy = self._client is not None
                if not busy:
                    self._client, self._client_addr = conn, addr
            if busy:
                self._send(
                    {
                        "type": "error",
                        "code": "SESSION_BUSY",
                        "message": "another client already controls the robot",
                    },
                    conn,
                )
                conn.close()
                continue
            threading.Thread(
                target=self._client_loop,
                args=(conn,),
                name="vrbridge-client",
                daemon=True,
            ).start()

    def _client_loop(self, conn: socket.socket) -> None:
        conn.settimeout(0.5)
        buffer = b""
        try:
            while not self._stop.is_set():
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if len(line) > MAX_LINE:
                        buffer = b""
                        break
                    self._handle_control_line(line.strip(), conn)
        finally:
            self._close_client(conn)
            self.session.end_session()
            self.mailbox.clear()

    def _handle_control_line(self, line: bytes, conn: socket.socket) -> None:
        if not line:
            return
        try:
            msg = decode_control(line)
        except ProtocolError as exc:
            self._send(
                {"type": "error", "code": "BAD_MESSAGE", "message": str(exc)},
                conn,
            )
            return
        if msg.kind == "hello":
            self._send(self.session.handle_hello(msg.payload), conn)
        elif msg.kind == "ping":
            self._send(
                {
                    "type": "pong",
                    "client_time_ms": msg.payload["client_time_ms"],
                    "server_time_ms": int(now_ms()),
                },
                conn,
            )
        elif msg.kind == "event":
            from .messages import EventMsg

            self.events.put(EventMsg(**msg.payload))
            if msg.payload["event"] == "menu" and msg.payload["pressed"]:
                self._send(
                    {"type": "estop_ack", "server_time_ms": int(now_ms())},
                    conn,
                )
        elif msg.kind == "bye":
            self._close_client(conn)

    def send_to_client(self, msg: dict) -> None:
        """Send a message to the active control client (if any)."""
        with self._client_lock:
            conn = self._client
        if conn is not None:
            self._send(msg, conn)

    def _send(self, msg: dict, conn: socket.socket) -> None:
        try:
            conn.sendall(encode_message(msg))
        except OSError:
            pass

    def _close_client(self, conn: socket.socket | None = None) -> None:
        with self._client_lock:
            target = conn if conn is not None else self._client
            if target is not None and target == self._client:
                self._client, self._client_addr = None, None
        if target is not None:
            try:
                target.close()
            except OSError:
                pass
