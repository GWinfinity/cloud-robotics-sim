"""Mock VR client: drives the vr_bridge plugin without any headset.

Modes:
    script    built-in circular motion with the clutch engaged (default)
    keyboard  WASD/QE move, Space clutch, T trigger, A/B buttons,
              M estop toggle, ESC quit
    replay    replay an input stream recorded with --record

Recording: pass --record FILE in script/keyboard mode to capture the
input stream; replay it later with --mode replay --replay-file FILE.
This is the primary tool for reproducing field issues locally.

Usage:
    python mock_client.py --mode script
    python mock_client.py --mode keyboard --record session.jsonl
    python mock_client.py --mode replay --replay-file session.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.messages import PROTOCOL_VERSION, encode_message  # noqa: E402


def now_ms() -> int:
    """Client-side monotonic clock in milliseconds."""
    return int(time.monotonic() * 1000)


class MockClient:
    """Speaks protocol v1 against a running VRBridge."""

    def __init__(self, host: str, state_port: int, control_port: int) -> None:
        self.host = host
        self.state_port = state_port
        self.control_port = control_port
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.tcp: socket.socket | None = None
        self.seq = 0
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self.tcp = socket.create_connection((self.host, self.control_port))
        self.tcp.sendall(
            encode_message(
                {
                    "type": "hello",
                    "protocol_version": PROTOCOL_VERSION,
                    "device": "mock_client",
                    "client_time_ms": now_ms(),
                }
            )
        )
        reply = self._read_line()
        if reply.get("type") != "welcome":
            raise RuntimeError(f"handshake rejected: {reply}")
        print(f"[client] connected, session={reply['session_id']}")
        threading.Thread(target=self._recv_loop, daemon=True).start()

    def close(self) -> None:
        self._stop.set()
        if self.tcp is not None:
            try:
                self.tcp.sendall(encode_message({"type": "bye"}))
                self.tcp.close()
            except OSError:
                pass
        self.udp.close()

    def _read_line(self) -> dict:
        assert self.tcp is not None
        data = b""
        while b"\n" not in data:
            chunk = self.tcp.recv(4096)
            if not chunk:
                raise ConnectionError("server closed the control channel")
            data += chunk
        return json.loads(data.split(b"\n", 1)[0])

    def _recv_loop(self) -> None:
        assert self.tcp is not None
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = self.tcp.recv(4096)
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    print(f"[server] {line.decode().strip()}")

    # ------------------------------------------------------------------
    # sending
    # ------------------------------------------------------------------

    def send_state(self, state: dict) -> None:
        state.update({"type": "state", "seq": self.seq, "client_time_ms": now_ms()})
        self.seq += 1
        self.udp.sendto(json.dumps(state).encode(), (self.host, self.state_port))

    def send_event(self, event: str, pressed: bool = True) -> None:
        assert self.tcp is not None
        self.tcp.sendall(
            encode_message(
                {
                    "type": "event",
                    "event": event,
                    "pressed": pressed,
                    "client_time_ms": now_ms(),
                }
            )
        )


def base_state() -> dict:
    """Neutral pose ~30cm in front of the origin at table height."""
    hand = {
        "pos": [0.3, 0.0, 0.5],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "trigger": 0.0,
        "grip": 0.0,
        "thumbstick": [0.0, 0.0],
    }
    return {"left": dict(hand), "right": dict(hand)}


# ----------------------------------------------------------------------
# input modes
# ----------------------------------------------------------------------


def script_frames(t: float) -> tuple[dict, list[tuple[str, bool]]]:
    """Circular right-hand motion, clutch always on, trigger sinusoid."""
    state = base_state()
    r = 0.08
    state["right"]["pos"] = [0.3 + r * math.cos(t), r * math.sin(t), 0.5]
    state["right"]["grip"] = 1.0
    state["right"]["trigger"] = 0.5 + 0.5 * math.sin(0.5 * t)
    events: list[tuple[str, bool]] = []
    return state, events


KEY_HELP = """\
keyboard mode:
  W/S  move +x/-x      A/D  move +y/-y      R/F  move +z/-z
  Space  clutch (grip) toggle     T  trigger toggle
  A  reset_episode    B  record_toggle    M  estop toggle    ESC quit
"""


def keyboard_mode(client: MockClient, rate: float, record) -> None:
    """Interactive keyboard teleop (Windows msvcrt / POSIX termios)."""
    try:
        import msvcrt  # Windows

        getch = lambda: (msvcrt.getwch() if msvcrt.kbhit() else None)  # noqa: E731
    except ImportError:
        print("keyboard mode requires Windows msvcrt; use --mode script")
        return

    print(KEY_HELP)
    state = base_state()
    step = 0.01
    dt = 1.0 / rate
    grip_on = False
    trigger_on = False
    estop_on = False
    t0 = time.monotonic()
    while True:
        key = getch()
        events: list[tuple[str, bool]] = []
        if key == "\x1b":
            break
        elif key == "w":
            state["right"]["pos"][0] += step
        elif key == "s":
            state["right"]["pos"][0] -= step
        elif key == "a":
            state["right"]["pos"][1] += step
        elif key == "d":
            state["right"]["pos"][1] -= step
        elif key == "r":
            state["right"]["pos"][2] += step
        elif key == "f":
            state["right"]["pos"][2] -= step
        elif key == " ":
            grip_on = not grip_on
            state["right"]["grip"] = 1.0 if grip_on else 0.0
            print(f"[client] clutch {'ON' if grip_on else 'OFF'}")
        elif key == "t":
            trigger_on = not trigger_on
            state["right"]["trigger"] = 1.0 if trigger_on else 0.0
        elif key == "A":
            events.append(("a", True))
        elif key == "B":
            events.append(("b", True))
        elif key == "M":
            # Protocol v1: estop toggles on each menu press; pressed=false
            # is ignored by the server, so always send pressed=True.
            estop_on = not estop_on
            events.append(("menu", True))
            print(f"[client] estop {'ENGAGED' if estop_on else 'released'}")
        for name, pressed in events:
            client.send_event(name, pressed)
        client.send_state(state)
        if record is not None:
            record.write(
                json.dumps(
                    {"t": time.monotonic() - t0, "state": state, "events": events}
                )
                + "\n"
            )
        time.sleep(dt)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--state-port", type=int, default=5555)
    parser.add_argument("--control-port", type=int, default=5556)
    parser.add_argument(
        "--mode", choices=["script", "keyboard", "replay"], default="script"
    )
    parser.add_argument("--rate", type=float, default=90.0)
    parser.add_argument("--duration", type=float, default=0.0, help="0 = forever")
    parser.add_argument("--record", type=Path, default=None)
    parser.add_argument("--replay-file", type=Path, default=None)
    args = parser.parse_args()

    client = MockClient(args.host, args.state_port, args.control_port)
    client.connect()
    record = open(args.record, "w", encoding="utf-8") if args.record else None
    dt = 1.0 / args.rate
    t0 = time.monotonic()
    try:
        if args.mode == "keyboard":
            keyboard_mode(client, args.rate, record)
        elif args.mode == "replay":
            if args.replay_file is None:
                raise SystemExit("--mode replay requires --replay-file")
            for line in open(args.replay_file, encoding="utf-8"):
                frame = json.loads(line)
                for name, pressed in frame.get("events", []):
                    client.send_event(name, pressed)
                client.send_state(frame["state"])
                time.sleep(dt)
        else:  # script
            while True:
                t = time.monotonic() - t0
                if args.duration and t > args.duration:
                    break
                state, events = script_frames(t)
                for name, pressed in events:
                    client.send_event(name, pressed)
                client.send_state(state)
                if record is not None:
                    record.write(json.dumps({"t": t, "state": state}) + "\n")
                time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        if record is not None:
            record.close()
        client.close()
        print(f"[client] done, sent {client.seq} state packets")


if __name__ == "__main__":
    main()
