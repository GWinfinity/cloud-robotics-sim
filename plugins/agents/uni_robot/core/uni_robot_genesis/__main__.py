"""``python -m uni_robot_genesis`` 入口。

子命令：
- ``serve``：宿主端，把 SceneSpec 场景起成环境服务（HTTP）。
- ``agent``：agent 侧，注册 genesis 集成后转交官方 CLI。
"""

from __future__ import annotations

import sys


def main() -> int:
    """分发 serve / agent 两个子命令。"""
    if len(sys.argv) < 2 or sys.argv[1] not in {"serve", "agent"}:
        print("usage: python -m uni_robot_genesis {serve|agent} ...", file=sys.stderr)
        return 2
    command, rest = sys.argv[1], sys.argv[2:]
    if command == "serve":
        from .host import serve_main

        return serve_main(rest)
    from .integration import agent_main

    return agent_main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
