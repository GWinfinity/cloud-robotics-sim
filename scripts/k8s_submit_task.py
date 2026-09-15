#!/usr/bin/env python
"""Submit a simulation task to a Redis queue (Kubernetes/KEDA deployment).

Pairs with the queue worker (``cloud-robotics-sim worker``) consumed by the
KEDA redis scaler — see deploy/kubernetes/ and docs/guides/kubernetes.md.

Example:
    python scripts/k8s_submit_task.py --queue sim-tasks-cpu \
        --type patent --param run=US821393 --param steps=500
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--redis-url",
        default="redis://localhost:6379/0",
        help="Redis connection URL (default: %(default)s)",
    )
    parser.add_argument(
        "--queue",
        default="sim-tasks-cpu",
        help="Queue (Redis list) to push to (default: %(default)s)",
    )
    parser.add_argument(
        "--type",
        required=True,
        help="Task type, e.g. 'patent'",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Task parameter (repeatable); numbers are parsed automatically",
    )
    args = parser.parse_args(argv)

    params: dict = {}
    for item in args.param:
        if "=" not in item:
            parser.error(f"--param must be KEY=VALUE, got: {item!r}")
        key, value = item.split("=", 1)
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        params[key] = value

    task = {"task_id": uuid.uuid4().hex, "type": args.type, "params": params}

    try:
        import redis
    except ImportError:
        print(
            "error: the 'redis' package is required; "
            "install it with: pip install cloud-robotics-sim[k8s]",
            file=sys.stderr,
        )
        return 1

    client = redis.Redis.from_url(args.redis_url)
    client.lpush(args.queue, json.dumps(task))
    print(f"submitted task {task['task_id']} to {args.queue}: {task}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
