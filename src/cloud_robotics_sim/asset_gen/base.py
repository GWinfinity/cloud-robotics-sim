"""Shared types, errors, and HTTP helpers for 3D asset generation providers.

This module is provider-agnostic: it defines the request/result dataclasses,
the error hierarchy, a task-polling helper, and thin ``urllib.request``
wrappers (``post_json`` / ``get_json`` / ``download_file``) that provider
implementations build on. The wrappers are module-level functions so tests
can monkeypatch them without touching the network.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1 << 20  # 1 MiB


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AssetGenError(RuntimeError):
    """Base error for all asset generation failures."""


class AuthenticationError(AssetGenError):
    """Missing or rejected API credentials."""


class TaskFailedError(AssetGenError):
    """The provider reported the generation task as failed."""


class TaskTimeoutError(AssetGenError):
    """The generation task did not finish within the polling timeout."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class GenerationRequest:
    """A single 3D asset generation request.

    Exactly one of ``prompt`` (text-to-3D) or ``image_path`` (image-to-3D)
    must be given.

    Args:
        prompt: Text description of the desired asset.
        image_path: Path to a reference image on disk.
        texture: Whether to generate PBR textures (default True).
        negative_prompt: Things to avoid in the output.
        topology: ``"quad"`` or ``"triangle"``; not all providers support it.
        extra: Provider-specific options passed through verbatim.
    """

    prompt: str | None = None
    image_path: str | Path | None = None
    texture: bool = True
    negative_prompt: str | None = None
    topology: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that exactly one input source is given."""
        if (self.prompt is None) == (self.image_path is None):
            raise ValueError("exactly one of 'prompt' or 'image_path' must be given")
        if self.topology is not None and self.topology not in ("quad", "triangle"):
            raise ValueError(
                f"topology must be 'quad' or 'triangle', got {self.topology!r}"
            )
        if self.image_path is not None and not Path(self.image_path).is_file():
            raise FileNotFoundError(f"image not found: {self.image_path}")

    @property
    def is_text(self) -> bool:
        return self.prompt is not None


@dataclass
class GenerationResult:
    """A finished generation: local file plus provenance."""

    provider: str
    task_id: str
    model_file: Path
    source_url: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskHandle:
    """Opaque reference to a submitted provider task.

    ``meta`` lets providers stash provider-specific identifiers (e.g.
    Rodin's ``subscription_key``) needed for polling.
    """

    provider: str
    task_id: str
    kind: str  # "text" or "image"
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------


class Provider(Protocol):
    """Interface implemented by every 3D generation provider."""

    name: str
    env_vars: tuple[str, ...]

    def configured(self) -> bool:
        """Return True if the credentials for this provider are available."""
        ...

    def submit(self, request: GenerationRequest) -> TaskHandle:
        """Submit a generation task to the provider."""
        ...

    def poll(self, handle: TaskHandle) -> tuple[str, str | None]:
        """Return ``(status, model_url)`` for a submitted task.

        ``status`` must be one of ``"running"``, ``"succeeded"``, ``"failed"``.
        """
        ...


def env_configured(env_vars: tuple[str, ...]) -> bool:
    """Return True when every listed environment variable is non-empty."""
    return all(os.environ.get(v) for v in env_vars)


# ---------------------------------------------------------------------------
# Task polling
# ---------------------------------------------------------------------------


def poll_task(
    handle: TaskHandle,
    poll_fn: Callable[[TaskHandle], tuple[str, str | None]],
    *,
    timeout: float = 600.0,
    interval: float = 5.0,
    backoff: float = 1.5,
    max_interval: float = 30.0,
) -> str:
    """Poll ``poll_fn`` until the task succeeds or fails.

    Args:
        handle: The task to wait for.
        poll_fn: Callable returning ``(status, model_url)``.
        timeout: Total seconds to wait before raising :class:`TaskTimeoutError`.
        interval: Initial polling interval in seconds.
        backoff: Multiplier applied after each poll.
        max_interval: Upper bound for the polling interval.

    Returns:
        The model download URL.

    Raises:
        TaskFailedError: The provider reported the task as failed.
        TaskTimeoutError: The task did not finish within ``timeout``.
    """
    deadline = time.monotonic() + timeout
    wait = interval
    while True:
        status, model_url = poll_fn(handle)
        if status == "succeeded":
            if not model_url:
                raise AssetGenError(
                    f"{handle.provider} task {handle.task_id} succeeded "
                    f"without a model URL"
                )
            return model_url
        if status == "failed":
            raise TaskFailedError(f"{handle.provider} task {handle.task_id} failed")
        if status != "running":
            raise AssetGenError(f"{handle.provider} returned unknown status {status!r}")
        if time.monotonic() + wait > deadline:
            raise TaskTimeoutError(
                f"{handle.provider} task {handle.task_id} did not finish "
                f"within {timeout:.0f}s"
            )
        logger.info(
            "%s task %s still running; retrying in %.0fs",
            handle.provider,
            handle.task_id,
            wait,
        )
        time.sleep(wait)
        wait = min(wait * backoff, max_interval)


# ---------------------------------------------------------------------------
# HTTP helpers (module-level so tests can monkeypatch them)
# ---------------------------------------------------------------------------


def _request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    timeout: float = 60.0,
) -> bytes:
    """Send an HTTP request and return the response body.

    Raises:
        AuthenticationError: On HTTP 401/403.
        AssetGenError: On other HTTP errors, with the provider message.
    """
    req = urllib.request.Request(url, data=body, method=method)
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data: bytes = resp.read()
            return data
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:  # noqa: BLE001 - best-effort error detail
            pass
        if exc.code in (401, 403):
            raise AuthenticationError(
                f"HTTP {exc.code} for {url}: {detail or exc.reason}"
            ) from exc
        raise AssetGenError(
            f"HTTP {exc.code} for {url}: {detail or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise AssetGenError(f"connection error for {url}: {exc.reason}") from exc


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """POST a JSON payload and return the decoded JSON response."""
    hdrs = {"Content-Type": "application/json"}
    hdrs.update(headers or {})
    raw = _request(
        "POST",
        url,
        headers=hdrs,
        body=json.dumps(payload).encode("utf-8"),
        timeout=timeout,
    )
    result: dict[str, Any] = json.loads(raw.decode("utf-8"))
    return result


def get_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """GET a URL and return the decoded JSON response."""
    raw = _request("GET", url, headers=headers, timeout=timeout)
    result: dict[str, Any] = json.loads(raw.decode("utf-8"))
    return result


def download_file(url: str, dest: str | Path, *, timeout: float = 120.0) -> Path:
    """Download ``url`` to ``dest`` atomically via a temp file."""
    import tempfile

    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".part", dir=str(dest_path.parent))
    tmp = Path(tmp_path)
    try:
        body = _request("GET", url, timeout=timeout)
        with os.fdopen(tmp_fd, "wb") as fh:
            for i in range(0, len(body), _CHUNK_SIZE):
                fh.write(body[i : i + _CHUNK_SIZE])
        tmp.replace(dest_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return dest_path


def encode_multipart(
    fields: dict[str, str],
    files: list[tuple[str, str, bytes, str]],
) -> tuple[bytes, str]:
    """Encode ``fields`` and ``files`` as multipart/form-data.

    Args:
        fields: Regular form fields.
        files: ``(field_name, filename, content, content_type)`` tuples.

    Returns:
        ``(body, content_type_header)`` ready for a POST request.
    """
    import secrets

    boundary = f"----assetgen{secrets.token_hex(16)}"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )
    for name, filename, content, content_type in files:
        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"; '
            f'filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
        parts.append(header + content + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)
    return body, f"multipart/form-data; boundary={boundary}"


def image_to_data_uri(path: str | Path) -> tuple[str, str]:
    """Read an image file and return ``(data_uri, media_type)``."""
    import base64

    p = Path(path)
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(p.suffix.lower())
    if media_type is None:
        raise AssetGenError(f"unsupported image type: {p.suffix}")
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{data}", media_type


__all__ = [
    "AssetGenError",
    "AuthenticationError",
    "GenerationRequest",
    "GenerationResult",
    "Provider",
    "TaskFailedError",
    "TaskHandle",
    "TaskTimeoutError",
    "download_file",
    "encode_multipart",
    "env_configured",
    "get_json",
    "image_to_data_uri",
    "poll_task",
    "post_json",
]
