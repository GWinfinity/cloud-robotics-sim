"""Shared types, errors, and HTTP helpers for 3D asset generation providers.

This module is provider-agnostic: it defines the request/result dataclasses,
the error hierarchy, a task-polling helper, and thin ``urllib.request``
wrappers (``post_json`` / ``get_json`` / ``download_file``) that provider
implementations build on. The wrappers are module-level functions so tests
can monkeypatch them without touching the network.

All URLs are screened by :func:`validate_url` before any connection is made
(https-only, no userinfo, no non-public resolved addresses, optional host
allowlist) and response bodies are size-capped so a hostile or broken endpoint
cannot exhaust memory or reach internal services.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, NoReturn, Protocol

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1 << 20  # 1 MiB
_MAX_RESPONSE_BYTES = 16 << 20  # 16 MiB — API JSON responses are small
_DEFAULT_MAX_DOWNLOAD_BYTES = 256 << 20  # 256 MiB

# Escape hatches for local testing only.
_ENV_ALLOW_HTTP = "CRS_ASSET_GEN_ALLOW_HTTP"
_ENV_ALLOW_PRIVATE = "CRS_ASSET_GEN_ALLOW_PRIVATE"
_ENV_ALLOWED_HOSTS = "CRS_ASSET_GEN_ALLOWED_HOSTS"
_ENV_MAX_DOWNLOAD_BYTES = "CRS_ASSET_GEN_MAX_BYTES"

_TASK_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


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
# URL safety (SSRF guard)
# ---------------------------------------------------------------------------


def _env_flag(name: str) -> bool:
    """Return True when the environment variable is exactly ``"1"``."""
    return os.environ.get(name, "").strip() == "1"


def _host_in_allowlist(host: str) -> bool:
    """Return True when ``host`` passes the ``CRS_ASSET_GEN_ALLOWED_HOSTS`` check.

    An empty allowlist means "no host restriction". Entries are compared
    case-insensitively; an entry starting with ``"."`` also matches the bare
    parent domain and any subdomain (``.example.com`` covers ``example.com``
    and ``api.example.com``).
    """
    entries = [
        entry.strip().lower().rstrip(".")
        for entry in os.environ.get(_ENV_ALLOWED_HOSTS, "").split(",")
        if entry.strip()
    ]
    if not entries:
        return True
    host = host.lower().rstrip(".")
    for entry in entries:
        if entry.startswith("."):
            if host == entry[1:] or host.endswith(entry):
                return True
        elif host == entry:
            return True
    return False


def _resolve_addresses(host: str) -> list[str]:
    """Return every IP address ``host`` resolves to (A + AAAA)."""
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise AssetGenError(f"cannot resolve host {host!r}: {exc}") from exc
    return [str(info[4][0]) for info in infos]


def validate_url(url: str) -> None:
    """Raise :class:`AssetGenError` unless ``url`` is safe to fetch.

    Rules (SSRF hardening):

    - only ``https``; ``http`` additionally requires ``CRS_ASSET_GEN_ALLOW_HTTP=1``
    - no userinfo (``user:pass@host``) in the URL
    - the host must match ``CRS_ASSET_GEN_ALLOWED_HOSTS`` when that whitelist
      is set (comma-separated hostnames, ``.suffix`` entries allowed)
    - every resolved IP address must be global/public; loopback, private,
      link-local, reserved, multicast and unspecified (IPv4 + IPv6) are
      rejected unless ``CRS_ASSET_GEN_ALLOW_PRIVATE=1``
    """
    parts = urllib.parse.urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme != "https" and not (scheme == "http" and _env_flag(_ENV_ALLOW_HTTP)):
        raise AssetGenError(f"refusing non-https URL: {url!r}")
    if parts.username is not None or parts.password is not None:
        raise AssetGenError(f"refusing URL with embedded credentials: {url!r}")
    host = parts.hostname
    if not host:
        raise AssetGenError(f"URL has no hostname: {url!r}")
    if not _host_in_allowlist(host):
        raise AssetGenError(
            f"host {host!r} is not allowed by {_ENV_ALLOWED_HOSTS}: {url!r}"
        )
    if _env_flag(_ENV_ALLOW_PRIVATE):
        return
    for addr in _resolve_addresses(host):
        try:
            ip = ipaddress.ip_address(addr.split("%", 1)[0])
        except ValueError as exc:
            raise AssetGenError(
                f"host {host!r} resolved to unparseable address {addr!r}"
            ) from exc
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
            or not ip.is_global
        ):
            raise AssetGenError(
                f"refusing to contact host {host!r}: resolves to "
                f"non-public address {addr}"
            )


def _max_download_bytes() -> int:
    """Return the download size cap from ``CRS_ASSET_GEN_MAX_BYTES`` (default 256 MiB)."""
    raw = os.environ.get(_ENV_MAX_DOWNLOAD_BYTES, "").strip()
    if not raw:
        return _DEFAULT_MAX_DOWNLOAD_BYTES
    try:
        value = int(raw)
    except ValueError as exc:
        raise AssetGenError(
            f"{_ENV_MAX_DOWNLOAD_BYTES} must be an integer, got {raw!r}"
        ) from exc
    if value <= 0:
        raise AssetGenError(f"{_ENV_MAX_DOWNLOAD_BYTES} must be positive, got {value}")
    return value


def _declared_content_length(resp: Any) -> int | None:
    """Return the response's ``Content-Length`` as int, or None when absent/invalid."""
    raw = resp.headers.get("Content-Length")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _raise_http_error(url: str, exc: urllib.error.HTTPError) -> NoReturn:
    """Translate an ``HTTPError`` into the project error hierarchy."""
    detail = ""
    try:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
    except Exception:  # noqa: BLE001 - best-effort error detail
        pass
    if exc.code in (401, 403):
        raise AuthenticationError(f"HTTP {exc.code} for {url}: {detail or exc.reason}")
    raise AssetGenError(f"HTTP {exc.code} for {url}: {detail or exc.reason}")


# ---------------------------------------------------------------------------
# Task id safety
# ---------------------------------------------------------------------------


def safe_task_id(raw: object, provider: str) -> str:
    """Return ``raw`` when it is safe to use as a filename component.

    Provider-supplied task ids are untrusted: anything outside
    ``[A-Za-z0-9_-]{1,64}`` (path separators, ``..``, whitespace, ...) is
    replaced with a 16-hex-char sha256 digest of its ``str()`` form so it can
    never escape the staging directory.
    """
    text = str(raw)
    if _TASK_ID_RE.fullmatch(text):
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    logger.warning(
        "%s task id %r is not filename-safe; using digest %s",
        provider,
        text,
        digest,
    )
    return digest


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

    The URL is screened by :func:`validate_url` first; the body is read in
    chunks and rejected beyond ``_MAX_RESPONSE_BYTES``.

    Raises:
        AuthenticationError: On HTTP 401/403.
        AssetGenError: On other HTTP errors, with the provider message.
    """
    validate_url(url)
    req = urllib.request.Request(url, data=body, method=method)
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            chunks: list[bytes] = []
            received = 0
            while True:
                chunk = resp.read(_CHUNK_SIZE)
                if not chunk:
                    break
                received += len(chunk)
                if received > _MAX_RESPONSE_BYTES:
                    raise AssetGenError(
                        f"response body over {_MAX_RESPONSE_BYTES} bytes for {url}"
                    )
                chunks.append(chunk)
            return b"".join(chunks)
    except urllib.error.HTTPError as exc:
        _raise_http_error(url, exc)
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


def download_file(
    url: str,
    dest: str | Path,
    *,
    timeout: float = 120.0,
    max_bytes: int | None = None,
) -> Path:
    """Stream ``url`` to ``dest`` atomically via a temp file.

    The body is written incrementally (never materialized in memory) and
    rejected once it exceeds ``max_bytes`` — which defaults to
    ``CRS_ASSET_GEN_MAX_BYTES`` (256 MiB). A ``Content-Length`` that already
    declares an oversized body is refused before any bytes are written.

    Raises:
        AssetGenError: Unsafe URL, HTTP errors, or an oversized body.
        AuthenticationError: On HTTP 401/403.
    """
    import tempfile

    if max_bytes is None:
        max_bytes = _max_download_bytes()
    validate_url(url)
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".part", dir=str(dest_path.parent))
    tmp = Path(tmp_path)
    try:
        with os.fdopen(tmp_fd, "wb") as fh:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                declared = _declared_content_length(resp)
                if declared is not None and declared > max_bytes:
                    raise AssetGenError(
                        f"download of {url} declares {declared} bytes, "
                        f"over the {max_bytes}-byte limit"
                    )
                received = 0
                while True:
                    chunk = resp.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > max_bytes:
                        raise AssetGenError(
                            f"download of {url} exceeded the {max_bytes}-byte limit"
                        )
                    fh.write(chunk)
        tmp.replace(dest_path)
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        _raise_http_error(url, exc)
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        raise AssetGenError(f"connection error for {url}: {exc.reason}") from exc
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
    "safe_task_id",
    "validate_url",
]
