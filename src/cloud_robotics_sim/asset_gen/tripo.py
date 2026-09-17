"""Tripo AI (VAST) text/image-to-3D provider.

Uses the Tripo V2 OpenAPI (``https://api.tripo3d.ai/v2/openapi``).

.. note::
   Tripo V2 is maintained until 2026-10-01 and shut down from 2026-11-01;
   the successor is the V3 API (``https://openapi.tripo3d.ai/v3``). Set the
   ``TRIPO_BASE_URL`` environment variable to migrate without code changes.

Credentials: ``TRIPO_API_KEY`` (Bearer, key shape ``tsk_...``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import (
    AssetGenError,
    GenerationRequest,
    TaskHandle,
    encode_multipart,
    get_json,
    post_json,
)

DEFAULT_BASE_URL = "https://api.tripo3d.ai/v2/openapi"

_RUNNING = {"queued", "running"}
_FAILED = {"failed", "banned", "expired", "cancelled", "unknown"}


class TripoProvider:
    """Tripo AI provider (text/image to 3D, V2 OpenAPI)."""

    name = "tripo"
    env_vars: tuple[str, ...] = ("TRIPO_API_KEY",)

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = (
            base_url or os.environ.get("TRIPO_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self._api_key = api_key

    # -- credentials ---------------------------------------------------------

    @property
    def api_key(self) -> str:
        return self._api_key or os.environ.get("TRIPO_API_KEY", "")

    def configured(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise AssetGenError("TRIPO_API_KEY is not set")
        return {"Authorization": f"Bearer {self.api_key}"}

    # -- submit ---------------------------------------------------------------

    def submit(self, request: GenerationRequest) -> TaskHandle:
        payload: dict[str, Any] = {
            "texture": request.texture,
            "pbr": request.texture,
        }
        if request.is_text:
            kind = "text_to_model"
            payload["type"] = kind
            payload["prompt"] = request.prompt
            if request.negative_prompt:
                payload["negative_prompt"] = request.negative_prompt
        else:
            kind = "image_to_model"
            payload["type"] = kind
            assert request.image_path is not None
            payload["file"] = self._upload_image(Path(request.image_path))
        if request.topology == "quad":
            payload["quad"] = True
        payload.update(request.extra)

        resp = post_json(f"{self.base_url}/task", payload, headers=self._headers())
        if resp.get("code") != 0:
            raise AssetGenError(
                f"tripo task creation failed: code={resp.get('code')} msg={resp.get('msg')}"
            )
        task_id = resp.get("data", {}).get("task_id")
        if not task_id:
            raise AssetGenError(f"tripo response missing task_id: {resp}")
        return TaskHandle(
            provider=self.name,
            task_id=task_id,
            kind="text" if request.is_text else "image",
        )

    def _upload_image(self, path: Path) -> dict[str, str]:
        """Upload an image via multipart and return the ``file`` reference."""
        media_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(path.suffix.lower())
        if media_type is None:
            raise AssetGenError(f"tripo: unsupported image type: {path.suffix}")
        body, content_type = encode_multipart(
            {},
            [("file", path.name, path.read_bytes(), media_type)],
        )
        from .base import _request  # noqa: PLC0415 - thin wrapper reuse

        raw = _request(
            "POST",
            f"{self.base_url}/upload/sts",
            headers={**self._headers(), "Content-Type": content_type},
            body=body,
        )
        import json

        resp = json.loads(raw.decode("utf-8"))
        if resp.get("code") != 0:
            raise AssetGenError(
                f"tripo image upload failed: code={resp.get('code')} msg={resp.get('msg')}"
            )
        data = resp.get("data", {})
        file_token = data.get("file_token")
        if not file_token:
            raise AssetGenError(f"tripo upload response missing file_token: {resp}")
        file_type = "jpg" if media_type == "image/jpeg" else media_type.split("/")[-1]
        return {"type": file_type, "file_token": file_token}

    # -- poll -----------------------------------------------------------------

    def poll(self, handle: TaskHandle) -> tuple[str, str | None]:
        resp = get_json(
            f"{self.base_url}/task/{handle.task_id}", headers=self._headers()
        )
        if resp.get("code") != 0:
            raise AssetGenError(
                f"tripo task query failed: code={resp.get('code')} msg={resp.get('msg')}"
            )
        data = resp.get("data", {})
        status = str(data.get("status", "")).lower()
        if status in _RUNNING:
            return "running", None
        if status in _FAILED:
            detail = data.get("error_msg") or resp.get("msg") or status
            raise AssetGenError(f"tripo task {handle.task_id} failed: {detail}")
        if status == "success":
            output = data.get("output", {})
            # Prefer the PBR-textured GLB when available.
            url = (
                output.get("pbr_model")
                or output.get("model")
                or output.get("base_model")
            )
            return "succeeded", url
        raise AssetGenError(f"tripo returned unknown status {status!r}")


__all__ = ["TripoProvider"]
