"""Meshy text/image-to-3D provider (``https://api.meshy.ai/openapi``).

Endpoints (verified against Meshy's official API reference, 2026):

- Text-to-3D: ``POST /v2/text-to-3d`` (``mode=preview``), poll
  ``GET /v2/text-to-3d/{id}``
- Image-to-3D: ``POST /v1/image-to-3d`` (image as public URL or base64 data
  URI), poll ``GET /v1/image-to-3d/{id}``

Credentials: ``MESHY_API_KEY`` (Bearer, key shape ``msy_...``).
"""

from __future__ import annotations

import os
from typing import Any

from .base import (
    AssetGenError,
    GenerationRequest,
    TaskHandle,
    get_json,
    image_to_data_uri,
    post_json,
)

BASE_URL = "https://api.meshy.ai/openapi"

_RUNNING = {"PENDING", "IN_PROGRESS"}
_FAILED = {"FAILED", "CANCELED"}


class MeshyProvider:
    """Meshy provider (text/image to 3D with PBR textures)."""

    name = "meshy"
    env_vars: tuple[str, ...] = ("MESHY_API_KEY",)

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key

    # -- credentials ---------------------------------------------------------

    @property
    def api_key(self) -> str:
        return self._api_key or os.environ.get("MESHY_API_KEY", "")

    def configured(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise AssetGenError("MESHY_API_KEY is not set")
        return {"Authorization": f"Bearer {self.api_key}"}

    # -- submit ---------------------------------------------------------------

    def submit(self, request: GenerationRequest) -> TaskHandle:
        if request.is_text:
            endpoint = "/v2/text-to-3d"
            payload: dict[str, Any] = {
                "mode": "preview",
                "prompt": request.prompt,
                "target_formats": ["glb"],
            }
            if request.negative_prompt:
                payload["negative_prompt"] = request.negative_prompt
        else:
            endpoint = "/v1/image-to-3d"
            assert request.image_path is not None
            data_uri, _ = image_to_data_uri(request.image_path)
            payload = {
                "image_url": data_uri,
                "should_texture": request.texture,
                "target_formats": ["glb"],
            }
        if request.topology:
            payload["topology"] = request.topology
        payload.update(request.extra)

        resp = post_json(f"{BASE_URL}{endpoint}", payload, headers=self._headers())
        task_id = resp.get("result")
        if not task_id:
            raise AssetGenError(f"meshy response missing result: {resp}")
        return TaskHandle(
            provider=self.name,
            task_id=task_id,
            kind="text" if request.is_text else "image",
        )

    # -- poll -----------------------------------------------------------------

    def poll(self, handle: TaskHandle) -> tuple[str, str | None]:
        endpoint = "/v2/text-to-3d" if handle.kind == "text" else "/v1/image-to-3d"
        resp = get_json(
            f"{BASE_URL}{endpoint}/{handle.task_id}", headers=self._headers()
        )
        status = str(resp.get("status", ""))
        if status in _RUNNING:
            return "running", None
        if status in _FAILED:
            error = (resp.get("task_error") or {}).get("message") or status
            raise AssetGenError(f"meshy task {handle.task_id} failed: {error}")
        if status == "SUCCEEDED":
            urls = resp.get("model_urls") or {}
            url = urls.get("glb")
            return "succeeded", url
        raise AssetGenError(f"meshy returned unknown status {status!r}")


__all__ = ["MeshyProvider"]
