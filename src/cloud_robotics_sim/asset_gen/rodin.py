"""Rodin (Hyper3D) image/text-to-3D provider.

Uses the Hyper3D v2 API (``https://api.hyper3d.com/api/v2``):

- Submit: ``POST /api/v2/rodin`` (multipart/form-data; ``prompt`` or
  ``images`` files, plus a ``tier`` such as ``Gen-2.5-Medium``)
- Poll: ``POST /api/v2/status`` (JSON ``{"subscription_key": ...}``)
- Download: ``POST /api/v2/download`` (JSON ``{"task_uuid": ...}``; the
  returned signed URLs expire, so download promptly)

Credentials: ``RODIN_API_KEY`` (Bearer).
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
    post_json,
)

BASE_URL = "https://api.hyper3d.com/api/v2"

_RUNNING = {"Waiting", "Generating"}
_FAILED = {"Failed", "Canceled"}


class RodinProvider:
    """Rodin / Hyper3D provider (high-fidelity generation)."""

    name = "rodin"
    env_vars: tuple[str, ...] = ("RODIN_API_KEY",)

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key

    # -- credentials ---------------------------------------------------------

    @property
    def api_key(self) -> str:
        return self._api_key or os.environ.get("RODIN_API_KEY", "")

    def configured(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise AssetGenError("RODIN_API_KEY is not set")
        return {"Authorization": f"Bearer {self.api_key}"}

    # -- submit ---------------------------------------------------------------

    def submit(self, request: GenerationRequest) -> TaskHandle:
        tier = str(request.extra.get("tier", "Gen-2.5-Medium"))
        fields: dict[str, str] = {
            "tier": tier,
            "geometry_file_format": "glb",
        }
        if request.texture:
            fields["material"] = "PBR"
        if request.topology == "quad":
            fields["mesh_mode"] = "Quad"
        for key, value in request.extra.items():
            if key != "tier":
                fields[key] = str(value)

        files: list[tuple[str, str, bytes, str]] = []
        kind = "text" if request.is_text else "image"
        if request.is_text:
            fields["prompt"] = request.prompt or ""
        else:
            assert request.image_path is not None
            path = Path(request.image_path)
            media_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
            }.get(path.suffix.lower())
            if media_type is None:
                raise AssetGenError(f"rodin: unsupported image type: {path.suffix}")
            files.append(("images", path.name, path.read_bytes(), media_type))

        body, content_type = encode_multipart(fields, files)
        from .base import _request  # noqa: PLC0415 - thin wrapper reuse

        raw = _request(
            "POST",
            f"{BASE_URL}/rodin",
            headers={**self._headers(), "Content-Type": content_type},
            body=body,
        )
        import json

        resp = json.loads(raw.decode("utf-8"))
        if resp.get("error"):
            raise AssetGenError(f"rodin submit failed: {resp['error']}")
        task_uuid = resp.get("uuid")
        subscription_key = (resp.get("jobs") or {}).get("subscription_key")
        if not task_uuid or not subscription_key:
            raise AssetGenError(f"rodin response missing uuid/subscription_key: {resp}")
        return TaskHandle(
            provider=self.name,
            task_id=task_uuid,
            kind=kind,
            meta={"subscription_key": subscription_key},
        )

    # -- poll -----------------------------------------------------------------

    def poll(self, handle: TaskHandle) -> tuple[str, str | None]:
        resp = post_json(
            f"{BASE_URL}/status",
            {"subscription_key": handle.meta["subscription_key"]},
            headers=self._headers(),
        )
        if resp.get("error") not in (None, "OK"):
            raise AssetGenError(f"rodin status failed: {resp['error']}")
        jobs = resp.get("jobs") or []
        statuses = [str(job.get("status", "")) for job in jobs]
        if any(s in _FAILED for s in statuses):
            raise AssetGenError(
                f"rodin task {handle.task_id} failed: job statuses {statuses}"
            )
        if statuses and all(s == "Done" for s in statuses):
            # All sub-jobs done: fetch the signed download URL.
            dl = post_json(
                f"{BASE_URL}/download",
                {"task_uuid": handle.task_id},
                headers=self._headers(),
            )
            if dl.get("error"):
                raise AssetGenError(f"rodin download failed: {dl['error']}")
            url = self._pick_model_url(dl.get("list") or [])
            return "succeeded", url
        return "running", None

    @staticmethod
    def _pick_model_url(entries: list[dict[str, Any]]) -> str | None:
        for entry in entries:
            name = str(entry.get("name", ""))
            if name.endswith(".glb"):
                return entry.get("url")
        return entries[0].get("url") if entries else None


__all__ = ["RodinProvider"]
