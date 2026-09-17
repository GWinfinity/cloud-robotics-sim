"""Tencent Hunyuan 3D (混元生3D) provider.

Uses the Tencent Cloud ``ai3d`` service (``ai3d.tencentcloudapi.com``, API
version ``2025-05-13``) with TC3-HMAC-SHA256 request signing:

- ``SubmitHunyuanTo3DProJob`` / ``QueryHunyuanTo3DProJob`` (专业版; text and
  image input share one Action — pass ``Prompt`` or ``ImageBase64``)
- ``SubmitHunyuanTo3DRapidJob`` / ``QueryHunyuanTo3DRapidJob`` (极速版;
  select with ``extra={"rapid": True}``)

Credentials: ``TENCENTCLOUD_SECRET_ID`` / ``TENCENTCLOUD_SECRET_KEY``;
optional ``HY3D_REGION`` (default ``ap-guangzhou``).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from typing import Any

from .base import (
    AssetGenError,
    AuthenticationError,
    GenerationRequest,
    TaskHandle,
    image_to_data_uri,
)

HOST = "ai3d.tencentcloudapi.com"
SERVICE = "ai3d"
VERSION = "2025-05-13"
CONTENT_TYPE = "application/json; charset=utf-8"
SIGNED_HEADERS = "content-type;host;x-tc-action"

_RUNNING = {"WAIT", "RUN"}


def _sha256_hex(data: str | bytes) -> str:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()


def _hmac_sha256(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def hashed_payload(payload: str) -> str:
    """SHA-256 hex of the request payload (TC3 canonical request step)."""
    return _sha256_hex(payload)


def canonical_request(payload_hash: str, host: str, action: str) -> str:
    """Build the TC3 canonical request string."""
    canonical_headers = (
        f"content-type:{CONTENT_TYPE}\n"
        f"host:{host}\n"
        f"x-tc-action:{action.lower()}\n"
        f"\n"
    )
    return "POST\n/\n\n" f"{canonical_headers}" f"{SIGNED_HEADERS}\n" f"{payload_hash}"


def string_to_sign(timestamp: int, date: str, service: str, canonical_hash: str) -> str:
    """Build the TC3 string-to-sign."""
    return (
        "TC3-HMAC-SHA256\n"
        f"{timestamp}\n"
        f"{date}/{service}/tc3_request\n"
        f"{canonical_hash}"
    )


def tc3_signature(secret_key: str, date: str, service: str, to_sign: str) -> str:
    """Compute the TC3-HMAC-SHA256 signature for ``to_sign``."""
    secret_date = _hmac_sha256(secret_key.encode("utf-8"), f"TC3{date}")
    secret_service = _hmac_sha256(secret_date, service)
    secret_signing = _hmac_sha256(secret_service, "tc3_request")
    return hmac.new(secret_signing, to_sign.encode("utf-8"), hashlib.sha256).hexdigest()


def _ymd(timestamp: int) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(timestamp))


class TencentProvider:
    """Tencent Hunyuan 3D provider (pro and rapid job tiers)."""

    name = "hunyuan3d"
    env_vars: tuple[str, ...] = ("TENCENTCLOUD_SECRET_ID", "TENCENTCLOUD_SECRET_KEY")

    def __init__(
        self,
        secret_id: str | None = None,
        secret_key: str | None = None,
        region: str | None = None,
    ):
        self._secret_id = secret_id
        self._secret_key = secret_key
        self.region = region or os.environ.get("HY3D_REGION", "ap-guangzhou")

    # -- credentials ---------------------------------------------------------

    @property
    def secret_id(self) -> str:
        return self._secret_id or os.environ.get("TENCENTCLOUD_SECRET_ID", "")

    @property
    def secret_key(self) -> str:
        return self._secret_key or os.environ.get("TENCENTCLOUD_SECRET_KEY", "")

    def configured(self) -> bool:
        return bool(self.secret_id and self.secret_key)

    def _check_credentials(self) -> None:
        if not self.configured():
            raise AuthenticationError(
                "TENCENTCLOUD_SECRET_ID / TENCENTCLOUD_SECRET_KEY are not set"
            )

    # -- TC3 signing ----------------------------------------------------------

    def _authorization(self, action: str, payload: str, timestamp: int) -> str:
        date = _ymd(timestamp)
        payload_hash = hashed_payload(payload)
        canonical = canonical_request(payload_hash, HOST, action)
        canonical_hash = _sha256_hex(canonical)
        to_sign = string_to_sign(timestamp, date, SERVICE, canonical_hash)
        signature = tc3_signature(self.secret_key, date, SERVICE, to_sign)
        return (
            "TC3-HMAC-SHA256 "
            f"Credential={self.secret_id}/{date}/{SERVICE}/tc3_request, "
            f"SignedHeaders={SIGNED_HEADERS}, "
            f"Signature={signature}"
        )

    def _call(self, action: str, payload_dict: dict[str, Any]) -> dict[str, Any]:
        """Call a Tencent Cloud ai3d Action and return ``Response``."""
        from .base import _request  # noqa: PLC0415 - thin wrapper reuse

        self._check_credentials()
        payload = json.dumps(payload_dict)
        timestamp = int(time.time())
        headers = {
            "Content-Type": CONTENT_TYPE,
            "Host": HOST,
            "X-TC-Action": action,
            "X-TC-Version": VERSION,
            "X-TC-Region": self.region,
            "X-TC-Timestamp": str(timestamp),
            "Authorization": self._authorization(action, payload, timestamp),
        }
        raw = _request(
            "POST", f"https://{HOST}/", headers=headers, body=payload.encode("utf-8")
        )
        resp = json.loads(raw.decode("utf-8"))
        response = resp.get("Response")
        if not isinstance(response, dict):
            raise AssetGenError(f"hunyuan3d: unexpected response shape: {resp}")
        error = response.get("Error")
        if error:
            raise AssetGenError(
                f"hunyuan3d {action} failed: {error.get('Code')}: {error.get('Message')}"
            )
        return response

    # -- submit ---------------------------------------------------------------

    def submit(self, request: GenerationRequest) -> TaskHandle:
        rapid = bool(request.extra.get("rapid", False))
        action = "SubmitHunyuanTo3DRapidJob" if rapid else "SubmitHunyuanTo3DProJob"
        payload: dict[str, Any] = {"EnablePBR": request.texture}
        if request.is_text:
            payload["Prompt"] = request.prompt
        else:
            assert request.image_path is not None
            data_uri, _ = image_to_data_uri(request.image_path)
            # Tencent expects raw base64 (no data-URI scheme prefix).
            payload["ImageBase64"] = data_uri.split(",", 1)[1]
        if request.topology == "quad" and not rapid:
            payload["GenerateType"] = "LowPoly"
            payload["PolygonType"] = "quadrilateral"
        for key, value in request.extra.items():
            if key != "rapid":
                payload[key] = value

        response = self._call(action, payload)
        job_id = response.get("JobId")
        if not job_id:
            raise AssetGenError(f"hunyuan3d response missing JobId: {response}")
        return TaskHandle(
            provider=self.name,
            task_id=job_id,
            kind="text" if request.is_text else "image",
            meta={"rapid": rapid},
        )

    # -- poll -----------------------------------------------------------------

    def poll(self, handle: TaskHandle) -> tuple[str, str | None]:
        action = (
            "QueryHunyuanTo3DRapidJob"
            if handle.meta.get("rapid")
            else "QueryHunyuanTo3DProJob"
        )
        response = self._call(action, {"JobId": handle.task_id})
        status = str(response.get("Status", ""))
        if status in _RUNNING:
            return "running", None
        if status == "FAIL":
            raise AssetGenError(
                f"hunyuan3d job {handle.task_id} failed: "
                f"{response.get('ErrorCode')}: {response.get('ErrorMessage')}"
            )
        if status == "DONE":
            files = response.get("ResultFile3Ds") or []
            url = None
            for entry in files:
                if str(entry.get("Type", "")).upper() == "GLB":
                    url = entry.get("Url")
                    break
            if url is None and files:
                url = files[0].get("Url")
            return "succeeded", url
        raise AssetGenError(f"hunyuan3d returned unknown status {status!r}")


__all__ = ["TencentProvider"]
