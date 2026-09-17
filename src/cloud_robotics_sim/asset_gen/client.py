"""Unified client facade and provider registry for 3D asset generation.

Example::

    from cloud_robotics_sim.asset_gen import AssetGenClient, GenerationRequest

    client = AssetGenClient(provider="tripo")  # or auto-pick a configured one
    result = client.generate(GenerationRequest(prompt="a wooden dining chair"),
                             out_dir="outputs/asset_staging/gen3d")
    print(result.model_file)  # downloaded .glb + provenance .json

Credentials come from environment variables (``python-dotenv`` compatible):

- ``TRIPO_API_KEY`` — Tripo AI (VAST)
- ``MESHY_API_KEY`` — Meshy
- ``TENCENTCLOUD_SECRET_ID`` / ``TENCENTCLOUD_SECRET_KEY`` — Tencent Hunyuan 3D
  (optional ``HY3D_REGION``, default ``ap-guangzhou``)
- ``RODIN_API_KEY`` — Rodin / Hyper3D
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .base import (
    AssetGenError,
    GenerationRequest,
    GenerationResult,
    Provider,
    TaskHandle,
    download_file,
    poll_task,
)
from .hunyuan3d import TencentProvider
from .meshy import MeshyProvider
from .rodin import RodinProvider
from .tripo import TripoProvider

logger = logging.getLogger(__name__)

_PROVIDERS: dict[str, Provider] = {}


def register_provider(provider: Provider) -> None:
    """Register a provider instance under its ``name``."""
    _PROVIDERS[provider.name] = provider


def get_provider(name: str) -> Provider:
    """Return the provider called ``name``."""
    try:
        return _PROVIDERS[name]
    except KeyError:
        known = ", ".join(sorted(_PROVIDERS)) or "(none)"
        raise AssetGenError(f"unknown provider {name!r}; known: {known}") from None


def list_providers() -> list[str]:
    """Return the sorted names of all registered providers."""
    return sorted(_PROVIDERS)


def configured_providers() -> list[str]:
    """Return the sorted names of providers whose credentials are available."""
    return sorted(name for name, p in _PROVIDERS.items() if p.configured())


def _register_builtin() -> None:
    for provider in (
        TripoProvider(),
        MeshyProvider(),
        TencentProvider(),
        RodinProvider(),
    ):
        register_provider(provider)


_register_builtin()


class AssetGenClient:
    """High-level facade: submit, poll, download, and record provenance.

    Args:
        provider: Provider name (see :func:`list_providers`). When None, the
            first configured provider is used.
        poll_timeout: Total seconds to wait for a task (default 600).
        poll_interval: Initial polling interval in seconds (default 5).
    """

    def __init__(
        self,
        provider: str | None = None,
        *,
        poll_timeout: float = 600.0,
        poll_interval: float = 5.0,
    ):
        if provider is None:
            candidates = configured_providers()
            if not candidates:
                raise AssetGenError(
                    "no 3D generation provider is configured; set one of: "
                    "TRIPO_API_KEY, MESHY_API_KEY, TENCENTCLOUD_SECRET_ID"
                    "/TENCENTCLOUD_SECRET_KEY, RODIN_API_KEY"
                )
            provider = candidates[0]
            logger.info("auto-selected provider: %s", provider)
        self.provider = get_provider(provider)
        self.poll_timeout = poll_timeout
        self.poll_interval = poll_interval

    # -- low-level ------------------------------------------------------------

    def submit(self, request: GenerationRequest) -> TaskHandle:
        """Submit a generation task without waiting."""
        return self.provider.submit(request)

    def wait(self, handle: TaskHandle) -> str:
        """Wait for a submitted task and return the model download URL."""
        return poll_task(
            handle,
            self.provider.poll,
            timeout=self.poll_timeout,
            interval=self.poll_interval,
        )

    # -- high-level -----------------------------------------------------------

    def generate(
        self,
        request: GenerationRequest,
        *,
        out_dir: str | Path = "outputs/asset_staging/gen3d",
    ) -> GenerationResult:
        """Submit, wait, download the GLB, and write provenance JSON.

        The output directory layout is compatible with
        ``scripts/expand_object_library.py`` staging conventions::

            <out_dir>/<provider>_<task_id>.glb
            <out_dir>/<provider>_<task_id>.json   # provenance
        """
        handle = self.submit(request)
        logger.info("submitted %s task %s", handle.provider, handle.task_id)
        url = self.wait(handle)
        out_path = Path(out_dir)
        model_file = download_file(
            url, out_path / f"{handle.provider}_{handle.task_id}.glb"
        )
        result = GenerationResult(
            provider=handle.provider,
            task_id=handle.task_id,
            model_file=model_file,
            source_url=url,
            metadata={"kind": handle.kind},
        )
        self._write_provenance(result, request, out_path)
        return result

    def generate_text(
        self,
        prompt: str,
        *,
        out_dir: str | Path = "outputs/asset_staging/gen3d",
        **kwargs: Any,
    ) -> GenerationResult:
        """Convenience wrapper for text-to-3D."""
        request = GenerationRequest(prompt=prompt, **kwargs)
        return self.generate(request, out_dir=out_dir)

    def generate_image(
        self,
        image_path: str | Path,
        *,
        out_dir: str | Path = "outputs/asset_staging/gen3d",
        **kwargs: Any,
    ) -> GenerationResult:
        """Convenience wrapper for image-to-3D."""
        request = GenerationRequest(image_path=image_path, **kwargs)
        return self.generate(request, out_dir=out_dir)

    # -- provenance -----------------------------------------------------------

    @staticmethod
    def _write_provenance(
        result: GenerationResult, request: GenerationRequest, out_dir: Path
    ) -> None:
        provenance = {
            "provider": result.provider,
            "task_id": result.task_id,
            "prompt": request.prompt,
            "image": str(request.image_path) if request.image_path else None,
            "texture": request.texture,
            "source_url": result.source_url,
            "license": "generated; see provider terms for commercial use",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        prov_file = out_dir / f"{result.provider}_{result.task_id}.json"
        prov_file.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
        )


__all__ = [
    "AssetGenClient",
    "configured_providers",
    "get_provider",
    "list_providers",
    "register_provider",
]
