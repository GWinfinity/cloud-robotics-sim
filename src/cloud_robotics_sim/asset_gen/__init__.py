"""3D asset generation API clients (cloud provider integrations).

Unified access to commercial text/image-to-3D generation services:

- ``tripo``      — Tripo AI (VAST), ``TRIPO_API_KEY``
- ``meshy``      — Meshy, ``MESHY_API_KEY``
- ``hunyuan3d``  — Tencent Hunyuan 3D, ``TENCENTCLOUD_SECRET_ID/SECRET_KEY``
- ``rodin``      — Rodin / Hyper3D, ``RODIN_API_KEY``

Quick start::

    from cloud_robotics_sim.asset_gen import AssetGenClient

    client = AssetGenClient("tripo")
    result = client.generate_text("a wooden dining chair")
    print(result.model_file)

CLI::

    python -m cloud_robotics_sim.asset_gen providers
    python -m cloud_robotics_sim.asset_gen text "a wooden dining chair" --provider tripo
    python -m cloud_robotics_sim.asset_gen image chair.png --provider meshy

Generated ``.glb`` files plus ``.json`` provenance land in
``outputs/asset_staging/gen3d`` by default and can be imported into the
object library via ``scripts/expand_object_library.py``.
"""

from .base import (
    AssetGenError,
    AuthenticationError,
    GenerationRequest,
    GenerationResult,
    TaskFailedError,
    TaskHandle,
    TaskTimeoutError,
)
from .client import (
    AssetGenClient,
    configured_providers,
    get_provider,
    list_providers,
    register_provider,
)
from .hunyuan3d import TencentProvider
from .meshy import MeshyProvider
from .rodin import RodinProvider
from .tripo import TripoProvider

__all__ = [
    "AssetGenClient",
    "AssetGenError",
    "AuthenticationError",
    "GenerationRequest",
    "GenerationResult",
    "MeshyProvider",
    "RodinProvider",
    "TaskFailedError",
    "TaskHandle",
    "TaskTimeoutError",
    "TencentProvider",
    "TripoProvider",
    "configured_providers",
    "get_provider",
    "list_providers",
    "register_provider",
]
