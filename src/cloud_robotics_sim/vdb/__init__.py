"""Block-sparse voxel volumes with pluggable backends.

Two backends are provided:

* ``TorchSparseVolume`` (``core/vdb_torch.py``) — torch-only block-sparse
  volume with dynamic topology; always available (CUDA / MUSA / CPU via
  ``cloud_robotics_sim.utils.device``).
* ``XuvdbVolume`` (``core/vdb_xuvdb.py``) — adapter over the optional
  `xuvdb <https://pypi.org/project/xuvdb/>`_ package (editable sparse voxel
  volumes on quadrants kernels, with native ``.xuvdb`` / OpenVDB ``.vdb``
  interop, SDF CSG stamping, DDA ray casting and particle splat). Requires
  the ``xuvdb`` extra (``pip install cloud-robotics-sim[xuvdb]``); importing
  this package without it is safe — only instantiating the backend raises.

Use :func:`create_volume` to pick a backend (``"torch"`` / ``"xuvdb"`` /
``"auto"``, where ``auto`` prefers xuvdb when installed).
"""

from __future__ import annotations

from typing import Any

from cloud_robotics_sim.vdb.core.vdb_torch import TorchSparseVolume

__all__ = [
    "SdfRayCaster",
    "TorchSparseVolume",
    "XuvdbVolume",
    "bake_static_scene",
    "create_volume",
    "from_xuvdb",
    "has_xuvdb",
    "meshes_to_sdf",
    "to_xuvdb",
]


def has_xuvdb() -> bool:
    """Whether the optional ``xuvdb`` package is importable."""
    try:
        import xuvdb  # noqa: F401
    except ImportError:
        return False
    return True


def create_volume(backend: str = "auto", **kwargs: Any) -> Any:
    """Create a sparse volume with the requested backend.

    Parameters
    ----------
    backend:
        ``"torch"`` (always available), ``"xuvdb"`` (needs the ``xuvdb``
        extra) or ``"auto"`` — xuvdb when installed, otherwise torch.
    kwargs:
        Passed through to the backend constructor. ``background`` and
        ``voxel_size`` are common to both; ``TorchSparseVolume`` additionally
        requires ``nx``/``ny``/``nz`` (accepted but not required by the
        unbounded xuvdb backend).
    """
    if backend == "auto":
        backend = "xuvdb" if has_xuvdb() else "torch"
    if backend == "xuvdb":
        from cloud_robotics_sim.vdb.core.vdb_xuvdb import XuvdbVolume

        return XuvdbVolume(**kwargs)
    if backend == "torch":
        return TorchSparseVolume(**kwargs)
    raise ValueError(f"unknown vdb backend: {backend!r}")


# Re-exported lazily-safe: vdb_xuvdb never imports xuvdb at module level.
from cloud_robotics_sim.vdb.core.static_scene import (  # noqa: E402
    bake_static_scene,
    meshes_to_sdf,
)
from cloud_robotics_sim.vdb.core.vdb_xuvdb import (  # noqa: E402
    XuvdbVolume,
    from_xuvdb,
    to_xuvdb,
)
from cloud_robotics_sim.vdb.ray_caster import SdfRayCaster  # noqa: E402

# genesis_contact pulls in genesis (a core dependency, but heavy); load it on
# first attribute access so `import cloud_robotics_sim.vdb` stays light.
_GENESIS_CONTACT_EXPORTS = {
    "GENESIS_CONTACT_CACHE_VERSION",
    "bake_sdf_grid",
    "bridge_status",
    "cached_closest_verts",
    "clear_mesh_sdf",
    "grid_fingerprint",
    "install_bake_bridge",
    "register_mesh_sdf",
}


def __getattr__(name: str) -> Any:
    if name in _GENESIS_CONTACT_EXPORTS:
        from cloud_robotics_sim.vdb.core import genesis_contact

        return getattr(genesis_contact, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__ + list(_GENESIS_CONTACT_EXPORTS))
