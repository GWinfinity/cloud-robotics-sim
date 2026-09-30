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
    "TorchSparseVolume",
    "XuvdbVolume",
    "create_volume",
    "from_xuvdb",
    "has_xuvdb",
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
from cloud_robotics_sim.vdb.core.vdb_xuvdb import (  # noqa: E402
    XuvdbVolume,
    from_xuvdb,
    to_xuvdb,
)
