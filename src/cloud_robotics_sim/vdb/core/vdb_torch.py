"""Torch backend for the block-sparse volume (VDB-lite).

Design notes (benchmarked against fVDB / NanoVDB):

* Single-level block sparsity with a **separate index grid and value pool** —
  the same "indexed storage" idea as fVDB's ``GridBatch`` and NanoVDB's
  linearized tree: ``table[bi, bj, bk] -> slot`` (i32, -1 = inactive) plus a
  dense ``pool[slot, B, B, B]`` of active blocks. The block size defaults to
  8 (the OpenVDB/NanoVDB/fVDB leaf size).
* Unlike NanoVDB (static topology) and fVDB (build-time topology), the
  topology here is **dynamic**: blocks can be activated / deactivated at
  runtime with swap-remove compaction, which suits evolving simulation
  fields.
* Per-slot min/max value metadata is maintained by the device ops (the
  NanoVDB node-embedded min/max idea), ready for pruning / ray traversal.
* Stencil ops look neighbours up through the block table directly (6 table
  lookups per voxel). The fVDB-style optimisation — staging a 10^3 halo
  window into block-local shared memory per 8^3 leaf — is documented as a
  future optimisation path.

Device selection goes through ``cloud_robotics_sim.utils.device`` so the
same code runs on CUDA, MUSA (Moore Threads via ``torch_musa``) or CPU.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from cloud_robotics_sim.utils.device import get_device


def _check_block_size(block_size: int) -> None:
    if block_size < 2 or block_size > 32 or (block_size & (block_size - 1)) != 0:
        raise ValueError("block_size must be a power of two in [2, 32]")


class TorchSparseVolume:
    """Block-sparse scalar (f32) volume on the default torch device.

    Parameters
    ----------
    nx, ny, nz : int
        Logical dense extent in voxels.
    block_size : int
        Edge length of a sparse block (power of two, default 8 like VDB).
    background : float
        Value of inactive voxels (VDB background semantics).
    device : str | None
        ``"cuda"`` / ``"musa"`` / ``"cpu"`` (see ``utils.device``); None =
        auto-select (MUSA > CUDA > CPU).
    max_blocks : int
        Initial pool capacity in blocks; grows by doubling on demand.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        block_size: int = 8,
        background: float = 0.0,
        device: str | None = None,
        max_blocks: int = 1024,
    ) -> None:
        _check_block_size(block_size)
        if nx < 1 or ny < 1 or nz < 1:
            raise ValueError("grid extent must be positive")
        self.nx, self.ny, self.nz = nx, ny, nz
        self.block_size = block_size
        self.background = float(background)
        self.device = torch.device(get_device(device))
        self.bx = (nx + block_size - 1) // block_size
        self.by = (ny + block_size - 1) // block_size
        self.bz = (nz + block_size - 1) // block_size
        self.max_blocks = max(1, int(max_blocks))
        self._n_active = 0
        # Host topology mirror (block table and slot -> block coords).
        self._table_np = np.full((self.bx, self.by, self.bz), -1, dtype=np.int32)
        self._slot_block_np = np.zeros((self.max_blocks, 3), dtype=np.int32)
        # Device tensors.
        self._table = torch.full(
            (self.bx, self.by, self.bz), -1, dtype=torch.int32, device=self.device
        )
        self._slot_block = torch.zeros(
            (self.max_blocks, 3), dtype=torch.int32, device=self.device
        )
        self._pool = torch.zeros(
            (self.max_blocks, block_size, block_size, block_size),
            dtype=torch.float32,
            device=self.device,
        )
        self._pool_min = torch.full((self.max_blocks,), math.inf, device=self.device)
        self._pool_max = torch.full((self.max_blocks,), -math.inf, device=self.device)

    # ------------------------------------------------------------------ #
    # Basic properties
    # ------------------------------------------------------------------ #
    @property
    def backend(self) -> str:
        return "torch"

    @property
    def device_name(self) -> str:
        return str(self.device)

    @property
    def n_active_blocks(self) -> int:
        return self._n_active

    @property
    def active_voxels_capacity(self) -> int:
        return self._n_active * self.block_size**3

    def count_active_voxels(self) -> int:
        """Voxels of active blocks inside the logical domain."""
        b = self.block_size
        total = 0
        for s in range(self._n_active):
            bi, bj, bk = (int(v) for v in self._slot_block_np[s])
            total += (
                min(b, self.nx - bi * b)
                * min(b, self.ny - bj * b)
                * min(b, self.nz - bk * b)
            )
        return total

    # ------------------------------------------------------------------ #
    # Pool growth
    # ------------------------------------------------------------------ #
    def _grow_pool(self, needed: int) -> None:
        if needed <= self.max_blocks:
            return
        new_max = max(self.max_blocks * 2, needed)
        new_pool = torch.zeros(
            (new_max, self.block_size, self.block_size, self.block_size),
            dtype=torch.float32,
            device=self.device,
        )
        new_slot_block = torch.zeros(
            (new_max, 3), dtype=torch.int32, device=self.device
        )
        new_min = torch.full((new_max,), math.inf, device=self.device)
        new_maxv = torch.full((new_max,), -math.inf, device=self.device)
        n = self._n_active
        if n > 0:
            new_pool[:n] = self._pool[:n]
            new_slot_block[:n] = self._slot_block[:n]
            new_min[:n] = self._pool_min[:n]
            new_maxv[:n] = self._pool_max[:n]
        self._pool = new_pool
        self._slot_block = new_slot_block
        self._pool_min = new_min
        self._pool_max = new_maxv
        self._slot_block_np = np.zeros((new_max, 3), dtype=np.int32)
        self._slot_block_np[:n] = self._slot_block[:n].cpu().numpy()
        self.max_blocks = new_max

    def _upload_topology(self) -> None:
        self._table.copy_(torch.from_numpy(self._table_np).to(self.device))
        if self._n_active > 0:
            sb = torch.from_numpy(self._slot_block_np[: self.max_blocks]).to(
                self.device
            )
            self._slot_block.copy_(sb)

    # ------------------------------------------------------------------ #
    # Topology operations (host mirror, then bulk upload)
    # ------------------------------------------------------------------ #
    def activate_blocks(self, coords: np.ndarray) -> int:
        """Activate the given block coordinates ``(m, 3)``; returns newly added."""
        coords = np.atleast_2d(np.asarray(coords, dtype=np.int64)).reshape(-1, 3)
        added = 0
        for c in coords:
            bi, bj, bk = (int(v) for v in c)
            if not (0 <= bi < self.bx and 0 <= bj < self.by and 0 <= bk < self.bz):
                raise ValueError(f"block coord {(bi, bj, bk)} out of range")
            if self._table_np[bi, bj, bk] >= 0:
                continue
            self._grow_pool(self._n_active + 1)
            slot = self._n_active
            self._table_np[bi, bj, bk] = slot
            self._slot_block_np[slot] = (bi, bj, bk)
            self._pool[slot].fill_(self.background)
            self._pool_min[slot] = math.inf
            self._pool_max[slot] = -math.inf
            self._n_active += 1
            added += 1
        if added:
            self._upload_topology()
        return added

    def deactivate_blocks(self, coords: np.ndarray) -> int:
        """Deactivate blocks with swap-remove compaction; returns removed count."""
        coords = np.atleast_2d(np.asarray(coords, dtype=np.int64)).reshape(-1, 3)
        removed = 0
        for c in coords:
            bi, bj, bk = (int(v) for v in c)
            slot = int(self._table_np[bi, bj, bk])
            if slot < 0:
                continue
            last = self._n_active - 1
            if slot != last:
                # Swap-remove: move the last live block into the freed slot.
                self._pool[slot] = self._pool[last]
                self._pool_min[slot] = self._pool_min[last]
                self._pool_max[slot] = self._pool_max[last]
                lb = self._slot_block_np[last].copy()
                self._slot_block_np[slot] = lb
                self._table_np[int(lb[0]), int(lb[1]), int(lb[2])] = slot
            self._table_np[bi, bj, bk] = -1
            self._n_active = last
            removed += 1
        if removed:
            self._upload_topology()
        return removed

    def from_mask(self, mask: np.ndarray) -> int:
        """Activate every block that contains at least one True voxel."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.nx, self.ny, self.nz):
            raise ValueError(
                f"mask shape {mask.shape} != {(self.nx, self.ny, self.nz)}"
            )
        b = self.block_size
        blocked = mask.reshape(self.bx, b, self.by, b, self.bz, b).any(axis=(1, 3, 5))
        coords = np.argwhere(blocked)
        return self.activate_blocks(coords)

    def from_dense(self, arr: np.ndarray, threshold: float | None = None) -> int:
        """Activate blocks and copy values from a dense array.

        ``threshold=None`` activates every block; otherwise activates blocks
        containing at least one value strictly above the threshold.
        """
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape != (self.nx, self.ny, self.nz):
            raise ValueError(f"arr shape {arr.shape} != {(self.nx, self.ny, self.nz)}")
        n = (
            self.from_mask(arr > threshold)
            if threshold is not None
            else self.from_mask(np.ones((self.nx, self.ny, self.nz), dtype=bool))
        )
        self.import_window((0, 0, 0), arr)
        return n

    def topology(self) -> np.ndarray:
        """Copy of the block table (``-1`` = inactive)."""
        out: np.ndarray = self._table_np.copy()
        return out

    # ------------------------------------------------------------------ #
    # Value access
    # ------------------------------------------------------------------ #
    def set_background(self, value: float) -> None:
        self.background = float(value)

    def fill(self, value: float) -> None:
        """Fill all *active* voxels (inactive stays background)."""
        if self._n_active > 0:
            self._pool[: self._n_active].fill_(float(value))
            self._pool_min[: self._n_active] = float(value)
            self._pool_max[: self._n_active] = float(value)

    def to_dense(self) -> np.ndarray:
        """Export the logical domain as a dense numpy array (background fill)."""
        out = np.full((self.nx, self.ny, self.nz), self.background, dtype=np.float32)
        return self._export_into(out, (0, 0, 0))

    def _export_into(self, out: np.ndarray, origin: tuple[int, int, int]) -> np.ndarray:
        b = self.block_size
        for s in range(self._n_active):
            bi, bj, bk = (int(v) for v in self._slot_block_np[s])
            gx, gy, gz = bi * b, bj * b, bk * b
            block = self._pool[s].cpu().numpy()
            x0, y0, z0 = gx - origin[0], gy - origin[1], gz - origin[2]
            xs, ys, zs = max(x0, 0), max(y0, 0), max(z0, 0)
            xe = min(x0 + b, out.shape[0])
            ye = min(y0 + b, out.shape[1])
            ze = min(z0 + b, out.shape[2])
            if xe > xs and ye > ys and ze > zs:
                out[xs:xe, ys:ye, zs:ze] = block[
                    xs - x0 : xe - x0, ys - y0 : ye - y0, zs - z0 : ze - z0
                ]
        return out

    def export_window(
        self, origin: tuple[int, int, int], shape: tuple[int, int, int]
    ) -> np.ndarray:
        """Export a window starting at ``origin`` (background outside)."""
        out = np.full(shape, self.background, dtype=np.float32)
        return self._export_into(out, origin)

    def import_window(self, origin: tuple[int, int, int], arr: np.ndarray) -> None:
        """Write a dense window into active blocks, activating as needed."""
        arr = np.asarray(arr, dtype=np.float32)
        b = self.block_size
        x0, y0, z0 = origin
        bx0, by0, bz0 = x0 // b, y0 // b, z0 // b
        bx1 = (x0 + arr.shape[0] - 1) // b
        by1 = (y0 + arr.shape[1] - 1) // b
        bz1 = (y0 + arr.shape[2] - 1) // b
        coords = [
            (bi, bj, bk)
            for bi in range(max(bx0, 0), min(bx1, self.bx - 1) + 1)
            for bj in range(max(by0, 0), min(by1, self.by - 1) + 1)
            for bk in range(max(bz0, 0), min(bz1, self.bz - 1) + 1)
        ]
        if coords:
            self.activate_blocks(np.asarray(coords, dtype=np.int64))
        self._upload_pool_window(origin, arr)

    def _upload_pool_window(
        self, origin: tuple[int, int, int], arr: np.ndarray
    ) -> None:
        b = self.block_size
        x0, y0, z0 = origin
        for s in range(self._n_active):
            bi, bj, bk = (int(v) for v in self._slot_block_np[s])
            gx, gy, gz = bi * b, bj * b, bk * b
            xs, ys, zs = max(gx, x0), max(gy, y0), max(gz, z0)
            xe = min(gx + b, x0 + arr.shape[0])
            ye = min(gy + b, y0 + arr.shape[1])
            ze = min(gz + b, z0 + arr.shape[2])
            if xe > xs and ye > ys and ze > zs:
                self._pool[
                    s, xs - gx : xe - gx, ys - gy : ye - gy, zs - gz : ze - gz
                ] = torch.from_numpy(
                    arr[xs - x0 : xe - x0, ys - y0 : ye - y0, zs - z0 : ze - z0]
                ).to(
                    self.device
                )
        self._refresh_minmax()

    def _refresh_minmax(self) -> None:
        if self._n_active > 0:
            blk = self._pool[: self._n_active]
            self._pool_min[: self._n_active] = blk.amin(dim=(1, 2, 3))
            self._pool_max[: self._n_active] = blk.amax(dim=(1, 2, 3))

    # ------------------------------------------------------------------ #
    # Device ops (operate on active blocks only)
    # ------------------------------------------------------------------ #
    def scale(self, a: float) -> None:
        """In-place ``self *= a`` on active voxels; min/max metadata follows."""
        if self._n_active > 0:
            self._pool[: self._n_active] *= float(a)
            self._pool_min[: self._n_active] *= float(a)
            self._pool_max[: self._n_active] *= float(a)
            if a < 0:
                self._pool_min[: self._n_active], self._pool_max[: self._n_active] = (
                    self._pool_max[: self._n_active].clone(),
                    self._pool_min[: self._n_active].clone(),
                )

    def add_const(self, v: float) -> None:
        """In-place ``self += v`` on active voxels."""
        if self._n_active > 0:
            self._pool[: self._n_active] += float(v)
            self._pool_min[: self._n_active] += float(v)
            self._pool_max[: self._n_active] += float(v)

    def axpy(
        self, a: float, x: "TorchSparseVolume", b: float, y: "TorchSparseVolume"
    ) -> None:
        """In-place ``self = a*x + b*y`` (identical topology required)."""
        self._check_same_topology(x)
        self._check_same_topology(y)
        n = self._n_active
        if n > 0:
            self._pool[:n] = float(a) * x._pool[:n] + float(b) * y._pool[:n]
            self._refresh_minmax()

    def _check_same_topology(self, other: "TorchSparseVolume") -> None:
        if not isinstance(other, TorchSparseVolume):
            raise TypeError(f"expected TorchSparseVolume, got {type(other)!r}")
        if (
            (self.nx, self.ny, self.nz) != (other.nx, other.ny, other.nz)
            or self.block_size != other.block_size
            or self._n_active != other._n_active
            or not np.array_equal(
                self._slot_block_np[: self._n_active],
                other._slot_block_np[: other._n_active],
            )
        ):
            raise ValueError("topology mismatch between sparse volumes")

    def _neighbour_values(self) -> list[torch.Tensor]:
        """Per-direction neighbour values for all active voxels.

        Returns 6 tensors of shape ``(n_active, B, B, B)`` in the order
        -x, +x, -y, +y, -z, +z; out-of-domain and inactive-block neighbours
        read as ``background``.
        """
        b = self.block_size
        n = self._n_active
        sb = self._slot_block[:n]  # (n, 3)
        device = self.device
        bg = torch.tensor(self.background, dtype=torch.float32, device=device)
        out: list[torch.Tensor] = []
        arange_b = torch.arange(b, device=device)
        for axis, sign in ((0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)):
            origin = sb * b  # (n, 3) global coord of block corner
            # global neighbour voxel coords: (3, n, b, b, b)
            # Per-axis global voxel coordinates, each broadcast to
            # ``(n, b, b, b)``: x varies with dim 1, y with dim 2, z with 3.
            g = torch.stack(
                [
                    (origin[:, 0].view(n, 1, 1, 1) + arange_b.view(1, b, 1, 1)).expand(
                        n, b, b, b
                    )
                    + (sign if axis == 0 else 0),
                    (origin[:, 1].view(n, 1, 1, 1) + arange_b.view(1, 1, b, 1)).expand(
                        n, b, b, b
                    )
                    + (sign if axis == 1 else 0),
                    (origin[:, 2].view(n, 1, 1, 1) + arange_b.view(1, 1, 1, b)).expand(
                        n, b, b, b
                    )
                    + (sign if axis == 2 else 0),
                ],
                dim=0,
            )  # (3, n, b, b, b)
            nb = torch.div(g, b, rounding_mode="floor")  # neighbour block coords
            local = g - nb * b
            in_domain = (
                (g[0] >= 0)
                & (g[0] < self.nx)
                & (g[1] >= 0)
                & (g[1] < self.ny)
                & (g[2] >= 0)
                & (g[2] < self.nz)
            )
            nb_clamped = nb.clone()
            nb_clamped[0] = nb_clamped[0].clamp(0, self.bx - 1)
            nb_clamped[1] = nb_clamped[1].clamp(0, self.by - 1)
            nb_clamped[2] = nb_clamped[2].clamp(0, self.bz - 1)
            nslot = self._table[nb_clamped[0], nb_clamped[1], nb_clamped[2]]
            valid = in_domain & (nslot >= 0)
            safe_slot = torch.where(valid, nslot, torch.zeros_like(nslot))
            vals = self._pool[safe_slot, local[0], local[1], local[2]]
            vals = torch.where(valid, vals, bg)
            out.append(vals)
        return out

    def laplacian(self, out: "TorchSparseVolume") -> None:
        """6-neighbour stencil into ``out`` (identical topology required).

        Inactive-block and out-of-domain neighbours contribute the
        background value, matching the dense ``background`` semantics.
        """
        self._check_same_topology(out)
        if self._n_active == 0:
            return
        nb = self._neighbour_values()
        centre = self._pool[: self._n_active]
        lap = nb[1] + nb[0] + nb[3] + nb[2] + nb[5] + nb[4] - 6.0 * centre
        out._pool[: self._n_active] = lap
        out._refresh_minmax()

    # ------------------------------------------------------------------ #
    # Reductions (active voxels only)
    # ------------------------------------------------------------------ #
    def reduce_sum(self) -> float:
        if self._n_active == 0:
            return 0.0
        return float(self._pool[: self._n_active].sum())

    def reduce_max(self) -> float:
        if self._n_active == 0:
            return float("-inf")
        return float(self._pool_max[: self._n_active].max())

    def norm2(self) -> float:
        """Squared L2 norm over active voxels."""
        if self._n_active == 0:
            return 0.0
        return float((self._pool[: self._n_active] ** 2).sum())

    def block_minmax(self) -> np.ndarray:
        """Per-slot (min, max) metadata for active blocks, shape (n_active, 2)."""
        if self._n_active == 0:
            return np.zeros((0, 2), dtype=np.float32)
        mm = torch.stack(
            [self._pool_min[: self._n_active], self._pool_max[: self._n_active]],
            dim=1,
        )
        return mm.cpu().numpy()

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def memory_report(self) -> dict:
        """Memory footprint vs the equivalent dense grid."""
        dense_bytes = self.nx * self.ny * self.nz * 4
        pool_bytes = self.max_blocks * self.block_size**3 * 4
        table_bytes = self.bx * self.by * self.bz * 4
        active = self.count_active_voxels()
        return {
            "backend": self.backend,
            "device": self.device_name,
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "block_size": self.block_size,
            "active_blocks": self._n_active,
            "active_voxels": active,
            "max_blocks": self.max_blocks,
            "pool_bytes": pool_bytes,
            "table_bytes": table_bytes,
            "total_bytes": pool_bytes + table_bytes,
            "dense_bytes": dense_bytes,
            "compression_ratio": dense_bytes / max(pool_bytes + table_bytes, 1),
            "active_fill_ratio": active / max(self.nx * self.ny * self.nz, 1),
        }


__all__ = ["TorchSparseVolume"]
