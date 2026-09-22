"""Offline tests for immersed-solid (obstacle) support in the 3D CFD core.

No genesis needed: mask rasterization (trimesh), the masked pressure
operator, no-slip enforcement and mass conservation around an obstacle.
"""

# ruff: noqa: E402, I001
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from plugins.solvers.cfd_coupling import CFD3D, CFDOptions
from plugins.solvers.cfd_coupling.core.obstacles import (
    combine_masks,
    mask_from_box,
    mask_from_mesh,
)

DOMAIN = (0.2, 0.1, 0.1)
CELLS = (20, 10, 10)
BOX_CENTER = (0.1, 0.05, 0.05)
BOX_SIZE = (0.04, 0.04, 0.04)


def _opts(**kwargs) -> CFDOptions:
    base = dict(
        domain=DOMAIN,
        cells=CELLS,
        viscosity=1.0e-4,
        inlet_patch=((0.4, 0.6), (0.4, 0.6)),
        advect_temperature=False,
        cg_tol=1e-6,
    )
    base.update(kwargs)
    return CFDOptions(**base)


def _box_mask() -> np.ndarray:
    return mask_from_box(BOX_CENTER, BOX_SIZE, DOMAIN, CELLS)


def _run(cfd: CFD3D, steps: int, dt: float = 0.001, u_in: float = 0.5) -> CFD3D:
    cfd.set_inlet(u_in)
    for _ in range(steps):
        cfd.step(dt)
    return cfd


def _cell_centred_u(cfd: CFD3D) -> np.ndarray:
    return (0.5 * (cfd.u[:-1] + cfd.u[1:])).cpu().numpy()


# --------------------------------------------------------------------- #
# Mask rasterization
# --------------------------------------------------------------------- #
def test_mask_from_box_geometry() -> None:
    """Box mask has the expected size, extent and centroid."""
    mask = _box_mask()
    assert mask.shape == CELLS
    # 4 cells per side at h = 0.01 -> ~4x4x4 = 64 cells.
    assert mask.sum() == pytest.approx(64, abs=8)
    idx = np.argwhere(mask)
    lo, hi = idx.min(axis=0), idx.max(axis=0)
    assert hi[0] - lo[0] == pytest.approx(3, abs=1)  # x extent ~4 cells
    # Centroid at the box centre (in cell-index space, +0.5 for centres).
    centroid = idx.mean(axis=0) + 0.5
    expected = np.array(BOX_CENTER) / np.array(DOMAIN) * np.array(CELLS)
    assert np.allclose(centroid, expected, atol=0.6)


def test_mask_from_mesh_matches_analytic_box() -> None:
    """Mesh rasterization of a box equals the analytic mask."""
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.box(extents=BOX_SIZE)
    xf = np.eye(4)
    xf[:3, 3] = BOX_CENTER
    mesh_mask = mask_from_mesh(mesh, DOMAIN, CELLS, transform=xf)
    assert np.array_equal(mesh_mask, _box_mask())


def test_combine_masks_unions() -> None:
    """combine_masks unions non-None entries."""
    a = np.zeros(CELLS, dtype=bool)
    b = np.zeros(CELLS, dtype=bool)
    a[1, 1, 1] = True
    b[2, 2, 2] = True
    out = combine_masks(a, None, b)
    assert out is not None
    assert out.sum() == 2
    assert combine_masks(None, None) is None


def test_set_solid_mask_validation() -> None:
    """Invalid masks (shape, touching inlet/outlet, full solid) are rejected."""
    cfd = CFD3D(_opts())
    with pytest.raises(ValueError, match="shape"):
        cfd.set_solid_mask(np.zeros((2, 2, 2), dtype=bool))
    with pytest.raises(ValueError, match="x = 0"):
        bad = np.zeros(CELLS, dtype=bool)
        bad[0, 3, 3] = True
        cfd.set_solid_mask(bad)
    with pytest.raises(ValueError, match="fluid cell"):
        cfd.set_solid_mask(np.ones(CELLS, dtype=bool))


# --------------------------------------------------------------------- #
# Flow around a box obstacle
# --------------------------------------------------------------------- #
def _obstacle_case(**kwargs) -> CFD3D:
    cfd = CFD3D(_opts(**kwargs))
    cfd.set_solid_mask(_box_mask())
    return _run(cfd, steps=1200, u_in=1.0)


def test_obstacle_no_slip_and_excluded_pressure() -> None:
    """Velocity vanishes inside solids; pressure solve excludes them."""
    cfd = _obstacle_case()
    mask = cfd.solid_mask
    assert mask is not None and mask.any()
    # No-slip: cell-centred velocity inside the solid is exactly zero.
    assert np.abs(_cell_centred_u(cfd)[mask]).max() < 1e-12
    # Solid cells are excluded from the Poisson solve (identity rows).
    assert np.abs(cfd.p.cpu().numpy()[mask]).max() < 1e-12


def test_obstacle_mass_conservation() -> None:
    """Incompressible inflow equals outflow with an obstacle present."""
    cfd = _obstacle_case()
    q_in = cfd.inlet_flow()
    q_out = cfd.outlet_flow()
    assert q_in == pytest.approx(1.0 * cfd.inlet_patch_area(), rel=1e-3)
    assert q_out == pytest.approx(q_in, rel=0.05)


def test_obstacle_wake_forms() -> None:
    """A velocity deficit develops in the wake behind the obstacle."""
    cfd = _obstacle_case()
    u_c = _cell_centred_u(cfd)
    mask = cfd.solid_mask
    assert mask is not None
    cy, cz = CELLS[1] // 2, CELLS[2] // 2
    upstream_core = float(u_c[3, cy, cz])  # jet core ahead of the obstacle
    downstream_core = float(u_c[16, cy, cz])  # centreline wake behind it
    assert upstream_core > 0.5
    assert downstream_core < 0.5 * upstream_core  # wake deficit


def test_obstacle_divergence_small_away_from_solid() -> None:
    """Divergence stays small in fluid cells away from the staircase surface."""
    cfd = _obstacle_case()
    mask = cfd.solid_mask
    assert mask is not None
    u = cfd.u.cpu().numpy()
    v = cfd.v.cpu().numpy()
    w = cfd.w.cpu().numpy()
    div = (
        (u[1:] - u[:-1]) + (v[:, 1:] - v[:, :-1]) + (w[:, :, 1:] - w[:, :, :-1])
    ) / cfd.h
    # Only check fluid cells whose neighbours are all fluid: pinning faces
    # after the projection leaves a local first-order staircase residual at
    # the obstacle surface (standard immersed-boundary artefact).
    fluid = ~mask
    interior = (
        fluid
        & np.roll(fluid, 1, 0)
        & np.roll(fluid, -1, 0)
        & np.roll(fluid, 1, 1)
        & np.roll(fluid, -1, 1)
        & np.roll(fluid, 1, 2)
        & np.roll(fluid, -1, 2)
    )
    interior[0] = interior[-1] = False
    assert np.abs(div[interior]).max() < 5e-4


def test_obstacle_cg_matches_direct() -> None:
    """CG and sparse-LU pressure solves agree with an obstacle."""
    cg = CFD3D(_opts(pressure_solver="cg"))
    cg.set_solid_mask(_box_mask())
    direct = CFD3D(_opts(pressure_solver="direct"))
    direct.set_solid_mask(_box_mask())
    _run(cg, steps=300)
    _run(direct, steps=300)
    assert cg.inlet_flow() == pytest.approx(direct.inlet_flow(), rel=1e-3)
    assert cg.outlet_flow() == pytest.approx(direct.outlet_flow(), rel=1e-3)
    assert float(cg.p.mean()) == pytest.approx(float(direct.p.mean()), rel=1e-3)


def test_clear_mask_restores_baseline() -> None:
    """Clearing the mask restores the no-obstacle solution exactly."""
    baseline = _run(CFD3D(_opts()), steps=200)
    disturbed = CFD3D(_opts())
    disturbed.set_solid_mask(_box_mask())
    disturbed.set_solid_mask(None)
    _run(disturbed, steps=200)
    for f in ("u", "v", "w", "p", "T"):
        a = getattr(baseline, f).cpu().numpy()
        b = getattr(disturbed, f).cpu().numpy()
        assert np.allclose(a, b, atol=1e-6), f
