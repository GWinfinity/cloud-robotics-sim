# SPDX-FileCopyrightText: Copyright (c) 2025 Cloud Robotics Sim
# SPDX-License-Identifier: Apache-2.0

"""Generic Wave Function Collapse (WFC) solver for grid layouts.

The solver is domain-agnostic: callers describe tiles as *variants* carrying a
4-tuple of edge "sockets" (N, E, S, W) plus a weight, and provide two
compatibility predicates:

- interior adjacency: ``compatible(socket_a, socket_b)`` between two cells.
- exterior boundary: ``exterior_ok(socket)`` — whether a socket may face the
  outside of the grid (the virtual neighbour socket ``"outside"``).

Collapse uses minimum-entropy cell selection, weighted random picking,
queue-based constraint propagation, and snapshot backtracking on
contradiction. Everything is driven by a ``random.Random`` instance so a
given seed always produces the same layout.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

# Directions in socket order: N(+y), E(+x), S(-y), W(-x).
DIRS: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (0, -1), (-1, 0))

# Virtual socket name used for edges that face the outside of the grid.
OUTSIDE = "outside"

# Rotation multiples (degrees). Rotating a tile clockwise by ``rot`` moves its
# local W edge to N, so the rotated socket tuple is the list rotated right by
# rot // 90.
ROTATIONS: tuple[int, ...] = (0, 90, 180, 270)


class WFCContradictionError(RuntimeError):
    """Raised when the grid cannot be collapsed under the given rules."""


@dataclass(frozen=True)
class TileVariant:
    """One tile wall-pattern in one rotation.

    Attributes:
        tile: Logical tile name (shared by all variants of a tile).
        pattern: Index of the wall pattern within the tile (0 = default).
        rot: Clockwise rotation in degrees (0/90/180/270).
        sockets: Socket names of the N/E/S/W edges after rotation.
    """

    tile: str
    pattern: int
    rot: int
    sockets: tuple[str, str, str, str]


def rotated_sockets(sockets: Sequence[str], rot: int) -> tuple[str, str, str, str]:
    """Return the socket tuple of a tile rotated clockwise by ``rot`` degrees."""
    k = ((rot // 90) % 4 + 4) % 4
    seq = tuple(sockets)
    return seq[-k:] + seq[:-k] if k else seq


def expand_variants(
    tile_patterns: dict[str, Sequence[Sequence[str]]],
    rotations: Sequence[int] = ROTATIONS,
) -> list[TileVariant]:
    """Expand ``{tile: [wall patterns]}`` into one variant per pattern/rotation."""
    variants: list[TileVariant] = []
    for tile, patterns in tile_patterns.items():
        for pattern, sockets in enumerate(patterns):
            for rot in rotations:
                variants.append(
                    TileVariant(
                        tile=tile,
                        pattern=pattern,
                        rot=rot,
                        sockets=rotated_sockets(sockets, rot),
                    )
                )
    return variants


class _ContradictionError(Exception):
    """Internal signal: a cell ended up with an empty domain."""

    def __init__(self, cell: int):
        super().__init__(f"cell {cell} contradiction")
        self.cell = cell


def collapse(
    width: int,
    height: int,
    variants: Sequence[TileVariant],
    weights: Sequence[float],
    compatible: Callable[[str, str], bool],
    exterior_ok: Callable[[str], bool],
    rng: random.Random,
    max_backtracks: int = 400,
    presets: Optional[dict[int, set[int]]] = None,
    max_counts: Optional[dict[str, int]] = None,
) -> list[list[int]]:
    """Collapse a ``width x height`` grid and return per-cell variant indices.

    Args:
        width: Grid width (number of cells along +x).
        height: Grid height (number of cells along +y).
        variants: Tile variants (see :func:`expand_variants`).
        weights: Sampling weight per variant.
        compatible: ``compatible(socket_here, socket_neighbor)`` for interior
            adjacency; must be symmetric.
        exterior_ok: Whether a socket may face the grid exterior.
        rng: Seeded ``random.Random`` driving all stochastic choices.
        max_backtracks: Safety cap on backtracking steps.
        presets: Optional[dict[int, set[int]]] = restrictions applied before
            collapsing — the standard WFC mechanism for guaranteeing
            "at least one X" style global constraints.
        max_counts: Optional ``{tile name: max placements}`` enforced during
            collapse (hard constraint, not a post-check).

    Returns:
        ``height x width`` list of variant indices (``result[j][i]``).

    Raises:
        WFCContradictionError: If no assignment satisfies all constraints
            within the backtrack budget.
    """
    if width < 1 or height < 1:
        raise ValueError("grid must be at least 1x1")
    if len(weights) != len(variants):
        raise ValueError("need one weight per variant")

    n = width * height
    domains: list[set[int]] = []
    for idx in range(n):
        if presets is not None and idx in presets:
            domains.append(set(presets[idx]))
        else:
            domains.append(set(range(len(variants))))
    snapshots: list[list[set[int]]] = []
    bans: list[tuple[int, int]] = []  # (cell, banned variant) per snapshot
    backtrack_count = 0

    def cell_index(i: int, j: int) -> int:
        return j * width + i

    def apply_exterior(idx: int, domain: set[int]) -> None:
        """Drop variants whose edge socket may not face the grid exterior."""
        i, j = idx % width, idx // width
        for d, (dx, dy) in enumerate(DIRS):
            ni, nj = i + dx, j + dy
            if 0 <= ni < width and 0 <= nj < height:
                continue
            domain.intersection_update(
                {v for v in domain if exterior_ok(variants[v].sockets[d])}
            )

    def tile_count_cap(tile: str) -> int:
        if max_counts is None:
            return len(variants)
        return max_counts.get(tile, len(variants))

    def counts_of(dom: list[set[int]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for domain in dom:
            if len(domain) == 1:
                tile = variants[next(iter(domain))].tile
                counts[tile] = counts.get(tile, 0) + 1
        return counts

    def check_cap(idx: int, domain: set[int], counts: dict[str, int]) -> None:
        """Raise when a decided cell would exceed its tile's count cap."""
        if len(domain) == 1:
            variant = variants[next(iter(domain))]
            if counts.get(variant.tile, 0) > tile_count_cap(variant.tile):
                raise _ContradictionError(idx)

    def propagate(queue: list[int]) -> None:
        """Constraint propagation: shrink neighbour domains until stable."""
        while queue:
            idx = queue.pop()
            i, j = idx % width, idx // width
            for d, (dx, dy) in enumerate(DIRS):
                ni, nj = i + dx, j + dy
                if not (0 <= ni < width and 0 <= nj < height):
                    continue
                nidx = cell_index(ni, nj)
                edge_sockets = {variants[v].sockets[d] for v in domains[idx]}
                opposite = (d + 2) % 4
                keep = {
                    v
                    for v in domains[nidx]
                    if any(
                        compatible(edge, variants[v].sockets[opposite])
                        for edge in edge_sockets
                    )
                }
                if not keep:
                    raise _ContradictionError(nidx)
                if keep != domains[nidx]:
                    domains[nidx] = keep
                    if len(keep) == 1:
                        check_cap(nidx, keep, counts_of(domains))
                    queue.append(nidx)

    def backtrack() -> None:
        """Undo snapshots until the banned choice leaves a non-empty domain."""
        nonlocal backtrack_count
        while snapshots:
            backtrack_count += 1
            if backtrack_count > max_backtracks:
                raise WFCContradictionError(
                    "no layout satisfies the rules "
                    f"(backtrack budget {max_backtracks} spent)"
                )
            domains[:] = snapshots.pop()
            cell, banned = bans.pop()
            domains[cell].discard(banned)
            if domains[cell]:
                try:
                    propagate([cell])
                    return
                except _ContradictionError:
                    continue  # keep popping to an earlier snapshot
        raise WFCContradictionError(
            "no layout satisfies the rules (search space exhausted)"
        )

    for idx in range(n):
        apply_exterior(idx, domains[idx])
        if not domains[idx]:
            raise WFCContradictionError(
                f"cell {idx} has no variant compatible with the exterior"
            )
    initial_counts = counts_of(domains)
    for idx in range(n):
        check_cap(idx, domains[idx], initial_counts)

    # Presets can pin cells to singletons before any observation; propagate
    # them now so conflicting presets fail fast instead of silently surviving.
    try:
        propagate(list(range(n)))
    except _ContradictionError as exc:
        raise WFCContradictionError(str(exc)) from exc

    while True:
        undecided = [idx for idx in range(n) if len(domains[idx]) > 1]
        if not undecided:
            return [
                [next(iter(domains[cell_index(i, j)])) for i in range(width)]
                for j in range(height)
            ]
        idx = min(undecided, key=lambda c: (len(domains[c]), c))
        try:
            counts = counts_of(domains)
            pool = [
                v
                for v in sorted(domains[idx])
                if counts.get(variants[v].tile, 0) < tile_count_cap(variants[v].tile)
            ]
            if not pool:
                # Every remaining option for this cell exceeds a count cap.
                raise _ContradictionError(idx)
            choice = rng.choices(pool, weights=[weights[v] for v in pool], k=1)[0]

            snapshots.append([set(d) for d in domains])
            bans.append((idx, choice))
            domains[idx] = {choice}
            propagate([idx])
        except _ContradictionError:
            backtrack()
