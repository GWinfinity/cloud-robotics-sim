"""Parallel seed search for expert data generation (migration doc 5.1).

Replaces RoboTwin's sequential per-seed search with batched sampling on a
Genesis scene built with ``n_envs=N``: each batch assigns one random
initialization per env, rolls the scene forward, and collects a per-env
success mask. Accepted seeds (with their env-local initial conditions) are
then replayed one-by-one for recording ("初始状态快照 + 动作序列回放",
doc risk R3).

The sampler is deliberately engine-agnostic: the caller provides three
callbacks operating on a scene built with :meth:`SceneBackend.build`:

- ``apply_seed(seed, env_idx)``: set the env-local initial state (object
  poses via ``set_pos(envs_idx=...)``, robot qpos, DR draws) derived from
  ``seed``;
- ``rollout()``: advance the batched scene (e.g. run the expert trajectory
  for all envs in parallel);
- ``evaluate()``: return a boolean array ``(n_batch,)`` with the per-env
  success mask for the current batch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["SeedOutcome", "SeedSearchResult", "batched_seed_search"]

ApplySeedFn = Callable[[int, int], None]
RolloutFn = Callable[[], None]
EvaluateFn = Callable[[], Any]


@dataclass
class SeedOutcome:
    """Result of evaluating one seed in one parallel env."""

    seed: int
    env_idx: int
    success: bool
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class SeedSearchResult:
    """Aggregated outcome of a batched seed search."""

    outcomes: list[SeedOutcome] = field(default_factory=list)
    n_batches: int = 0

    @property
    def accepted(self) -> list[int]:
        """Seeds that produced a successful rollout, in search order."""
        return [o.seed for o in self.outcomes if o.success]

    @property
    def acceptance_rate(self) -> float:
        """Fraction of evaluated seeds that succeeded."""
        if not self.outcomes:
            return 0.0
        return len(self.accepted) / len(self.outcomes)


def batched_seed_search(
    seeds: list[int] | np.ndarray,
    n_envs: int,
    apply_seed: ApplySeedFn,
    rollout: RolloutFn,
    evaluate: EvaluateFn,
    *,
    target_accepted: int | None = None,
) -> SeedSearchResult:
    """Evaluate seeds in batches of ``n_envs`` and collect successes.

    Args:
        seeds: Candidate seeds to evaluate, in priority order.
        n_envs: Batch size; must match the scene's built ``n_envs``.
        apply_seed: Callback ``(seed, env_idx) -> None`` installing the
            env-local initial conditions before each rollout.
        rollout: Callback advancing the batched scene once per batch.
        evaluate: Callback returning the per-env success mask for the
            current batch as a length-``n_batch`` boolean array.
        target_accepted: Optional early-stop threshold: stop as soon as at
            least this many seeds were accepted (doc 5.1: one failed seed
            must not block the batch).

    Returns:
        :class:`SeedSearchResult` with per-seed outcomes.
    """
    if n_envs < 1:
        raise ValueError("n_envs must be >= 1")
    seeds = [int(s) for s in seeds]
    result = SeedSearchResult()

    for batch_start in range(0, len(seeds), n_envs):
        batch = seeds[batch_start : batch_start + n_envs]
        for env_idx, seed in enumerate(batch):
            apply_seed(seed, env_idx)
        rollout()
        mask = np.asarray(evaluate(), dtype=bool).reshape(-1)
        if mask.shape[0] != len(batch):
            raise ValueError(
                f"evaluate() returned {mask.shape[0]} results, "
                f"expected {len(batch)} (one per env in the batch)"
            )
        for env_idx, (seed, ok) in enumerate(zip(batch, mask)):
            result.outcomes.append(
                SeedOutcome(seed=seed, env_idx=env_idx, success=bool(ok))
            )
        result.n_batches += 1
        logger.debug(
            "seed batch %d: %d/%d accepted",
            result.n_batches,
            int(mask.sum()),
            len(batch),
        )
        if target_accepted is not None and len(result.accepted) >= target_accepted:
            break

    logger.info(
        "seed search: %d/%d accepted in %d batches (n_envs=%d)",
        len(result.accepted),
        len(result.outcomes),
        result.n_batches,
        n_envs,
    )
    return result
