"""Adaptive RSI (Reference State Initialization) sampling.

Ported from wbc_lab/env/mdp/sampling.py  -implements similarity-weighted
adaptive bin sampling for curriculum-based motion tracking training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveSimilarityTermCfg:
    """Configuration for one similarity term used in RSI weighting."""

    term: str
    """Observation/reward term name."""
    weight: float = 1.0
    """Weight in the similarity computation."""


@dataclass
class RsiCfg:
    """Configuration for Reference State Initialization."""

    sampling_mode: str = "adaptive"
    """'adaptive' (similarity-weighted) or 'uniform'."""
    strategy: str = "similarity_ema"
    """Sampling strategy: 'similarity_ema' or 'uniform'."""
    similarity_terms: tuple[AdaptiveSimilarityTermCfg, ...] = (
        AdaptiveSimilarityTermCfg(term="joint_pos", weight=1.0),
    )
    bin_width_s: float = 4.0
    """Bin width in seconds."""
    uniform_ratio: float = 0.15
    """Fraction of samples drawn uniformly (exploration)."""
    alpha: float = 0.005
    """EMA smoothing factor for failure levels."""
    temperature_base: float = 1.0
    """Base temperature for softmax sampling over bins."""
    min_bin_span_ratio: float = 0.0
    """Minimum bin span as fraction of total bins."""


@dataclass
class TrackingSimilarityState:
    """Per-environment tracking similarity state."""

    # EMA failure levels per bin
    failure_levels: np.ndarray | None = None
    # Current bin index
    current_bin: int = 0


class AdaptiveRsiSampler:
    """Adaptive RSI sampler that weights bin selection by failure history.

    Bins divide the motion clip into temporal segments. Bins with higher
    failure rates (harder regions) get sampled more frequently, creating
    a natural curriculum.
    """

    def __init__(
        self,
        cfg: RsiCfg,
        clip_duration_s: float,
        fps: float = 30.0,
    ):
        self.cfg = cfg
        self.clip_duration_s = clip_duration_s
        self.fps = fps
        self.total_frames = int(clip_duration_s * fps)

        # Compute bins
        bin_frames = int(cfg.bin_width_s * fps)
        self.num_bins = max(1, self.total_frames // bin_frames)
        self.bin_width_frames = self.total_frames // self.num_bins

        # Failure levels (EMA)
        self.failure_levels = np.ones(self.num_bins, dtype=np.float64)

        # Sampling probability cache
        self._prob_cache_valid = False
        self._cached_probs: np.ndarray | None = None

    def reset(self) -> None:
        """Reset failure levels to uniform."""
        self.failure_levels[:] = 1.0
        self._prob_cache_valid = False

    def bin_for_frame(self, frame: int) -> int:
        """Get bin index for a given frame."""
        return min(frame // self.bin_width_frames, self.num_bins - 1)

    def sample_start_frame(self, rng: np.random.Generator | None = None) -> int:
        """Sample a start frame for RSI reset.

        Returns:
            Frame index to initialize from.
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.cfg.sampling_mode == "uniform" or self.failure_levels is None:
            return int(rng.integers(0, self.total_frames))

        # Mix: uniform_ratio fraction drawn uniformly, rest from adaptive
        if rng.random() < self.cfg.uniform_ratio:
            return int(rng.integers(0, self.total_frames))

        # Compute softmax probabilities over bins
        probs = self._compute_bin_probs()
        bin_idx = rng.choice(self.num_bins, p=probs)

        # Sample uniformly within selected bin
        start = bin_idx * self.bin_width_frames
        end = min(start + self.bin_width_frames, self.total_frames)
        return int(rng.integers(start, max(start + 1, end)))

    def _compute_bin_probs(self) -> np.ndarray:
        """Compute sampling probabilities via softmax over failure levels."""
        if self._prob_cache_valid and self._cached_probs is not None:
            return self._cached_probs

        # Softmax with temperature
        temp = self.cfg.temperature_base
        logits = self.failure_levels / max(temp, 1e-8)
        logits -= logits.max()  # numerical stability
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum()

        self._cached_probs = probs
        self._prob_cache_valid = True
        return probs

    def update_failure(
        self,
        bin_idx: int,
        failure_delta: float,
    ) -> None:
        """Update failure level for a bin using EMA.

        Args:
            bin_idx: Bin index.
            failure_delta: New failure signal (0 = success, >0 = failure).
        """
        alpha = self.cfg.alpha
        self.failure_levels[bin_idx] = (
            (1 - alpha) * self.failure_levels[bin_idx] + alpha * failure_delta
        )
        self._prob_cache_valid = False

    def step_tracking_similarity(
        self,
        reward_terms: dict[str, float],
        bin_idx: int,
    ) -> float:
        """Compute step-level tracking similarity and update failure levels.

        Args:
            reward_terms: Dict of reward term name -> value.
            bin_idx: Current bin index.

        Returns:
            Similarity score (0-1, higher = better tracking).
        """
        total_weight = 0.0
        weighted_similarity = 0.0

        for term_cfg in self.cfg.similarity_terms:
            if term_cfg.term in reward_terms:
                val = reward_terms[term_cfg.term]
                # Clamp to [0, 1] range for similarity
                similarity = max(0.0, min(1.0, val))
                weighted_similarity += term_cfg.weight * similarity
                total_weight += abs(term_cfg.weight)

        if total_weight > 0:
            similarity = weighted_similarity / total_weight
        else:
            similarity = 1.0

        # Update failure level: failure = 1 - similarity
        self.update_failure(bin_idx, 1.0 - similarity)

        return similarity

    def save_state(self) -> dict[str, Any]:
        """Save RSI state for checkpoint persistence."""
        return {
            "failure_levels": self.failure_levels.copy(),
            "num_bins": self.num_bins,
            "bin_width_frames": self.bin_width_frames,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load RSI state from checkpoint."""
        if "failure_levels" in state:
            fl = state["failure_levels"]
            if len(fl) == self.num_bins:
                self.failure_levels = fl.copy()
            else:
                logger.warning(
                    "RSI bin count mismatch: checkpoint=%d, current=%d. Ignoring.",
                    len(fl), self.num_bins,
                )
        self._prob_cache_valid = False
