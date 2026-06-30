"""Multi-rate SE(3) action composer."""

from __future__ import annotations

import numpy as np

from .math_utils import compose, identity
from .specs import ComposeSpec


class ActionComposer:
    """Compose semantic anchor, predictive motion, and reactive residual.

    Implements Equations (1) and (2) from CoStream:

        W T_nkom(t) = W T_Ik  *  I T_traj(t)
        W T_cmd(t)  = W T_nkom(t)  *  T_tac(t)
    """

    def __init__(self) -> None:
        self._last_anchor = identity()
        self._last_nominal = identity()

    def compose(
        self,
        anchor: np.ndarray | None,
        nominal: np.ndarray | None,
        residual: np.ndarray | None,
        compose_spec: ComposeSpec,
    ) -> np.ndarray:
        """Return composed end-effector command W_T_cmd.

        Missing or stale streams degrade gracefully according to
        compose_spec.fallback.
        """
        # Latch or hold anchor.
        if anchor is None:
            if compose_spec.fallback.get("missing_anchor") == "identity":
                anchor = identity()
            else:
                anchor = self._last_anchor
        else:
            self._last_anchor = np.asarray(anchor, dtype=float)

        # Latch or hold nominal motion.
        if nominal is None:
            if compose_spec.fallback.get("missing_nominal") == "identity":
                nominal = identity()
            else:
                nominal = self._last_nominal
        else:
            self._last_nominal = np.asarray(nominal, dtype=float)

        # Residual fallback.
        if residual is None:
            residual = identity()
        else:
            residual = np.asarray(residual, dtype=float)
            if compose_spec.fallback.get("missing_residual") == "zero":
                # If caller explicitly passed None we already handled it above.
                pass

        W_T_nkom = compose(anchor, nominal)
        W_T_cmd = compose(W_T_nkom, residual)
        return W_T_cmd

    def reset(self) -> None:
        """Clear latched transforms, e.g. on stage transition."""
        self._last_anchor = identity()
        self._last_nominal = identity()
