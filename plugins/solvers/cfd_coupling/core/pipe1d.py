"""1D pipe transient solver: method of characteristics (MOC) water hammer.

Single pipe with a constant-head upstream reservoir and a downstream
nozzle/valve boundary. The nozzle boundary accepts a time-varying backpressure
head, which is what the 3D CFD side feeds back in the coupled simulation.

Governing equations (Allievi, friction in quasi-steady Darcy-Weisbach form):

    dH/dt + (a^2 / gA) dQ/dt                = 0          along C+  (dx/dt = +a)
    dH/dt - (a^2 / gA) dQ/dt + f Q|Q| ...   = 0          along C-  (dx/dt = -a)

With dx = a*dt (Courant = 1) the interior update is exact. The downstream
nozzle couples the pipe to a plenum with head H_back:

    H_N     = C_P - B * Q_N                    (C+ characteristic at valve)
    Q_N     = Cd * Av * sqrt(2 g (H_N - H_back))

which has a closed-form solution (quadratic in Q_N), see ``_solve_nozzle``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.81  # gravitational acceleration [m/s^2]


@dataclass
class PipeOptions:
    """Options for the 1D MOC pipe solver."""

    length: float = 100.0  # pipe length [m]
    diameter: float = 0.1  # pipe diameter [m]
    n_reaches: int = 50  # number of reaches (cells)
    wave_speed: float = 1000.0  # pressure wave speed [m/s]
    friction: float = 0.02  # Darcy-Weisbach friction factor [-]
    reservoir_head: float = 50.0  # upstream constant reservoir head [m]
    discharge_head: float = 0.0  # fixed backpressure head (standalone mode) [m]
    nozzle_area: float = 1.0e-3  # nozzle area [m^2]
    discharge_coeff: float = 0.8  # nozzle discharge coefficient [-]
    fluid_temperature: float = 300.0  # fluid temperature passed to the 3D side [K]


class Pipe1D:
    """Single-pipe MOC solver with reservoir inlet and nozzle outlet."""

    def __init__(self, options: PipeOptions) -> None:
        o = options
        self.o = o
        self.dx = o.length / o.n_reaches
        self.dt = self.dx / o.wave_speed  # Courant = 1 by construction
        self.area = np.pi * o.diameter**2 / 4.0
        # Characteristic impedance B [s/m^2] and friction coefficient R [s^2/m^5]
        self.B = o.wave_speed / (G * self.area)
        self.R = o.friction * self.dx / (2.0 * G * o.diameter * self.area**2)

        n = o.n_reaches + 1
        self.H = np.zeros(n)  # head at nodes 0..N [m]
        self.Q = np.zeros(n)  # discharge at nodes 0..N [m^3/s]
        self.t = 0.0
        self.n_steps = 0
        self._init_steady()

    # ------------------------------------------------------------------ #
    def _init_steady(self) -> None:
        """Steady-state initial condition compatible with the boundaries."""
        o = self.o
        k_pipe = o.friction * o.length / (2.0 * G * o.diameter * self.area**2)
        k_valve = 1.0 / (2.0 * G * (o.discharge_coeff * o.nozzle_area) ** 2)
        dh = o.reservoir_head - o.discharge_head
        if dh <= 0.0:
            raise ValueError("reservoir_head must exceed discharge_head")
        q0 = np.sqrt(dh / (k_pipe + k_valve))
        self.Q[:] = q0
        # Head drops linearly (Darcy-Weisbach) from reservoir to valve
        drop = k_pipe * q0**2 * np.linspace(0.0, 1.0, o.n_reaches + 1)
        self.H[:] = o.reservoir_head - drop

    # ------------------------------------------------------------------ #
    def step(
        self,
        backpressure_head: float | None = None,
        valve_fraction: float = 1.0,
    ) -> float:
        """Advance one MOC step; returns the nozzle discharge Q_N [m^3/s].

        Parameters
        ----------
        backpressure_head:
            Plenum head downstream of the nozzle [m]. Defaults to the fixed
            ``discharge_head`` option (standalone water-hammer mode).
        valve_fraction:
            Valve opening area fraction in (0, 1]; scales the nozzle area.
        """
        if backpressure_head is None:
            backpressure_head = self.o.discharge_head
        h_back = float(backpressure_head)
        frac = min(max(float(valve_fraction), 1.0e-6), 1.0)

        H, Q, B, R = self.H, self.Q, self.B, self.R
        N = self.o.n_reaches

        # Characteristic constants arriving at each node.
        Cp = H[:-1] + B * Q[:-1] - R * Q[:-1] * np.abs(Q[:-1])  # C+ : node i-1 -> i
        Cm = H[1:] - B * Q[1:] + R * Q[1:] * np.abs(Q[1:])  # C- : node i+1 -> i

        Hn = np.empty_like(H)
        Qn = np.empty_like(Q)

        # Interior nodes 1..N-1
        Qn[1:N] = (Cp[1:N] - Cm[: N - 1]) / (2.0 * B)
        Hn[1:N] = 0.5 * (Cp[1:N] + Cm[: N - 1])

        # Upstream node 0: constant-head reservoir (H0 known, Q follows C-)
        Hn[0] = self.o.reservoir_head
        Qn[0] = (Hn[0] - Cm[0]) / B

        # Downstream node N: nozzle with plenum backpressure (C+ known)
        q_nozzle = self._solve_nozzle(Cp[N - 1], h_back, frac)
        Qn[N] = q_nozzle
        Hn[N] = Cp[N - 1] - B * q_nozzle

        self.H, self.Q = Hn, Qn
        self.t += self.dt
        self.n_steps += 1
        return q_nozzle

    def _solve_nozzle(self, cp: float, h_back: float, frac: float) -> float:
        """Nozzle flow from C+ constant and plenum head (closed form).

        Solves  B*Q + Q^2 / (2 g S^2) = C_P - H_back  with S = Cd * Av * frac.
        """
        S = self.o.discharge_coeff * self.o.nozzle_area * frac
        d = cp - h_back
        if d <= 0.0:
            return 0.0
        return G * S**2 * (np.sqrt(self.B**2 + 2.0 * d / (G * S**2)) - self.B)

    # ------------------------------------------------------------------ #
    def nozzle_discharge(self) -> float:
        return float(self.Q[-1])

    def get_state(self) -> tuple[np.ndarray, np.ndarray, float, int]:
        return self.H.copy(), self.Q.copy(), self.t, self.n_steps

    def set_state(self, state: tuple[np.ndarray, np.ndarray, float, int]) -> None:
        H, Q, t, n = state
        self.H = H.copy()
        self.Q = Q.copy()
        self.t = t
        self.n_steps = n
