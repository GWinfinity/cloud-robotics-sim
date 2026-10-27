"""Seeded, vectorized domain randomization for Genesis rigid entities.

Genesis stores link / joint / dof model parameters per environment when
``RigidOptions(batch_links_info=True, batch_dofs_info=True)`` is set, and every
relevant entity setter accepts ``envs_idx``. This module adds the missing layer
on top: seeded, vectorized sampling of per-environment physics parameters, to be
called on episode reset inside an RL environment.

Supported targets (all applied through public entity setters):

* ``mass``            total entity mass, inertia auto-scaled (per env)
* ``friction_ratio``  per-link geom friction ratio, relative to authored friction (per env, per link)
* ``kp`` / ``kv``     PD gains (per env, per dof)
* ``armature`` / ``stiffness`` / ``damping`` / ``frictionloss``  dof dynamics (per env, per dof)

Values written through these setters survive ``scene.reset()``, so a randomizer
called after every reset gives every episode fresh physics. Bases for
multiplicative (``scale="relative"``) specs are snapshotted at construction, so
repeated ``sample`` calls do not random-walk the parameters.

Example:
--------
>>> randomizer = DomainRandomizer(
...     robot,
...     [
...         ParamSpec("mass", "mass", 0.8, 1.2),
...         ParamSpec("friction", "friction_ratio", 0.5, 1.5, scale="absolute"),
...         ParamSpec("kp", "kp", 0.9, 1.1),
...     ],
...     seed=1234,
... )
>>> sampled = randomizer.sample()          # all envs
>>> sampled = randomizer.sample(envs_idx)  # only reset envs

A config-dict bridge for legged-RL style configs is provided by
:meth:`DomainRandomizer.from_config` (``friction_range`` / ``mass_scale_range``
/ ``kp_range`` / ``kv_range`` / ...), so existing plugins can adopt the core
without rewriting their configuration files.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch

__all__ = ["ParamSpec", "DomainRandomizer", "draw_specs"]

# target -> (per-element granularity, requires which batched-info option)
_TARGETS: dict[str, tuple[str, str | None]] = {
    "mass": ("entity", "batch_links_info"),
    "friction_ratio": ("link", None),
    "kp": ("dof", "batch_dofs_info"),
    "kv": ("dof", "batch_dofs_info"),
    "armature": ("dof", "batch_dofs_info"),
    "stiffness": ("dof", "batch_dofs_info"),
    "damping": ("dof", "batch_dofs_info"),
    "frictionloss": ("dof", "batch_dofs_info"),
}

# dof targets with a public getter to snapshot a multiplicative base from.
_DOF_GETTERS = ("kp", "kv", "armature")

#: Keys recognised by :meth:`DomainRandomizer.from_config`, mapped to
#: ``(target, scale)``. Ranges are two-element ``[low, high]`` sequences.
_CONFIG_KEYS: dict[str, tuple[str, str]] = {
    "friction_range": ("friction_ratio", "absolute"),
    "mass_scale_range": ("mass", "relative"),
    "kp_range": ("kp", "relative"),
    "kv_range": ("kv", "relative"),
    "armature_range": ("armature", "relative"),
    "stiffness_range": ("stiffness", "absolute"),
    "damping_range": ("damping", "absolute"),
    "frictionloss_range": ("frictionloss", "absolute"),
}


def draw_specs(
    specs: Any,
    n_sel: int,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Sample every spec without touching an entity; returns numpy arrays.

    Shared by :meth:`DomainRandomizer.sample` and by lightweight callers that
    only need the sampled values (e.g. scalar per-episode randomization for
    non-Genesis objects). Relative-scale specs are returned as the raw sampled
    factor (no base multiplication, since no entity base is available).
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0 if seed is None else seed)
    out: dict[str, np.ndarray] = {}
    for spec in specs:
        shape = (n_sel, *_spec_element_shape(spec))
        if spec.distribution == "uniform":
            u = torch.rand(shape, generator=gen)
            values = spec.low + (spec.high - spec.low) * u
        elif spec.distribution == "log_uniform":
            u = torch.rand(shape, generator=gen)
            log_low = float(np.log(spec.low))
            log_high = float(np.log(spec.high))
            values = torch.exp(log_low + (log_high - log_low) * u)
        else:  # gaussian, 3-sigma clipped
            mean = (spec.low + spec.high) / 2.0
            std = (spec.high - spec.low) / 6.0
            g = torch.randn(shape, generator=gen)
            values = (mean + std * g).clamp(spec.low, spec.high)
        out[spec.name] = values.numpy()
    return out


def _spec_element_shape(spec: "ParamSpec") -> tuple[int, ...]:
    """Element shape for a spec without an entity: indices or a scalar."""
    if spec.indices is not None:
        return (len(spec.indices),)
    return ()


@dataclasses.dataclass(frozen=True)
class ParamSpec:
    """One randomizable physics parameter.

    Args:
        name:
            Key under which the sampled values appear in the dict returned by
            :meth:`DomainRandomizer.sample`.
        target:
            What to randomize; see the module docstring for the supported
            targets.
        low, high:
            Sampling bounds. For ``scale="relative"`` they bound the
            multiplicative factor around the base value captured at
            randomizer construction.
        distribution:
            ``"uniform"`` (default), ``"log_uniform"`` (positive ranges,
            uniform in log-space), or ``"gaussian"`` (mean at the interval
            center, std at 1/6 of its width, clipped to ``[low, high]``).
        scale:
            ``"relative"`` multiplies the snapshotted base value;
            ``"absolute"`` writes the sampled value directly (the natural
            choice for ``friction_ratio``, which is already a ratio).
        indices:
            Optional subset of local link indices (for ``friction_ratio``) or
            local dof indices (dof targets). ``None`` selects all.
    """

    name: str
    target: str
    low: float
    high: float
    distribution: str = "uniform"
    scale: str = "relative"
    indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the spec fields."""
        if self.target not in _TARGETS:
            raise ValueError(
                f"unknown target {self.target!r}, expected one of {sorted(_TARGETS)}"
            )
        if self.distribution not in ("uniform", "log_uniform", "gaussian"):
            raise ValueError(f"unknown distribution {self.distribution!r}")
        if self.scale not in ("relative", "absolute"):
            raise ValueError(f"unknown scale {self.scale!r}")
        if self.low > self.high:
            raise ValueError(f"low ({self.low}) must not exceed high ({self.high})")
        if self.distribution == "log_uniform" and (self.low <= 0 or self.high <= 0):
            raise ValueError("log_uniform requires positive bounds")


class DomainRandomizer:
    """Seeded, vectorized domain randomizer bound to a built ``RigidEntity``.

    Args:
        entity:
            A built ``genesis.engine.entities.RigidEntity``.
        specs:
            Iterable of :class:`ParamSpec`.
        seed:
            Seed of the private ``torch.Generator``. Two randomizers built with
            the same seed and the same base parameters sample identically.
    """

    def __init__(self, entity: Any, specs: Any, seed: int | None = None) -> None:
        self._entity = entity
        self._solver = entity._solver
        self._specs = tuple(specs)
        if not self._specs:
            raise ValueError("at least one ParamSpec is required")

        options = getattr(self._solver, "_options", None)
        n_envs_hint: int | None = None
        mass = entity.get_mass()
        if isinstance(mass, torch.Tensor) and mass.ndim > 0:
            n_envs_hint = int(mass.shape[0])
            mass_base = mass[0]
        else:
            mass_base = torch.as_tensor(mass, dtype=torch.float32, device=entity.device)
        self._device = mass_base.device

        for spec in self._specs:
            _, required = _TARGETS[spec.target]
            if (
                required is not None
                and options is not None
                and not getattr(options, required)
            ):
                raise RuntimeError(
                    f"ParamSpec {spec.name!r} targets {spec.target!r}, which needs "
                    f"per-environment {required}=True in RigidOptions to write "
                    "individual envs."
                )
            if spec.target == "mass" and float(mass_base.sum().item()) <= 0:
                raise RuntimeError(
                    "entity moves no mass; mass randomization is ill-defined"
                )

        self._n_envs = n_envs_hint if n_envs_hint is not None else 1
        self._bases: dict[str, torch.Tensor] = {"mass": mass_base}
        self._resolve_dof_bases()
        self.manual_seed(seed)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def manual_seed(self, seed: int | None) -> None:
        """Reseed the sampling generator; the next ``sample`` reproduces the first."""
        self._gen = torch.Generator(device=self._device)
        self._gen.manual_seed(0 if seed is None else seed)

    def sample(self, envs_idx: Any = None) -> dict[str, torch.Tensor]:
        """Sample every spec and write it, optionally to a subset of environments.

        Args:
            envs_idx:
                ``None`` writes all environments; otherwise the env indices
                (int tensor or array_like) that were just reset and need new
                physics.

        Returns:
            dict mapping each spec name to the values written, of shape
            ``(n_sel, ...)`` at the spec granularity. Kept on device for
            logging; convert with ``.detach().cpu().numpy()`` if needed.
        """
        if envs_idx is None:
            n_sel = self._n_envs
            envs_arg = None
        else:
            envs_arg = torch.as_tensor(
                envs_idx, dtype=torch.int64, device=self._device
            ).reshape(-1)
            n_sel = int(envs_arg.shape[0])

        out: dict[str, torch.Tensor] = {}
        for spec in self._specs:
            factor = self._draw(spec, n_sel)
            if spec.scale == "relative":
                base = self._base_for(spec, factor.shape)
                values = base * factor
            else:
                values = factor
            self._apply(spec, values, envs_arg)
            out[spec.name] = values
        return out

    @classmethod
    def from_config(
        cls,
        entity: Any,
        config: dict[str, Any],
        seed: int | None = None,
    ) -> "DomainRandomizer":
        """Build a randomizer from a legged-RL style config dict.

        Recognized keys (two-element ``[low, high]`` ranges):

        * ``friction_range``     -> ``friction_ratio`` (absolute)
        * ``mass_scale_range``   -> ``mass`` (relative)
        * ``kp_range`` / ``kv_range`` / ``armature_range`` -> dof targets (relative)
        * ``stiffness_range`` / ``damping_range`` / ``frictionloss_range``
          -> dof targets (absolute; genesis exposes no getter to snapshot a base from)

        Unknown keys are ignored. Raises if none of the recognized keys are
        present.
        """
        specs = [
            ParamSpec(
                name=key.replace("_range", ""),
                target=target,
                low=float(config[key][0]),
                high=float(config[key][1]),
                scale=scale,
            )
            for key, (target, scale) in _CONFIG_KEYS.items()
            if key in config and config[key] is not None
        ]
        if not specs:
            raise ValueError(
                f"none of the recognized range keys {sorted(_CONFIG_KEYS)} found in config"
            )
        return cls(entity, specs, seed=seed)

    # ----------------------------------------------------------------- #
    # internals
    # ----------------------------------------------------------------- #

    def _draw(self, spec: ParamSpec, n_sel: int) -> torch.Tensor:
        shape = (n_sel, *self._element_shape(spec))
        if spec.distribution == "uniform":
            u = torch.rand(shape, generator=self._gen, device=self._device)
            return spec.low + (spec.high - spec.low) * u
        if spec.distribution == "log_uniform":
            u = torch.rand(shape, generator=self._gen, device=self._device)
            log_low = float(np.log(spec.low))
            log_high = float(np.log(spec.high))
            return torch.exp(log_low + (log_high - log_low) * u)
        # gaussian, 3-sigma clipped
        mean = (spec.low + spec.high) / 2.0
        std = (spec.high - spec.low) / 6.0
        g = torch.randn(shape, generator=self._gen, device=self._device)
        return (mean + std * g).clamp(spec.low, spec.high)

    def _element_shape(self, spec: ParamSpec) -> tuple[int, ...]:
        if spec.indices is not None:
            return (len(spec.indices),)
        granularity = _TARGETS[spec.target][0]
        if granularity == "link":
            return (self._entity.n_links,)
        if granularity == "dof":
            return (self._entity.n_dofs,)
        return ()

    def _base_for(self, spec: ParamSpec, shape: tuple[int, ...]) -> torch.Tensor:
        if spec.target == "mass":
            return self._bases["mass"].reshape(()).expand(shape)
        base = self._bases[spec.target]
        if spec.indices is not None:
            base = base[list(spec.indices)]
        return base.reshape((1, *base.shape)).expand(shape)

    def _apply(self, spec: ParamSpec, values: torch.Tensor, envs_arg: Any) -> None:
        entity = self._entity
        if spec.target == "mass":
            entity.set_mass(values, envs_idx=envs_arg)
        elif spec.target == "friction_ratio":
            entity.set_friction_ratio(
                values, links_idx_local=spec.indices, envs_idx=envs_arg
            )
        else:
            setter = {
                "kp": entity.set_dofs_kp,
                "kv": entity.set_dofs_kv,
                "armature": entity.set_dofs_armature,
                "stiffness": entity.set_dofs_stiffness,
                "damping": entity.set_dofs_damping,
                "frictionloss": entity.set_dofs_frictionloss,
            }[spec.target]
            setter(values, dofs_idx_local=spec.indices, envs_idx=envs_arg)

    def _resolve_dof_bases(self) -> None:
        getters = {
            "kp": self._entity.get_dofs_kp,
            "kv": self._entity.get_dofs_kv,
            "armature": self._entity.get_dofs_armature,
        }
        needed = {spec.target for spec in self._specs} & set(getters)
        for target in needed:
            base = getters[target]()
            if isinstance(base, torch.Tensor) and base.ndim > 1:
                base = base[0]
            self._bases[target] = torch.as_tensor(base, device=self._device)
        # dof targets without public getters take an explicit base from
        # env_cfg-style knowledge; fall back to ones and require absolute scale.
        for spec in self._specs:
            if _TARGETS[spec.target][0] == "dof" and spec.target not in self._bases:
                if spec.scale == "relative":
                    raise ValueError(
                        f"target {spec.target!r} has no public getter to snapshot "
                        "a base from; use scale='absolute' with explicit bounds"
                    )
                self._bases[spec.target] = torch.ones(
                    self._element_shape(spec), device=self._device
                )
