"""Deterministic solver counters and immutable performance snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class SolverCounterSnapshot:
    """Immutable counters captured at one solver boundary."""

    phi_evaluations: int = 0
    jacobian_evaluations: int = 0
    ivp_solves: int = 0
    phi_ivp_solves: int = 0
    variational_ivp_solves: int = 0
    final_validation_ivp_solves: int = 0
    total_ivp_nfev: int = 0
    variational_ivp_nfev: int = 0
    final_validation_ivp_nfev: int = 0
    root_residual_calls: int = 0
    least_squares_residual_calls: int = 0
    continuation_steps_attempted: int = 0
    continuation_steps_completed: int = 0
    newton_iterations: int = 0
    newton_updates: int = 0
    damping_trials: int = 0
    differential_rhs_evaluations: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class SolverCounters:
    """Mutable, solver-local counters; callers only receive frozen snapshots."""

    phi_evaluations: int = 0
    jacobian_evaluations: int = 0
    ivp_solves: int = 0
    phi_ivp_solves: int = 0
    variational_ivp_solves: int = 0
    final_validation_ivp_solves: int = 0
    total_ivp_nfev: int = 0
    variational_ivp_nfev: int = 0
    final_validation_ivp_nfev: int = 0
    root_residual_calls: int = 0
    least_squares_residual_calls: int = 0
    continuation_steps_attempted: int = 0
    continuation_steps_completed: int = 0
    newton_iterations: int = 0
    newton_updates: int = 0
    damping_trials: int = 0
    differential_rhs_evaluations: int = 0

    def record_ivp_start(self, *, purpose: str) -> None:
        self.ivp_solves += 1
        if purpose == "phi":
            self.phi_ivp_solves += 1
        elif purpose == "variational":
            self.variational_ivp_solves += 1
        elif purpose == "final_validation":
            self.final_validation_ivp_solves += 1

    def record_ivp_result(self, solution: Any, *, purpose: str) -> None:
        nfev = getattr(solution, "nfev", None)
        if nfev is None:
            return
        value = int(nfev)
        self.total_ivp_nfev += value
        if purpose == "variational":
            self.variational_ivp_nfev += value
        elif purpose == "final_validation":
            self.final_validation_ivp_nfev += value

    def snapshot(self) -> SolverCounterSnapshot:
        return SolverCounterSnapshot(**asdict(self))

    def frozen_mapping(self) -> Mapping[str, int]:
        return MappingProxyType(self.snapshot().to_dict())
