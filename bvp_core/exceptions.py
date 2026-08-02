"""Numerical failure types shared by the headless BVP core."""

from __future__ import annotations

from typing import Any

import numpy as np


class IVPIntegrationError(RuntimeError):
    """Failure of the inner initial-value problem with preserved SciPy diagnostics."""

    def __init__(
        self, message: str, solution: Any = None, p: np.ndarray | None = None
    ) -> None:
        self.solution = solution
        self.p = None if p is None else np.asarray(p, dtype=float).copy()
        self.ivp_success = bool(getattr(solution, "success", False))
        self.ivp_status = getattr(solution, "status", None)
        self.ivp_message = str(getattr(solution, "message", message))
        t_values = np.asarray(getattr(solution, "t", []), dtype=float)
        y_values = np.asarray(getattr(solution, "y", []), dtype=float)
        self.t_final = float(t_values[-1]) if t_values.size else None
        self.state_finite = bool(
            solution is not None
            and np.isfinite(t_values).all()
            and np.isfinite(y_values).all()
        )
        super().__init__(message)
