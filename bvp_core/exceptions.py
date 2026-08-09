"""Numerical failure types shared by the headless BVP core."""

from __future__ import annotations

from typing import Any

import numpy as np

from .models import BVPValidationError


class ExpressionValidationError(BVPValidationError):
    """Structured rejection raised at the mathematical-expression boundary."""

    PREVIEW_LIMIT = 160

    def __init__(
        self,
        *,
        context: str,
        expression: object,
        error_code: str,
        message: str,
        field_index: int | None = None,
        position: int | None = None,
    ) -> None:
        raw_expression = expression if isinstance(expression, str) else repr(expression)
        preview = raw_expression.replace("\r", "\\r").replace("\n", "\\n")
        if len(preview) > self.PREVIEW_LIMIT:
            preview = preview[: self.PREVIEW_LIMIT - 1] + "…"
        self.context = str(getattr(context, "value", context))
        self.expression_preview = preview
        self.expression = preview
        self.error_code = str(error_code)
        self.human_message = str(message)
        self.reason = self.human_message
        self.field_index = field_index
        self.position = position
        location = self.context
        if field_index is not None:
            location += f" expression #{field_index + 1}"
        if position is not None:
            location += f" at column {position}"
        diagnostic = (
            f"[{self.error_code}] {location}: {self.human_message}; "
            f"expression={self.expression_preview!r}"
        )
        super().__init__(diagnostic)


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
