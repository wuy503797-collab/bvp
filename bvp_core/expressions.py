"""Restricted SymPy expression handling for the headless BVP core."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify

from .exceptions import ExpressionValidationError
from .expression_policy import (
    ALLOWED_CONSTANTS,
    ALLOWED_FUNCTIONS,
    ExpressionContext,
    parse_restricted_expression,
    validate_identifiers,
)


class SymPyParser:
    """Parse the project's restricted math language and expose NumPy callables."""

    ALLOWED_NAMES = frozenset(ALLOWED_FUNCTIONS) | frozenset(ALLOWED_CONSTANTS)

    def __init__(self, equations: Sequence[str], var_names: Sequence[str]):
        self.equations_str = list(equations)
        self.var_names = list(var_names)
        self.n = len(self.equations_str)
        validate_identifiers(self.var_names, context=ExpressionContext.ODE)
        if len(self.var_names) != self.n:
            raise ExpressionValidationError(
                context=ExpressionContext.ODE.value,
                expression=", ".join(self.equations_str),
                error_code="equation_count_mismatch",
                message=(
                    f"equation count {self.n} does not match variable count "
                    f"{len(self.var_names)}"
                ),
            )
        self.t_sym = sp.Symbol("t", real=True)
        self.x_syms = [sp.Symbol(name, real=True) for name in self.var_names]
        self.all_syms = [self.t_sym, *self.x_syms]
        self._symbol_map = {symbol.name: symbol for symbol in self.all_syms}
        self.parsed_exprs: list[sp.Expr] = []
        self.jacobian_exprs: sp.Matrix | None = None
        self.ode_fn: Callable | None = None
        self.jac_fn: Callable | None = None

    def parse(self) -> None:
        """Validate and parse all ODE right-hand sides, then build the Jacobian."""
        self.parsed_exprs = [
            parse_restricted_expression(
                expression,
                context=ExpressionContext.ODE,
                allowed_symbols=self._symbol_map,
                field_index=index,
            )
            for index, expression in enumerate(self.equations_str)
        ]
        self.jacobian_exprs = sp.Matrix(self.parsed_exprs).jacobian(self.x_syms)

    def lambdify_all(self) -> None:
        """Create NumPy-compatible ODE and Jacobian functions."""
        if not self.parsed_exprs:
            self.parse()
        try:
            self.ode_fn = lambdify(
                self.all_syms, self.parsed_exprs, modules="numpy"
            )
            if self.jacobian_exprs is not None:
                self.jac_fn = lambdify(
                    self.all_syms, list(self.jacobian_exprs), modules="numpy"
                )
        except Exception as exc:
            raise ExpressionValidationError(
                context=ExpressionContext.ODE.value,
                expression=", ".join(self.equations_str),
                error_code="lambdify_failed",
                message=f"numeric ODE construction failed: {type(exc).__name__}: {exc}",
            ) from exc

    def f(self, t: float, x: np.ndarray) -> np.ndarray:
        """Evaluate ``dx/dt = f(t, x)``."""
        if self.ode_fn is None:
            self.lambdify_all()
        values = np.asarray(self.ode_fn(t, *list(x)), dtype=float)
        if values.shape != (self.n,):
            raise ValueError(
                f"ODE output shape {values.shape} does not match ({self.n},)"
            )
        return values

    def jac(self, t: float, x: np.ndarray) -> np.ndarray:
        """Evaluate the state Jacobian of the ODE right-hand sides."""
        if self.jac_fn is None:
            self.lambdify_all()
        flat = np.array(self.jac_fn(t, *list(x)), dtype=float)
        return flat.reshape(self.n, self.n)

    def parse_aux(self, aux_outputs: Mapping[str, str]) -> dict[str, sp.Expr]:
        """Validate and parse auxiliary output expressions."""
        names = list(aux_outputs)
        validate_identifiers(names, context=ExpressionContext.AUXILIARY)
        return {
            name: parse_restricted_expression(
                expression,
                context=ExpressionContext.AUXILIARY,
                allowed_symbols=self._symbol_map,
                field_index=index,
            )
            for index, (name, expression) in enumerate(aux_outputs.items())
        }

    def lambdify_aux(self, aux_outputs: Mapping[str, str]) -> dict[str, Callable]:
        """Create NumPy-compatible auxiliary output functions."""
        functions: dict[str, Callable] = {}
        parsed = self.parse_aux(aux_outputs)
        for index, (name, expression) in enumerate(parsed.items()):
            try:
                functions[name] = lambdify(
                    self.all_syms, expression, modules="numpy"
                )
            except Exception as exc:
                raise ExpressionValidationError(
                    context=ExpressionContext.AUXILIARY.value,
                    expression=aux_outputs[name],
                    error_code="lambdify_failed",
                    message=(
                        f"numeric auxiliary construction failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    field_index=index,
                ) from exc
        return functions

    @staticmethod
    def parse_boundary_conditions(
        bc_strings: Sequence[str], var_names: Sequence[str]
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """Build ``residual(x_start, x_end)`` from restricted BC expressions."""
        validate_identifiers(list(var_names), context=ExpressionContext.ODE)
        dimension = len(var_names)
        symbols_0 = [sp.Symbol(f"x{index}_0", real=True) for index in range(dimension)]
        symbols_T = [sp.Symbol(f"x{index}_T", real=True) for index in range(dimension)]
        all_symbols = [*symbols_0, *symbols_T]
        symbol_map = {symbol.name: symbol for symbol in all_symbols}
        parsed = [
            parse_restricted_expression(
                expression,
                context=ExpressionContext.BOUNDARY,
                allowed_symbols=symbol_map,
                field_index=index,
            )
            for index, expression in enumerate(bc_strings)
        ]
        if not parsed:
            raise ExpressionValidationError(
                context=ExpressionContext.BOUNDARY.value,
                expression="",
                error_code="empty_boundary_conditions",
                message="at least one boundary condition is required",
            )
        try:
            boundary_function = lambdify(all_symbols, parsed, modules="numpy")
        except Exception as exc:
            raise ExpressionValidationError(
                context=ExpressionContext.BOUNDARY.value,
                expression=", ".join(bc_strings),
                error_code="lambdify_failed",
                message=(
                    f"numeric boundary construction failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            ) from exc

        def residual(x0: np.ndarray, xT: np.ndarray) -> np.ndarray:
            values = np.asarray(
                boundary_function(*list(x0), *list(xT)), dtype=float
            )
            if values.shape != (len(parsed),):
                raise ValueError(
                    "boundary residual shape "
                    f"{values.shape} does not match ({len(parsed)},)"
                )
            return values

        return residual

    @staticmethod
    def parse_scalar(expression: str) -> sp.Expr:
        """Parse a constant scalar used by GUI initial-guess fields."""
        return parse_restricted_expression(
            expression,
            context=ExpressionContext.SCALAR,
            allowed_symbols={},
        )
