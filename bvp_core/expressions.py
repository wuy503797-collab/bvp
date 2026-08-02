"""SymPy-backed expression parsing for the headless BVP core."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations
from sympy.utilities.lambdify import lambdify

class SymPyParser:
    """
    Безопасный парсер математических выражений через SymPy.
    Преобразует список строк уравнений в callable NumPy-функцию.
    """

    # Разрешённые символы — белый список
    ALLOWED_NAMES = {
        "sin", "cos", "tan", "exp", "log", "sqrt",
        "pi", "E", "abs", "sign", "atan", "asin", "acos",
        "sinh", "cosh", "tanh", "atan2",
        # константы
        "inf", "nan",
    }

    def __init__(self, equations: List[str], var_names: List[str]):
        """
        equations: список строк правых частей ODE
        var_names: имена переменных состояния (x0, x1, ...)
        """
        self.equations_str = equations
        self.var_names = var_names
        self.n = len(equations)
        # SymPy символы: t, x0, x1, ...
        self.t_sym = sp.Symbol("t", real=True)
        self.x_syms = [sp.Symbol(v, real=True) for v in var_names]
        self.all_syms = [self.t_sym] + self.x_syms
        # Распарсенные выражения
        self.parsed_exprs: List[sp.Expr] = []
        # Якобиан (n x n матрица выражений)
        self.jacobian_exprs: Optional[sp.Matrix] = None
        # lambdified функции
        self.ode_fn: Optional[Callable] = None
        self.jac_fn: Optional[Callable] = None

    def parse(self) -> None:
        """Парсинг всех выражений и построение Якобиана."""
        local_dict = {s.name: s for s in self.all_syms}
        # Разрешённые функции и типы из SymPy (нужны для standard_transformations)
        global_dict = {name: getattr(sp, name) for name in self.ALLOWED_NAMES
                       if hasattr(sp, name)}
        global_dict.update({
            "pi": sp.pi, "E": sp.E, "inf": sp.oo,
            "Integer": sp.Integer, "Float": sp.Float, "Rational": sp.Rational,
        })

        self.parsed_exprs = []
        for eq_str in self.equations_str:
            expr = parse_expr(
                eq_str,
                local_dict=local_dict,
                global_dict=global_dict,
                transformations=standard_transformations,
                evaluate=True,
            )
            self.parsed_exprs.append(expr)

        # Якобиан: ∂f_i/∂x_j
        self.jacobian_exprs = sp.Matrix(self.parsed_exprs).jacobian(self.x_syms)

    def lambdify_all(self) -> None:
        """Преобразование в NumPy-совместимые функции."""
        if not self.parsed_exprs:
            self.parse()

        # ODE функция: f(t, x) -> array(n)
        self.ode_fn = lambdify(self.all_syms, self.parsed_exprs, modules="numpy")

        # Якобиан: J(t, x) -> array(n, n)
        if self.jacobian_exprs is not None:
            flat_jac = list(self.jacobian_exprs)
            self.jac_fn = lambdify(self.all_syms, flat_jac, modules="numpy")

    def f(self, t: float, x: np.ndarray) -> np.ndarray:
        """Обертка: f(t, x) -> dx/dt"""
        if self.ode_fn is None:
            self.lambdify_all()
        args = [t] + list(x)
        return np.array(self.ode_fn(*args), dtype=float)

    def jac(self, t: float, x: np.ndarray) -> np.ndarray:
        """Обертка: Jacobian J(t,x) = ∂f/∂x"""
        if self.jac_fn is None:
            self.lambdify_all()
        args = [t] + list(x)
        flat = np.array(self.jac_fn(*args), dtype=float)
        return flat.reshape(self.n, self.n)

    # ---- вспомогательные выходные переменные ----

    def parse_aux(self, aux_outputs: Dict[str, str]) -> Dict[str, sp.Expr]:
        """Парсит вспомогательные переменные. Возвращает {name: expr}."""
        local_dict = {s.name: s for s in self.all_syms}
        global_dict = {name: getattr(sp, name) for name in self.ALLOWED_NAMES
                       if hasattr(sp, name)}
        global_dict.update({
            "pi": sp.pi, "E": sp.E, "inf": sp.oo,
            "Integer": sp.Integer, "Float": sp.Float, "Rational": sp.Rational,
        })
        parsed = {}
        for name, expr_str in aux_outputs.items():
            expr = parse_expr(
                expr_str,
                local_dict=local_dict,
                global_dict=global_dict,
                transformations=standard_transformations,
                evaluate=True,
            )
            parsed[name] = expr
        return parsed

    def lambdify_aux(self, aux_outputs: Dict[str, str]) -> Dict[str, Callable]:
        """lambdify вспомогательных переменных. Возвращает {name: fn(t,x0,x1,...)}."""
        parsed = self.parse_aux(aux_outputs)
        fns = {}
        for name, expr in parsed.items():
            fns[name] = lambdify(self.all_syms, expr, modules="numpy")
        return fns

    @staticmethod
    def parse_boundary_conditions(
        bc_strings: List[str],
        var_names: List[str]
    ) -> Callable:
        """
        Парсит граничные условия вида ["x0_T - 1.0738644361", "x2_T + 1.0995343576"]
        и возвращает функцию residual(x0, xT) -> np.ndarray.

        Переменные в BC (универсальный формат по индексам):
            x0_0, x1_0, ..., x{n-1}_0 — значения в t=0
            x0_T, x1_T, ..., x{n-1}_T — значения в t=T
        """
        n = len(var_names)
        # Используем ТОЛЬКО индексные имена — универсальные и не зависят от var_names
        symbols_0 = [sp.Symbol(f"x{i}_0", real=True) for i in range(n)]
        symbols_T = [sp.Symbol(f"x{i}_T", real=True) for i in range(n)]
        all_bc_syms = symbols_0 + symbols_T

        local_dict = {s.name: s for s in all_bc_syms}
        # Только необходимое — без Integer/Float чтобы избежать Symbol not defined
        global_dict = {
            "pi": sp.pi, "E": sp.E,
            "Integer": sp.Integer, "Float": sp.Float, "Rational": sp.Rational,
            "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
            "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
            "abs": sp.Abs, "sign": sp.sign,
            "atan": sp.atan, "asin": sp.asin, "acos": sp.acos,
            "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
            "atan2": sp.atan2,
        }

        parsed_bcs = []
        for bc_str in bc_strings:
            s_clean = bc_str.strip()
            if s_clean == "":
                continue
            try:
                expr = parse_expr(
                    s_clean,
                    local_dict=local_dict,
                    global_dict=global_dict,
                    transformations=standard_transformations,
                    evaluate=True,
                )
                parsed_bcs.append(expr)
            except Exception as e:
                raise ValueError(f"Ошибка парсинга BC '{s_clean}': {e}")

        if not parsed_bcs:
            raise ValueError("Нет валидных граничных условий")

        lambdified_bcs = lambdify(all_bc_syms, parsed_bcs, modules="numpy")

        def residual(x0: np.ndarray, xT: np.ndarray) -> np.ndarray:
            args = list(x0) + list(xT)
            return np.array(lambdified_bcs(*args), dtype=float)

        return residual

