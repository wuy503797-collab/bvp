#!/usr/bin/env python3
"""
通用BVP求解器 v2 -- 集成式实时交互绘图
基于参数延拓法, PyQt5 + matplotlib 嵌入

改进:
  - IntegratedPlotWidget: matplotlib嵌入PyQt5, 实时控件交互
  - 无需弹对话框, 所有绘图参数在图像窗口直接调整
  - 支持多Y叠加, S/F标记, 等比例, 网格等实时切换
"""

#!/usr/bin/env python3
"""
Универсальный решатель краевых задач (BVP) с методом продолжения по параметру.
通用常微分方程边值问题求解器（含参数延续法）

Архитектура:
  - Dataset      : dataclass для хранения параметров задачи
  - SymPyParser  : безопасный парсинг уравнений через SymPy
  - BVPSolver    : ядро — метод стрельбы и продолжения по параметру
  - SolverWorker : QThread для неблокирующих вычислений
  - PlotDialog   : универсальный диалог выбора осей графика
  - BvpSolverApp : PyQt5 GUI

См. кафедру ОУ ВМиК МГУ — Метод продолжения по параметру.
"""

import sys
import json
import re
import warnings
from dataclasses import dataclass, field, asdict
from typing import List, Callable, Optional, Tuple, Dict, Any

import numpy as np
from numpy.linalg import norm, solve as np_solve
from scipy.integrate import solve_ivp
from scipy.optimize import root

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations
from sympy.utilities.lambdify import lambdify

import matplotlib
matplotlib.use("Qt5Agg")
matplotlib.rcParams["toolbar"] = "toolmanager"
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLineEdit,
    QComboBox, QTableWidget, QTableWidgetItem, QGroupBox,
    QDoubleSpinBox, QSpinBox, QFrame, QAction, QShortcut,
    QHeaderView, QColorDialog, QFileDialog,
    QMessageBox, QInputDialog, QDialog, QCheckBox,
    QDialogButtonBox, QTextEdit, QSplitter, QTabWidget,
    QScrollArea, QFormLayout, QSizePolicy,
    QListWidget, QListWidgetItem,
    QRadioButton, QButtonGroup,
    QDockWidget, QActionGroup,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QKeySequence, QColor

# ---------------------------------------------------------------------------
# 1. Dataset — централизованное хранение параметров задачи
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    """
    Универсальный контейнер для параметров краевой задачи.
    """
    name: str = "Untitled"
    # ODE: список строк правых частей dx_i/dt = f_i(t, x0, x1, ...)
    equations: List[str] = field(default_factory=list)
    # Имена переменных состояния (автоматически: x0, x1, ... если не заданы)
    var_names: List[str] = field(default_factory=list)
    # Время интегрирования
    T: float = 1.0
    # Начальные условия: {index: value} — известные; None — неизвестные (искомые)
    initial_values: Dict[int, Optional[float]] = field(default_factory=dict)
    # Граничные условия: список строк вида "expr = 0" или "expr"
    # Переменные: x0_0=x0(0), x1_0=x1(0), ..., x0_T=x0(T), x1_T=x1(T)
    boundary_conditions: List[str] = field(default_factory=list)
    # Начальное приближение для НЕИЗВЕСТНЫХ начальных значений (по порядку индексов)
    guess: List[float] = field(default_factory=list)
    # Параметры интегрирования
    eps: float = 1e-8
    # Независимый критерий приёмки конечной граничной невязки.
    # boundary_rtol зарезервирован для будущего масштабированного критерия;
    # текущая проверка использует ||Phi||_2 <= boundary_atol.
    boundary_atol: float = 1e-8
    boundary_rtol: float = 0.0
    method: str = "RK45"
    # Выбор метода решения: "shooting" | "continuation"
    solver_method: str = "continuation"
    # Параметры продолжения по параметру
    continuation_steps: int = 50
    # Индексы переменных с фиксированными начальными значениями (known)
    known_indices: List[int] = field(default_factory=list)
    # Индексы переменных с неизвестными начальными значениями (unknown = to solve for)
    unknown_indices: List[int] = field(default_factory=list)
    # t* — точка, где задаётся начальное условие (0 по умолчанию)
    t_star: float = 0.0
    # Вспомогательные выходные переменные: {name: expr_str}
    # Примеры: {"u": "0.5*(sqrt(1e-10+(x5/2+1)**2)-sqrt(1e-10+(x5/2-1)**2))"}
    aux_outputs: Dict[str, str] = field(default_factory=dict)

    # ---- сериализация (JSON-совместимая) --------------------------------

    def to_dict(self) -> dict:
        """Полная сериализация в JSON-словарь."""
        return {
            "name": self.name,
            "equations": self.equations,
            "var_names": self.var_names,
            "T": self.T,
            "initial_values": {str(k): v for k, v in self.initial_values.items()},
            "boundary_conditions": self.boundary_conditions,
            "guess": self.guess,
            "eps": self.eps,
            "boundary_atol": self.boundary_atol,
            "boundary_rtol": self.boundary_rtol,
            "method": self.method,
            "solver_method": self.solver_method,
            "continuation_steps": self.continuation_steps,
            "known_indices": self.known_indices,
            "unknown_indices": self.unknown_indices,
            "t_star": self.t_star,
            "aux_outputs": dict(self.aux_outputs),
        }

    @staticmethod
    def from_dict(d: dict) -> "Dataset":
        """Десериализация из JSON-словаря."""
        # Конвертируем строковые ключи initial_values обратно в int
        iv = d.get("initial_values", {})
        if iv:
            initial_values = {}
            for k, v in iv.items():
                if v is not None:
                    initial_values[int(k)] = float(v)
                else:
                    initial_values[int(k)] = None
        else:
            initial_values = {}
        return Dataset(
            name=d.get("name", "Untitled"),
            equations=list(d.get("equations", [])),
            var_names=list(d.get("var_names", [])),
            T=float(d.get("T", 1.0)),
            initial_values=initial_values,
            boundary_conditions=list(d.get("boundary_conditions", [])),
            guess=list(d.get("guess", [])),
            eps=float(d.get("eps", 1e-8)),
            boundary_atol=float(d.get("boundary_atol", 1e-8)),
            boundary_rtol=float(d.get("boundary_rtol", 0.0)),
            method=d.get("method", "RK45"),
            solver_method=d.get("solver_method", "continuation"),
            continuation_steps=int(d.get("continuation_steps", 50)),
            known_indices=list(d.get("known_indices", [])),
            unknown_indices=list(d.get("unknown_indices", [])),
            t_star=float(d.get("t_star", 0.0)),
            aux_outputs=dict(d.get("aux_outputs", {})),
        )

    def dim(self) -> int:
        return len(self.equations)

    def num_unknown(self) -> int:
        return len(self.unknown_indices)

    def validate(self) -> List[str]:
        """Return core-model validation errors while preserving the legacy list API."""
        from bvp_core.adapters import config_from_dataset, problem_from_dataset
        from bvp_core.models import BVPValidationError

        errors: List[str] = []
        try:
            problem_from_dataset(self)
        except BVPValidationError as exc:
            errors.extend(exc.errors)
        try:
            config_from_dataset(self)
        except BVPValidationError as exc:
            errors.extend(exc.errors)
        return errors


# ---------------------------------------------------------------------------
# 2. SymPyParser — безопасный парсинг уравнений
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 3. BVPSolver — ядро: метод стрельбы + продолжения по параметру
# ---------------------------------------------------------------------------

class IVPIntegrationError(RuntimeError):
    """Failure of the inner initial-value problem with preserved SciPy diagnostics."""

    def __init__(
        self, message: str, solution: Any = None, p: Optional[np.ndarray] = None
    ):
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


def is_result_acceptable(result: dict) -> bool:
    """Return whether a solver result may enter GUI history, plots, and export."""
    required_flags = ("success", "ivp_success", "finite_success", "boundary_success")
    return all(result.get(flag) is True for flag in required_flags)


class BVPSolver:
    """
    Универсальный решатель краевых задач.
    Реализует:
      - Метод стрельбы (shooting) через scipy.optimize.root
      - Метод продолжения по параметру (continuation / homotopy)
        см. разд. 7.25–7.26 методички.
    """

    def __init__(self, dataset: Dataset, parser: SymPyParser):
        self.ds = dataset
        self.parser = parser
        # Граничные условия как функция
        self.bc_residual = SymPyParser.parse_boundary_conditions(
            dataset.boundary_conditions, dataset.var_names
        )
        self._build_initial_state_mapper()

    def _build_initial_state_mapper(self):
        """
        Строит маппер: вектор параметров p -> полный вектор начального состояния.
        known_indices  — фиксированные значения из initial_values
        unknown_indices — искомые (подставляются из p)
        """
        self.known = sorted(self.ds.known_indices)
        self.unknown = sorted(self.ds.unknown_indices)
        self.known_vals = {}
        for idx in self.known:
            self.known_vals[idx] = float(self.ds.initial_values[idx])

    def _p_to_state(self, p: np.ndarray) -> np.ndarray:
        """Собирает полный вектор x(t*) из параметров p (неизвестные)."""
        state = np.zeros(self.ds.dim())
        for idx in self.known:
            state[idx] = self.known_vals[idx]
        for i, idx in enumerate(self.unknown):
            state[idx] = p[i]
        return state

    def _state_to_p(self, state: np.ndarray) -> np.ndarray:
        """Извлекает параметры p из полного состояния."""
        return state[self.unknown]

    def _solve_ivp(self, p: np.ndarray, t_span: List[float],
                   dense_output: bool = False) -> Any:
        """Solve the inner IVP and preserve diagnostics for every failure."""
        x0 = self._p_to_state(p)
        try:
            sol = solve_ivp(
                self.parser.f, t_span, x0,
                method=self.ds.method,
                dense_output=dense_output,
                rtol=self.ds.eps, atol=self.ds.eps / 10,
            )
        except Exception as exc:
            raise IVPIntegrationError(
                f"solve_ivp raised {type(exc).__name__}: {exc}", p=p
            ) from exc

        t_values = np.asarray(sol.t, dtype=float)
        y_values = np.asarray(sol.y, dtype=float)
        state_finite = bool(
            np.isfinite(t_values).all() and np.isfinite(y_values).all()
        )
        t_final = float(t_values[-1]) if t_values.size else None
        terminal_reached = bool(
            t_values.size
            and np.isclose(
                t_final,
                float(t_span[-1]),
                rtol=0.0,
                atol=max(1e-12, abs(float(t_span[-1])) * 1e-12),
            )
        )
        if not sol.success or not state_finite or not terminal_reached:
            message = (
                "IVP integration failed: "
                f"success={bool(sol.success)}, status={getattr(sol, 'status', None)}, "
                f"t_final={t_final}, finite={state_finite}, "
                f"message={getattr(sol, 'message', '')}"
            )
            raise IVPIntegrationError(message, sol, p=p)
        return sol

    def _build_validated_result(
        self,
        *,
        p: np.ndarray,
        sol: Any,
        boundary_residual: Optional[np.ndarray],
        optimizer_success: bool,
        algorithm_success: bool,
        method: str,
        iterations: int,
        solver_metadata: Optional[dict] = None,
        failure_status: Optional[str] = None,
        failure_message: Optional[str] = None,
    ) -> dict:
        """Build one result shape and independently accept or reject the BVP."""
        p_opt = np.asarray(p, dtype=float)
        t_values = np.asarray(getattr(sol, "t", []), dtype=float)
        y_values = np.asarray(getattr(sol, "y", []), dtype=float)
        residual_available = boundary_residual is not None
        residual = (
            np.asarray(boundary_residual, dtype=float).reshape(-1)
            if residual_available
            else np.array([], dtype=float)
        )

        ivp_success = bool(getattr(sol, "success", False))
        finite_success = bool(
            residual_available
            and np.isfinite(p_opt).all()
            and np.isfinite(t_values).all()
            and np.isfinite(y_values).all()
            and np.isfinite(residual).all()
        )
        residual_norm = (
            float(norm(residual)) if residual_available and np.isfinite(residual).all()
            else float("inf")
        )
        boundary_success = bool(
            ivp_success
            and finite_success
            and residual.size == len(self.ds.boundary_conditions)
            and residual_norm <= self.ds.boundary_atol
        )
        success = bool(
            ivp_success
            and finite_success
            and boundary_success
            and algorithm_success
        )

        ivp_status = getattr(sol, "status", None)
        ivp_message = str(getattr(sol, "message", "IVP solution is unavailable"))
        ivp_t_final = float(t_values[-1]) if t_values.size else None

        if success:
            status = "success"
            message = (
                "Validated BVP solution: final boundary residual norm "
                f"{residual_norm:.6e} <= boundary_atol "
                f"{self.ds.boundary_atol:.6e}."
            )
        elif not ivp_success:
            status = "ivp_failed"
            message = failure_message or (
                "IVP integration failed: "
                f"status={ivp_status}, t_final={ivp_t_final}, message={ivp_message}"
            )
        elif not finite_success:
            status = "non_finite_result"
            message = failure_message or (
                "Candidate parameters, IVP state, or boundary residual contain "
                "non-finite or unavailable values."
            )
        elif not algorithm_success:
            status = failure_status or "optimizer_failed"
            message = failure_message or "The numerical algorithm did not converge."
        elif not boundary_success:
            status = "boundary_residual_too_large"
            message = failure_message or (
                "The optimizer terminated, but the final boundary residual did not "
                f"meet acceptance: {residual_norm:.6e} > boundary_atol "
                f"{self.ds.boundary_atol:.6e}."
            )
        else:
            status = failure_status or "optimizer_failed"
            message = failure_message or "The BVP result did not pass final acceptance."

        return {
            "success": success,
            "status": status,
            "message": message,
            "method": method,
            "p_opt": p_opt,
            "t": t_values,
            "y": y_values,
            "sol": sol,
            "ivp_success": ivp_success,
            "ivp_status": ivp_status,
            "ivp_message": ivp_message,
            "ivp_t_final": ivp_t_final,
            "optimizer_success": bool(optimizer_success),
            "algorithm_success": bool(algorithm_success),
            "finite_success": finite_success,
            "boundary_success": boundary_success,
            "boundary_residual": residual,
            "boundary_residual_norm": residual_norm,
            "boundary_atol": float(self.ds.boundary_atol),
            "boundary_rtol": float(self.ds.boundary_rtol),
            "boundary_acceptance": "l2_norm <= boundary_atol",
            "solver_metadata": dict(solver_metadata or {}),
            "iterations": int(iterations),
            # Backward-compatible alias used by the GUI and phase-one tests.
            "residual_norm": residual_norm,
        }

    def _build_ivp_failure_result(
        self,
        *,
        p: np.ndarray,
        method: str,
        error: IVPIntegrationError,
        optimizer_success: bool,
        iterations: int,
        solver_metadata: Optional[dict] = None,
    ) -> dict:
        candidate = error.p if error.p is not None else p
        result = self._build_validated_result(
            p=candidate,
            sol=error.solution,
            boundary_residual=None,
            optimizer_success=optimizer_success,
            algorithm_success=False,
            method=method,
            iterations=iterations,
            solver_metadata=solver_metadata,
            failure_status="ivp_failed",
            failure_message=str(error),
        )
        result["ivp_status"] = error.ivp_status
        result["ivp_message"] = error.ivp_message
        result["ivp_t_final"] = error.t_final
        result["ivp_state_finite"] = error.state_finite
        return result

    def _validate_final_candidate(
        self,
        *,
        p: np.ndarray,
        optimizer_success: bool,
        algorithm_success: bool,
        method: str,
        iterations: int,
        solver_metadata: Optional[dict] = None,
        failure_status: Optional[str] = None,
        failure_message: Optional[str] = None,
    ) -> dict:
        """Re-integrate a candidate, recompute Phi, and apply final acceptance."""
        try:
            sol = self._solve_ivp(
                p, [self.ds.t_star, self.ds.T], dense_output=True
            )
        except IVPIntegrationError as exc:
            return self._build_ivp_failure_result(
                p=p,
                method=method,
                error=exc,
                optimizer_success=optimizer_success,
                iterations=iterations,
                solver_metadata=solver_metadata,
            )

        try:
            residual = self.bc_residual(self._p_to_state(p), sol.y[:, -1])
        except Exception as exc:
            return self._build_validated_result(
                p=p,
                sol=sol,
                boundary_residual=None,
                optimizer_success=optimizer_success,
                algorithm_success=False,
                method=method,
                iterations=iterations,
                solver_metadata=solver_metadata,
                failure_status="boundary_evaluation_failed",
                failure_message=(
                    "Final boundary residual evaluation failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

        return self._build_validated_result(
            p=p,
            sol=sol,
            boundary_residual=residual,
            optimizer_success=optimizer_success,
            algorithm_success=algorithm_success,
            method=method,
            iterations=iterations,
            solver_metadata=solver_metadata,
            failure_status=failure_status,
            failure_message=failure_message,
        )

    def _Phi(self, p: np.ndarray) -> np.ndarray:
        """
        Вычисляет вектор невязки граничных условий Φ(p).
        Φ(p) = R(x(t*,p), x(T,p)).

        Интегрирование всегда на фиксированном отрезке [t_star, T].
        Для задач с нормировкой времени (напр. 26.4) переменная T
        входит в уравнения как параметр-множитель, а интервал
        интегрирования остаётся [0, 1] (ds.T = 1.0).
        """
        sol = self._solve_ivp(p, [self.ds.t_star, self.ds.T])
        x0_full = self._p_to_state(p)
        xT_full = sol.y[:, -1]
        return self.bc_residual(x0_full, xT_full)

    def _dPhi_dp(self, p: np.ndarray) -> np.ndarray:
        """
        Вычисляет матрицу Φ'(p).

        Стратегия:
          1. Пробуем совместное интегрирование + вариационное уравнение.
          2. При неудаче — чистое численное дифференцирование (надёжный fallback).
        """
        k = len(self.unknown)
        # Определяем размерность Φ(p) реальным вызовом
        Phi_base = self._Phi(p)
        m = len(Phi_base)
        n = self.ds.dim()
        T = self.ds.T

        # --- Попытка 1: совместное интегрирование состояния + вариаций ---
        try:
            x0 = self._p_to_state(p)
            X0 = np.eye(n)
            y0 = np.concatenate([x0, X0.flatten()])

            def combined_ode(t, y):
                x = y[:n]
                Xmat = y[n:].reshape(n, n)
                try:
                    J = self.parser.jac(t, x)
                except Exception:
                    # Fallback: численный якобиан если аналитический не доступен
                    J = self._numerical_jac_f(t, x)
                dxdt = self.parser.f(t, x)
                dXdt = J @ Xmat
                return np.concatenate([dxdt, dXdt.flatten()])

            sol = solve_ivp(
                combined_ode, [self.ds.t_star, T], y0,
                method="RK45",
                rtol=max(1e-6, self.ds.eps), atol=max(1e-8, self.ds.eps / 100),
                dense_output=False,
            )
            if sol.success:
                yT = sol.y[:, -1]
                XT = yT[n:].reshape(n, n)
                # dΦ/dp через вариации + численное дифференцирование BC
                eps_jac = max(1e-7, self.ds.eps ** 0.5)
                jac_matrix = np.zeros((m, k))
                for j in range(k):
                    p_perturb = p.copy()
                    p_perturb[j] += eps_jac
                    Phi_perturb = self._Phi(p_perturb)
                    jac_matrix[:, j] = (Phi_perturb - Phi_base) / eps_jac
                return jac_matrix
        except IVPIntegrationError:
            raise
        except Exception:
            return self._dPhi_dp_numerical(p, Phi_base)

        # --- Попытка 2 (fallback): чистое численное дифференцирование ---
        return self._dPhi_dp_numerical(p, Phi_base)

    def _dPhi_dp_numerical(self, p: np.ndarray,
                           Phi_base: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Надёжный fallback: dΦ/dp численным дифференцированием.
        Используется когда вариационное уравнение не сходится.
        """
        k = len(self.unknown)
        if Phi_base is None:
            Phi_base = self._Phi(p)
        m = len(Phi_base)

        # Адаптивный шаг
        eps_jac = max(1e-7, min(1e-4, self.ds.eps ** 0.5))
        jac_matrix = np.zeros((m, k))
        for j in range(k):
            p_perturb = p.copy()
            h = eps_jac * max(1.0, abs(p[j]))
            p_perturb[j] += h
            try:
                Phi_perturb = self._Phi(p_perturb)
                jac_matrix[:, j] = (Phi_perturb - Phi_base) / h
            except Exception:
                # Односторонняя разность
                p_perturb2 = p.copy()
                p_perturb2[j] -= h
                Phi_perturb2 = self._Phi(p_perturb2)
                jac_matrix[:, j] = (Phi_base - Phi_perturb2) / h
        return jac_matrix

    def _numerical_jac_f(self, t: float, x: np.ndarray, h: float = 1e-8) -> np.ndarray:
        """Численный якобиан ∂f/∂x (fallback если аналитический не доступен)."""
        n = len(x)
        J = np.zeros((n, n))
        f0 = self.parser.f(t, x)
        for j in range(n):
            x_perturb = x.copy()
            x_perturb[j] += h
            f_perturb = self.parser.f(t, x_perturb)
            J[:, j] = (f_perturb - f0) / h
        return J

    # ------------------------------------------------------------------
    # 3.1 Метод стрельбы (shooting)
    # ------------------------------------------------------------------

    def solve_shooting(self, callback: Optional[Callable] = None) -> dict:
        """
        Классический метод стрельбы: ищем p такое, что Φ(p) = 0.
        
        Стратегия:
          1. Пробуем scipy.optimize.root (hybr) — быстрый, хороший при хорошем guess.
          2. При неудаче — scipy.optimize.least_squares (LM) — более робастный.
        """
        from scipy.optimize import least_squares

        p0 = np.array(self.ds.guess, dtype=float)

        def residual(p):
            return self._Phi(p)

        if callback:
            callback("shooting", 0, "Начало метода стрельбы (hybr)...")

        solver_metadata: Dict[str, Any] = {
            "optimizer": "root/hybr",
            "fallback_used": False,
        }
        candidate = p0.copy()
        optimizer_success = False
        iterations = 0
        optimizer_message = "root/hybr did not run"

        # Попытка 1: hybr (быстрый). Его success сохраняется как метаданные,
        # но окончательное решение всё равно независимо перевычисляется ниже.
        try:
            root_result = root(
                residual,
                p0,
                method="hybr",
                tol=self.ds.eps,
                options={"maxfev": 100 * len(p0)},
            )
            candidate = np.asarray(root_result.x, dtype=float)
            optimizer_success = bool(root_result.success)
            iterations = int(getattr(root_result, "nfev", 0))
            optimizer_message = str(getattr(root_result, "message", ""))
            solver_metadata["root"] = {
                "success": optimizer_success,
                "status": getattr(root_result, "status", None),
                "message": optimizer_message,
                "nfev": getattr(root_result, "nfev", None),
            }
        except IVPIntegrationError as exc:
            solver_metadata["root"] = {
                "success": False,
                "message": str(exc),
            }
            return self._build_ivp_failure_result(
                p=candidate,
                method="shooting",
                error=exc,
                optimizer_success=False,
                iterations=iterations,
                solver_metadata=solver_metadata,
            )
        except Exception as exc:
            optimizer_message = f"root/hybr raised {type(exc).__name__}: {exc}"
            solver_metadata["root"] = {
                "success": False,
                "message": optimizer_message,
            }

        # Попытка 2: least_squares. Нормальное завершение оптимизатора не
        # является критерием выполнения граничных условий.
        if not optimizer_success:
            solver_metadata["fallback_used"] = True
            solver_metadata["optimizer"] = "least_squares"
            if callback:
                callback("shooting", 0, "hybr не сошёлся, пробуем least_squares...")
            try:
                ls_result = least_squares(
                    residual,
                    p0,
                    ftol=self.ds.eps,
                    xtol=self.ds.eps,
                    gtol=self.ds.eps,
                    max_nfev=5000 * len(p0),
                )
                candidate = np.asarray(ls_result.x, dtype=float)
                optimizer_success = bool(ls_result.success)
                iterations = int(getattr(ls_result, "nfev", 0))
                optimizer_message = str(getattr(ls_result, "message", ""))
                solver_metadata["least_squares"] = {
                    "success": optimizer_success,
                    "status": getattr(ls_result, "status", None),
                    "message": optimizer_message,
                    "cost": getattr(ls_result, "cost", None),
                    "optimality": getattr(ls_result, "optimality", None),
                    "nfev": getattr(ls_result, "nfev", None),
                }
            except IVPIntegrationError as exc:
                solver_metadata["least_squares"] = {
                    "success": False,
                    "message": str(exc),
                }
                return self._build_ivp_failure_result(
                    p=candidate,
                    method="shooting",
                    error=exc,
                    optimizer_success=False,
                    iterations=iterations,
                    solver_metadata=solver_metadata,
                )
            except Exception as exc:
                optimizer_message = (
                    f"least_squares raised {type(exc).__name__}: {exc}"
                )
                solver_metadata["least_squares"] = {
                    "success": False,
                    "message": optimizer_message,
                }

        result = self._validate_final_candidate(
            p=candidate,
            optimizer_success=optimizer_success,
            algorithm_success=optimizer_success,
            method="shooting",
            iterations=iterations,
            solver_metadata=solver_metadata,
            failure_status="optimizer_failed" if not optimizer_success else None,
            failure_message=(
                f"Shooting optimizer did not converge: {optimizer_message}"
                if not optimizer_success
                else None
            ),
        )

        if callback:
            if result["success"]:
                callback("shooting", 100, "Готово: решение прошло проверку границы.")
            else:
                callback("shooting", 100, f"Отклонено: {result['message']}")
        return result

    # ------------------------------------------------------------------
    # 3.2 Метод продолжения по параметру
    # ------------------------------------------------------------------

    def solve_continuation(self, callback: Optional[Callable] = None) -> dict:
        """
        Метод продолжения по параметру (параметрическая гомотопия).

        Алгоритм (разд. 7.25–7.26):
          Вспомогательное уравнение:  Φ(p) = (1-μ)·Φ(p₀)  , μ∈[0,1]
          При μ=0: решение p(0)=p₀  (известно)
          При μ=1: Φ(p)=0  — исходная задача

        На каждом шаге μ решаем промежуточную задачу Φ(p) = (1-μ)·Φ(p₀)
        методом Ньютона, стартуя с решения предыдущего шага. Это позволяет
        устойчиво следовать вдоль выбранной ветви решения.
        """
        p0 = np.array(self.ds.guess, dtype=float)
        N = self.ds.continuation_steps
        p = p0.copy()
        total_newton = 0
        solver_metadata: Dict[str, Any] = {
            "continuation_steps": N,
            "max_newton_iterations": 20,
            "steps": [],
            "failure_step": None,
            "failure_reason": None,
        }

        # Φ(p₀) — начальная невязка
        try:
            Phi_p0 = self._Phi(p0)
        except IVPIntegrationError as exc:
            return self._build_ivp_failure_result(
                p=p0,
                method="continuation",
                error=exc,
                optimizer_success=False,
                iterations=0,
                solver_metadata=solver_metadata,
            )

        if callback:
            callback(
                "continuation",
                0,
                f"Начало продолжения: ‖Φ(p₀)‖={norm(Phi_p0):.4e}",
            )

        if not np.isfinite(Phi_p0).all():
            return self._validate_final_candidate(
                p=p0,
                optimizer_success=False,
                algorithm_success=False,
                method="continuation",
                iterations=0,
                solver_metadata=solver_metadata,
                failure_status="non_finite_result",
                failure_message="Initial continuation residual contains NaN or Inf.",
            )

        # Проверяем, не является ли p₀ уже допустимым решением.
        if norm(Phi_p0) <= self.ds.boundary_atol:
            return self._validate_final_candidate(
                p=p0,
                optimizer_success=True,
                algorithm_success=True,
                method="continuation",
                iterations=0,
                solver_metadata=solver_metadata,
            )

        failure_status: Optional[str] = None
        failure_message: Optional[str] = None

        for step in range(1, N + 1):
            mu = step / N
            target = (1.0 - mu) * Phi_p0
            step_metadata: Dict[str, Any] = {
                "step": step,
                "mu": mu,
                "newton_converged": False,
                "jacobian_success": None,
                "damping_success": None,
                "newton_updates": 0,
                "residual_norm": None,
                "failure_reason": None,
            }

            for _newton_iter in range(20):
                try:
                    Phi_current = self._Phi(p)
                except IVPIntegrationError as exc:
                    step_metadata["failure_reason"] = str(exc)
                    solver_metadata["steps"].append(step_metadata)
                    solver_metadata["failure_step"] = step
                    solver_metadata["failure_reason"] = str(exc)
                    return self._build_ivp_failure_result(
                        p=p,
                        method="continuation",
                        error=exc,
                        optimizer_success=False,
                        iterations=total_newton,
                        solver_metadata=solver_metadata,
                    )

                residual = Phi_current - target
                res_norm = float(norm(residual))
                step_metadata["residual_norm"] = res_norm
                if not np.isfinite(p).all() or not np.isfinite(residual).all():
                    failure_status = "non_finite_result"
                    failure_message = (
                        f"Continuation step {step} at mu={mu:.6g} produced NaN or Inf."
                    )
                    step_metadata["failure_reason"] = failure_message
                    break
                if res_norm <= self.ds.eps:
                    step_metadata["newton_converged"] = True
                    break

                try:
                    dPhi = self._dPhi_dp(p)
                    step_metadata["jacobian_success"] = bool(
                        np.isfinite(dPhi).all()
                        and dPhi.shape == (len(residual), len(p))
                    )
                    if not step_metadata["jacobian_success"]:
                        raise ValueError(
                            f"invalid Jacobian shape or values: {dPhi.shape}"
                        )
                    try:
                        delta = np_solve(dPhi, residual)
                    except np.linalg.LinAlgError:
                        delta = np.linalg.lstsq(dPhi, residual, rcond=None)[0]
                    if not np.isfinite(delta).all():
                        raise ValueError("Newton update contains NaN or Inf")
                except IVPIntegrationError as exc:
                    step_metadata["jacobian_success"] = False
                    step_metadata["failure_reason"] = str(exc)
                    solver_metadata["steps"].append(step_metadata)
                    solver_metadata["failure_step"] = step
                    solver_metadata["failure_reason"] = str(exc)
                    return self._build_ivp_failure_result(
                        p=p,
                        method="continuation",
                        error=exc,
                        optimizer_success=False,
                        iterations=total_newton,
                        solver_metadata=solver_metadata,
                    )
                except Exception as exc:
                    step_metadata["jacobian_success"] = False
                    failure_status = "continuation_failed"
                    failure_message = (
                        f"Continuation Jacobian failed at step {step}, mu={mu:.6g}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    step_metadata["failure_reason"] = failure_message
                    break

                accepted_update = False
                for factor in (1.0, 0.5, 0.25, 0.125, 0.0625):
                    p_try = p - factor * delta
                    if not np.isfinite(p_try).all():
                        continue
                    try:
                        Phi_try = self._Phi(p_try)
                    except IVPIntegrationError as exc:
                        step_metadata["damping_success"] = False
                        step_metadata["failure_reason"] = str(exc)
                        solver_metadata["steps"].append(step_metadata)
                        solver_metadata["failure_step"] = step
                        solver_metadata["failure_reason"] = str(exc)
                        return self._build_ivp_failure_result(
                            p=p_try,
                            method="continuation",
                            error=exc,
                            optimizer_success=False,
                            iterations=total_newton,
                            solver_metadata=solver_metadata,
                        )
                    candidate_residual = Phi_try - target
                    if (
                        np.isfinite(candidate_residual).all()
                        and norm(candidate_residual) < res_norm
                    ):
                        p = p_try
                        accepted_update = True
                        step_metadata["damping_success"] = True
                        step_metadata["newton_updates"] += 1
                        total_newton += 1
                        break

                if not accepted_update:
                    step_metadata["damping_success"] = False
                    failure_status = "continuation_failed"
                    failure_message = (
                        f"Continuation damping failed at step {step}, mu={mu:.6g}; "
                        "no tested step reduced the homotopy residual."
                    )
                    step_metadata["failure_reason"] = failure_message
                    break

            if not step_metadata["newton_converged"] and failure_message is None:
                try:
                    final_step_residual = self._Phi(p) - target
                    final_step_norm = float(norm(final_step_residual))
                    step_metadata["residual_norm"] = final_step_norm
                    step_metadata["newton_converged"] = bool(
                        np.isfinite(final_step_residual).all()
                        and final_step_norm <= self.ds.eps
                    )
                except IVPIntegrationError as exc:
                    step_metadata["failure_reason"] = str(exc)
                    solver_metadata["steps"].append(step_metadata)
                    solver_metadata["failure_step"] = step
                    solver_metadata["failure_reason"] = str(exc)
                    return self._build_ivp_failure_result(
                        p=p,
                        method="continuation",
                        error=exc,
                        optimizer_success=False,
                        iterations=total_newton,
                        solver_metadata=solver_metadata,
                    )

            if not step_metadata["newton_converged"]:
                if failure_message is None:
                    failure_status = "continuation_failed"
                    failure_message = (
                        f"Continuation Newton did not converge within 20 iterations "
                        f"at step {step}, mu={mu:.6g}; residual_norm="
                        f"{step_metadata['residual_norm']}."
                    )
                    step_metadata["failure_reason"] = failure_message
                solver_metadata["steps"].append(step_metadata)
                solver_metadata["failure_step"] = step
                solver_metadata["failure_reason"] = failure_message
                break

            solver_metadata["steps"].append(step_metadata)
            if callback and step % max(1, N // 10) == 0:
                progress = int(100 * step / N)
                callback(
                    "continuation",
                    progress,
                    f"Шаг {step}/{N}: ‖Φ-target‖="
                    f"{step_metadata['residual_norm']:.4e}",
                )

        algorithm_success = failure_message is None
        result = self._validate_final_candidate(
            p=p,
            optimizer_success=algorithm_success,
            algorithm_success=algorithm_success,
            method="continuation",
            iterations=total_newton,
            solver_metadata=solver_metadata,
            failure_status=failure_status,
            failure_message=failure_message,
        )

        if callback:
            if result["success"]:
                callback(
                    "continuation",
                    100,
                    f"Готово! Newton: {total_newton}, "
                    f"‖Φ‖={result['boundary_residual_norm']:.4e}",
                )
            else:
                callback("continuation", 100, f"Отклонено: {result['message']}")
        return result

    def solve(self, callback: Optional[Callable] = None) -> dict:
        """Диспетчер: выбирает метод в соответствии с Dataset."""
        errors = self.ds.validate()
        if errors:
            raise ValueError("; ".join(errors))

        if self.ds.solver_method == "shooting":
            return self.solve_shooting(callback)
        else:
            return self.solve_continuation(callback)


# ---------------------------------------------------------------------------
# 4. SolverWorker — вычисления в отдельном потоке
# ---------------------------------------------------------------------------

class SolverWorker(QThread):
    """
    Поток для выполнения BVP-решения.
    Сигналы:
      progress(str, int, str) — метод, прогресс%, сообщение
      finished(dict)          — результат
      error(str)              — ошибка
    """
    progress = pyqtSignal(str, int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, dataset: Dataset):
        super().__init__()
        self.dataset = dataset

    def run(self):
        try:
            parser = SymPyParser(self.dataset.equations, self.dataset.var_names)
            parser.lambdify_all()
            solver = BVPSolver(self.dataset, parser)
            result = solver.solve(callback=self._on_progress)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    def _on_progress(self, method: str, percent: int, message: str):
        self.progress.emit(method, percent, message)


# ---------------------------------------------------------------------------
# 5. IntegratedPlotWidget -- 集成式实时交互绘图
# ---------------------------------------------------------------------------

class IntegratedPlotWidget(QWidget):
    """
    集成式绘图窗口: matplotlib 嵌入 PyQt5 + 实时控制面板。
    所有控件变化立即重绘图像, 无需弹对话框。
    """

    TRANSLATIONS = {
        "zh": {
            "plot_controls": "绘图控制",
            "x_axis": "X 轴:",
            "y_axes": "Y 轴:",
            "line_width": "线宽:",
            "grid": "网格",
            "sf_markers": "S/F 标记",
            "equal_aspect": "等比例",
            "solutions": "解",
            "ls_all_solid": "全部实线",
            "ls_aux_solid": "辅实状虚",
            "ls_aux_dash": "辅虚实实",
            "ls_tooltip": "循环切换线型模式",
        },
        "ru": {
            "plot_controls": "Управление графиком",
            "x_axis": "Ось X:",
            "y_axes": "Оси Y:",
            "line_width": "Толщина:",
            "grid": "Сетка",
            "sf_markers": "Маркеры S/F",
            "equal_aspect": "Равные пропорции",
            "solutions": "Решения",
            "ls_all_solid": "Все сплошные",
            "ls_aux_solid": "Вспом. сплошн.",
            "ls_aux_dash": "Вспом. пунктир",
            "ls_tooltip": "Переключить стиль линии",
        },
    }

    def __init__(self, var_names: List[str], all_results: List[dict],
                 aux_names: List[str] = None, lang: str = "zh", parent=None):
        super().__init__(parent)
        self.var_names = var_names
        self.all_results = all_results
        self.aux_names = aux_names or []
        self.lang = lang
        self._visible = set(range(len(all_results)))
        self._ls_mode = 0
        self.setWindowFlags(Qt.Window)
        self.resize(950, 680)
        self._build_ui()
        self._refresh_plot()

    def update_data(self, var_names: List[str], all_results: List[dict],
                    aux_names: List[str] = None, lang: str = None):
        """更新数据(当新求解完成时调用)."""
        self.var_names = var_names
        self.all_results = all_results
        self.aux_names = aux_names or []
        if lang is not None:
            self.lang = lang
        self._visible = set(range(len(all_results)))
        self._rebuild_controls()
        self._refresh_plot()

    def set_lang(self, lang: str):
        """切换界面语言."""
        self.lang = lang
        self._update_ui_texts()
        self._refresh_plot()

    def _t(self, key: str) -> str:
        """获取当前语言的翻译."""
        return self.TRANSLATIONS.get(self.lang, self.TRANSLATIONS["zh"]).get(key, key)

    def _build_ui(self):
        """构建 UI: 左侧控制面板 + 右侧图像区."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # ===== 左侧控制面板 =====
        self.panel = QGroupBox()
        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setSpacing(6)

        self.form_layout = QFormLayout()

        # --- X 轴 ---
        self.combo_x = QComboBox()
        for name in self.var_names:
            self.combo_x.addItem(name, name)
        self.combo_x.addItem("t", "t")
        if len(self.var_names) > 2:
            self.combo_x.setCurrentIndex(self.combo_x.count() - 1)
        self.combo_x.currentIndexChanged.connect(self._refresh_plot)
        self._label_x = QLabel()
        self.form_layout.addRow(self._label_x, self.combo_x)

        # --- Y 轴 (多选) ---
        self.y_list = QListWidget()
        self.y_list.setSelectionMode(QListWidget.MultiSelection)
        self._populate_y_list()
        self.y_list.itemSelectionChanged.connect(self._refresh_plot)
        self.y_list.setMaximumHeight(150)
        self._label_y = QLabel()
        self.form_layout.addRow(self._label_y, self.y_list)

        # --- 线宽 ---
        self.spin_lw = QDoubleSpinBox()
        self.spin_lw.setRange(0.5, 5.0)
        self.spin_lw.setDecimals(1)
        self.spin_lw.setValue(1.5)
        self.spin_lw.setSingleStep(0.5)
        self.spin_lw.valueChanged.connect(self._refresh_plot)
        self._label_lw = QLabel()
        self.form_layout.addRow(self._label_lw, self.spin_lw)

        panel_layout.addLayout(self.form_layout)

        # --- 选项 ---
        self.check_grid = QCheckBox()
        self.check_grid.setChecked(True)
        self.check_grid.stateChanged.connect(self._refresh_plot)
        panel_layout.addWidget(self.check_grid)

        self.check_markers = QCheckBox()
        self.check_markers.setChecked(True)
        self.check_markers.stateChanged.connect(self._refresh_plot)
        panel_layout.addWidget(self.check_markers)

        self.check_equal = QCheckBox()
        self.check_equal.setChecked(False)
        self.check_equal.stateChanged.connect(self._refresh_plot)
        panel_layout.addWidget(self.check_equal)

        # 线型模式切换按钮
        self._ls_mode = 0
        self._ls_label_keys = ["ls_all_solid", "ls_aux_solid", "ls_aux_dash"]
        self.btn_linestyle = QPushButton()
        self.btn_linestyle.clicked.connect(self._toggle_linestyle)
        panel_layout.addWidget(self.btn_linestyle)

        panel_layout.addStretch()

        # Solutions 选择
        self.sol_group = QGroupBox()
        sol_layout = QVBoxLayout(self.sol_group)
        self.sol_list = QListWidget()
        self._colors_rgb = [(31,119,180),(255,127,14),(44,160,44),
                            (214,39,40),(148,103,189),(140,86,75),
                            (227,119,194),(127,127,127)]
        for i, r in enumerate(self.all_results):
            name = f"#{i+1} ({r['method']}, ‖Φ‖={r['residual_norm']:.2e})"
            item = QListWidgetItem(name)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, i)
            c = self._colors_rgb[i % len(self._colors_rgb)]
            item.setBackground(QColor(c[0], c[1], c[2], 40))
            self.sol_list.addItem(item)
        self.sol_list.itemChanged.connect(self._on_sol_toggled)
        sol_layout.addWidget(self.sol_list)
        panel_layout.addWidget(self.sol_group)

        self._update_ui_texts()
        main_layout.addWidget(self.panel, 0)

        # ===== 右侧图像区 =====
        right = QVBoxLayout()
        self.fig = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right.addWidget(self.toolbar)
        main_layout.addLayout(right, 1)

    def _populate_y_list(self):
        """填充 Y 轴列表: 状态变量 + [辅助变量]."""
        self.y_list.clear()
        for name in self.var_names:
            self.y_list.addItem(name)
        # 辅助变量（如果有）— 用不同前缀标识
        if self.aux_names:
            for name in self.aux_names:
                item = QListWidgetItem(f"[{name}]")
                item.setData(Qt.UserRole, name)  # 存储原始名
                item.setForeground(QColor(148, 103, 189))
                self.y_list.addItem(item)
        self.y_list.addItem("t")
        # 默认选中前 min(3, n_vars) 个状态变量
        n_vars = len(self.var_names)
        for r in range(min(3, n_vars)):
            self.y_list.item(r).setSelected(True)

    def _rebuild_controls(self):
        """数据更新后重建控件."""
        self._populate_y_list()

        cur_x = self.combo_x.currentData()
        self.combo_x.clear()
        for name in self.var_names:
            self.combo_x.addItem(name, name)
        self.combo_x.addItem("t (time)", "t")
        if len(self.var_names) > 2:
            self.combo_x.setCurrentIndex(self.combo_x.count() - 1)

        self.sol_list.clear()
        for i, r in enumerate(self.all_results):
            name = f"#{i+1} ({r['method']}, ‖Φ‖={r['residual_norm']:.2e})"
            item = QListWidgetItem(name)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, i)
            c = self._colors_rgb[i % len(self._colors_rgb)]
            item.setBackground(QColor(c[0], c[1], c[2], 40))
            self.sol_list.addItem(item)
        self._visible = set(range(len(self.all_results)))

    def _on_sol_toggled(self, item):
        idx = item.data(Qt.UserRole)
        if item.checkState() == Qt.Checked:
            self._visible.add(idx)
        else:
            self._visible.discard(idx)
        self._refresh_plot()

    def _get_y_axes(self) -> List[str]:
        y_axes = []
        for item in self.y_list.selectedItems():
            txt = item.text()
            if txt == "t":
                y_axes.append("t")
            elif txt.startswith("[") and txt.endswith("]"):
                # 辅助变量: [name] -> name
                y_axes.append(item.data(Qt.UserRole) or txt[1:-1])
            else:
                y_axes.append(txt)
        if not y_axes:
            y_axes = [self.var_names[0]] if self.var_names else ["t"]
        return y_axes

    def _refresh_plot(self):
        """核心: 根据当前控件状态重绘图像."""
        if not self.all_results:
            return

        x_axis = self.combo_x.currentData() or "t"
        y_axes = self._get_y_axes()
        if x_axis != "t":
            y_axes = [y_axes[0]]

        lw = self.spin_lw.value()
        show_grid = self.check_grid.isChecked()
        show_markers = self.check_markers.isChecked()
        equal = self.check_equal.isChecked()

        colors = [(31/255,119/255,180/255),(255/255,127/255,14/255),
                  (44/255,160/255,44/255),(214/255,39/255,40/255),
                  (148/255,103/255,189/255),(140/255,86/255,75/255),
                  (227/255,119/255,194/255),(127/255,127/255,127/255)]
        linestyles = ["-", "--", ":", "-."]
        is_phase = (x_axis != "t" and y_axes[0] != "t")

        def axis_data(name, t_vals, y_vals):
            if name == "t":
                return t_vals, "t"
            return y_vals[self.var_names.index(name)], name

        # 收集数据（状态变量 + 辅助变量）
        aux_set = set(self.aux_names)
        curves = []
        for i, r in enumerate(self.all_results):
            if i not in self._visible:
                continue
            try:
                sol = r["sol"]
                t0, t1 = float(sol.t[0]), float(sol.t[-1])
                td = np.linspace(t0, t1, 500)
                if hasattr(sol, 'sol') and callable(sol.sol):
                    yd = sol.sol(td)
                else:
                    yd = np.array([np.interp(td, sol.t, sol.y[j])
                                   for j in range(len(self.var_names))])
                xd, xl = axis_data(x_axis, td, yd)
                # 辅助变量数据
                aux_data = r.get("aux", {})
                for y_name in y_axes:
                    try:
                        if y_name in aux_set:
                            # 辅助变量: 从预计算数据中取
                            aux_vals = aux_data.get(y_name)
                            if aux_vals is not None:
                                curves.append((i, y_name, np.asarray(xd), np.asarray(aux_vals)))
                        else:
                            yd_arr, yl = axis_data(y_name, td, yd)
                            curves.append((i, y_name, np.asarray(xd), np.asarray(yd_arr)))
                    except Exception:
                        continue
            except Exception:
                continue

        if not curves:
            return

        # 重绘
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        n_y = len(y_axes)
        n_sol = len(set(c[0] for c in curves))

        # 线型模式: 0=全部实线, 1=辅实状虚(教材26.4), 2=全部虚线
        ls_mode = getattr(self, '_ls_mode', 0)
        for sol_idx, y_name, xd, yd in curves:
            y_idx = y_axes.index(y_name) if y_name in y_axes else 0
            c = colors[y_idx % len(colors)]
            is_aux = y_name in aux_set
            if n_sol > 1:
                base_ls = linestyles[sol_idx % len(linestyles)]
                if ls_mode == 0:
                    ls = base_ls  # 全部实线(按解区分)
                elif ls_mode == 1:
                    ls = "-" if is_aux else "--"  # 辅实状虚
                else:
                    ls = "--" if is_aux else "-"  # 辅虚实实
            else:
                if ls_mode == 0:
                    ls = "-"  # 全部实线
                elif ls_mode == 1:
                    ls = "-" if is_aux else "--"  # 辅实状虚
                else:
                    ls = "--" if is_aux else "-"  # 辅虚实实
            label = f"{y_name}" if n_sol == 1 else f"{y_name} #{sol_idx+1}"
            ax.plot(xd, yd, ls=ls, color=c, lw=lw, label=label)

        # 标签
        _, x_lbl = axis_data(x_axis, None, np.zeros(len(self.var_names)))
        ax.set_xlabel(f"$\\mathit{{{x_lbl}}}$", fontsize=13)
        if is_phase:
            y_lbl = y_axes[0]
            ax.set_ylabel(f"$\\mathit{{{y_lbl}}}$", fontsize=13)
            ax.set_title(f"${y_lbl}$ vs ${x_lbl}$", fontsize=14)
        elif x_axis == "t":
            y_label_str = ", ".join(y_axes)
            ax.set_ylabel(f"$\\mathit{{{y_label_str}}}$", fontsize=12)
            ax.set_title("$x_i(t)$" if len(y_axes) > 1 else f"${y_axes[0]}(t)$", fontsize=14)
        else:
            ax.set_ylabel(f"$\\mathit{{{y_axes[0]}}}$", fontsize=13)

        # S/F 标记
        if show_markers and is_phase:
            seen = set()
            sf_off = [(10, 8), (-14, -12), (10, -14), (-14, 8)]
            ff_off = [(10, -12), (-14, 8), (10, 10), (-14, -14)]
            for sol_idx, y_name, xd, yd in curves:
                if y_name != y_axes[0] or sol_idx in seen:
                    continue
                seen.add(sol_idx)
                y_idx = y_axes.index(y_name) if y_name in y_axes else 0
                c = colors[y_idx % len(colors)]
                ax.plot(xd[0], yd[0], "o", color=c, ms=7, mec="black", mew=0.5)
                ax.plot(xd[-1], yd[-1], "s", color=c, ms=7, mec="black", mew=0.5)
                idx = len(seen) - 1
                ax.annotate("S", (xd[0], yd[0]), textcoords="offset points",
                           xytext=sf_off[idx % 4], fontsize=13, fontweight="bold", color=c)
                ax.annotate("F", (xd[-1], yd[-1]), textcoords="offset points",
                           xytext=ff_off[idx % 4], fontsize=13, fontweight="bold", color=c)

        if show_grid:
            ax.grid(True, alpha=0.3, ls="--")
        ax.legend(loc="best", fontsize=10, framealpha=0.9)
        ax.tick_params(labelsize=11)
        # 零轴 x=0, y=0: 加粗实线, 像传统坐标系
        ax.axhline(y=0, color="black", linewidth=1.5, zorder=0)
        ax.axvline(x=0, color="black", linewidth=1.5, zorder=0)
        if is_phase and equal:
            ax.set_aspect("equal")
        self.fig.tight_layout()
        self.canvas.draw()

    def _update_ui_texts(self):
        """根据当前语言更新所有控件文本."""
        self.panel.setTitle(self._t("plot_controls"))
        self._label_x.setText(self._t("x_axis"))
        self._label_y.setText(self._t("y_axes"))
        self._label_lw.setText(self._t("line_width"))
        self.check_grid.setText(self._t("grid"))
        self.check_markers.setText(self._t("sf_markers"))
        self.check_equal.setText(self._t("equal_aspect"))
        self.sol_group.setTitle(self._t("solutions"))
        self.btn_linestyle.setText(self._t(self._ls_label_keys[self._ls_mode]))
        self.btn_linestyle.setToolTip(self._t("ls_tooltip"))
        # 更新窗口标题
        t = BvpSolverApp.TRANSLATIONS.get(self.lang, BvpSolverApp.TRANSLATIONS["zh"])
        self.setWindowTitle(t.get("btn_plot", "绘图"))

    def _toggle_linestyle(self):
        """循环切换线型模式: 0→1→2→0."""
        self._ls_mode = (self._ls_mode + 1) % 3
        self.btn_linestyle.setText(self._t(self._ls_label_keys[self._ls_mode]))
        self._refresh_plot()

    def closeEvent(self, event):
        event.accept()
        if self.parent() and hasattr(self.parent(), '_plot_widget'):
            self.parent()._plot_widget = None

class EquationEditor(QWidget):
    """
    Вкладка для ввода уравнений ODE, граничных условий и начальных значений.
    """

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # --- Размерность ---
        h = QHBoxLayout()
        self.label_dim = QLabel("Размерность системы / System dimension:")
        h.addWidget(self.label_dim)
        self.spin_dim = QSpinBox()
        self.spin_dim.setRange(1, 20)
        self.spin_dim.setValue(2)
        self.spin_dim.valueChanged.connect(self._on_dim_changed)
        h.addWidget(self.spin_dim)
        h.addStretch()
        layout.addLayout(h)

        # --- Уравнения ---
        self.group_eq = QGroupBox()
        self.eq_layout = QVBoxLayout(self.group_eq)
        self.eq_editors: List[QLineEdit] = []
        layout.addWidget(self.group_eq)

        # --- Граничные условия ---
        self.group_bc = QGroupBox()
        bc_layout = QVBoxLayout(self.group_bc)
        self.bc_editor = QTextEdit()
        self.bc_editor.setPlaceholderText("")
        self.bc_editor.setMaximumHeight(100)
        bc_layout.addWidget(self.bc_editor)
        layout.addWidget(self.group_bc)

        # --- Начальные условия ---
        self.group_iv = QGroupBox()
        self.iv_layout = QGridLayout(self.group_iv)
        self.iv_editors: List[QLineEdit] = []
        self.iv_checkboxes: List[QCheckBox] = []
        layout.addWidget(self.group_iv)

        # --- Имена переменных ---
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Имена переменных (через запятую):"))
        self.input_varnames = QLineEdit("x, y, vx, vy")
        h2.addWidget(self.input_varnames)
        layout.addLayout(h2)

        layout.addStretch()
        self._on_dim_changed(4)

    def _on_dim_changed(self, n: int):
        """Перестраивает редакторы уравнений и начальных значений."""
        # Уравнения: скрываем и удаляем старые редакторы
        for old in self.eq_editors:
            old.hide()
            old.deleteLater()
        self.eq_editors.clear()
        while self.eq_layout.count():
            item = self.eq_layout.takeAt(0)
            if item.widget():
                w = item.widget()
                w.hide()
                w.deleteLater()
            elif item.layout():
                # Рекурсивно очищаем вложенные layout
                sub_layout = item.layout()
                while sub_layout.count():
                    sub = sub_layout.takeAt(0)
                    if sub.widget():
                        sub.widget().hide()
                        sub.widget().deleteLater()

        defaults = self._default_equations(n)
        for i in range(n):
            h = QHBoxLayout()
            h.addWidget(QLabel(f"d/dt [{i}]:"))
            ed = QLineEdit(defaults[i] if i < len(defaults) else "0")
            ed.textChanged.connect(self.changed.emit)
            self.eq_editors.append(ed)
            h.addWidget(ed)
            self.eq_layout.addLayout(h)

        # Начальные значения: скрываем и удаляем старые контролы
        for old in self.iv_editors:
            old.hide()
            old.deleteLater()
        self.iv_editors.clear()
        for old in self.iv_checkboxes:
            old.hide()
            old.deleteLater()
        self.iv_checkboxes.clear()
        while self.iv_layout.count():
            item = self.iv_layout.takeAt(0)
            if item.widget():
                w = item.widget()
                w.hide()
                w.deleteLater()
            elif item.layout():
                sub_layout = item.layout()
                while sub_layout.count():
                    sub = sub_layout.takeAt(0)
                    if sub.widget():
                        sub.widget().hide()
                        sub.widget().deleteLater()

        iv_defaults = self._default_initial_values(n)
        for i in range(n):
            chk = QCheckBox(f"x{i}(0)")
            ed = QLineEdit()
            if i < len(iv_defaults) and iv_defaults[i] is not None:
                ed.setText(str(iv_defaults[i]))
                chk.setChecked(True)
            else:
                chk.setChecked(False)
            chk.stateChanged.connect(self.changed.emit)
            ed.textChanged.connect(self.changed.emit)
            self.iv_editors.append(ed)
            self.iv_checkboxes.append(chk)
            self.iv_layout.addWidget(chk, i, 0)
            self.iv_layout.addWidget(ed, i, 1)

    def _default_equations(self, n: int) -> List[str]:
        """默认方程全部填0, 由用户手动输入."""
        return ["0"] * n

    def _default_initial_values(self, n: int) -> List[Optional[float]]:
        return [None] * n

    def get_equations(self) -> List[str]:
        return [ed.text() for ed in self.eq_editors]

    def get_boundary_conditions(self) -> List[str]:
        text = self.bc_editor.toPlainText().strip()
        if not text:
            return []
        return [line.strip() for line in text.split("\n") if line.strip()]

    def get_initial_values(self) -> Dict[int, Optional[float]]:
        result = {}
        for i, (chk, ed) in enumerate(zip(self.iv_checkboxes, self.iv_editors)):
            if chk.isChecked() and ed.text().strip():
                try:
                    result[i] = float(ed.text().strip())
                except ValueError:
                    result[i] = None
            else:
                result[i] = None
        return result

    def get_known_unknown_indices(self) -> Tuple[List[int], List[int]]:
        known = []
        unknown = []
        for i, chk in enumerate(self.iv_checkboxes):
            if chk.isChecked():
                known.append(i)
            else:
                unknown.append(i)
        return known, unknown

    def get_var_names(self) -> List[str]:
        text = self.input_varnames.text().strip()
        if text:
            names = [s.strip() for s in text.split(",")]
            n = self.spin_dim.value()
            # Дополняем автоматически
            while len(names) < n:
                names.append(f"x{len(names)}")
            return names[:n]
        return [f"x{i}" for i in range(self.spin_dim.value())]

# ---------------------------------------------------------------------------
# 7. BvpSolverApp — главное окно PyQt5
# ---------------------------------------------------------------------------


class BvpSolverApp(QMainWindow):
    """
    Универсальный GUI для решения краевых задач ODE.
    """

    TRANSLATIONS = {
        "zh": {
            "window_title": "通用BVP求解器",
            "menu_language": "语言", "lang_chinese": "中文 (Ctrl+1)",
            "lang_russian": "俄文 (Ctrl+2)",
            "menu_theme": "主题", "theme_bg": "背景颜色...",
            "theme_frame": "框架颜色...",
            "menu_help": "帮助", "help_doc": "使用说明",
            "help_about": "关于作者",
            "help_title": "使用指南", "about_title": "关于",
            "help_html": (
                "<h3>BVP 求解器 - 使用指南</h3>"
                "<p><b>1. 新建任务</b><br>点击 [+] 创建新任务或加载已保存的 JSON</p>"
                "<p><b>2. 输入 ODE</b><br>dx/dt = ... 用变量名 (x0, x1, x2, ...)</p>"
                "<p><b>3. 边界条件</b><br>每行一个: x0_T - 1.0 (x0_T = x0 at t=T)</p>"
                "<p><b>4. 初始条件</b><br>勾选=已知数值, 不勾选=未知(由求解器找)</p>"
                "<p><b>5. 初始猜测</b><br>格式: 0, 6.28 (逗号分隔, 支持 pi)</p>"
                "<p><b>6. 求解 & 绘图</b><br>[Solve] -> [Plot] 选 X/Y 轴变量</p>"
            ),
            "about_html": (
                "<h3>Программа для решения краевой задачи "
                "методом продолжения по параметру</h3>"
                "<p>Преподаватели: Аввакумов С.Н., Орлов М.С.</p>"
                "<p>Автор: Ли Хунюй 313 группа, "
                "Факультет ВМК кафедра ОУ.</p>"
                "<p>2026г.</p>"
            ),
            # 任务库
            "task_library": "任务库 / Task Library",
            "task_col_num": "#",
            "task_col_name": "名称 / Name",
            "task_col_status": "状态 / Status",
            "btn_add": "添加",
            "btn_add_tip": "添加新任务",
            "btn_del": "删除",
            "btn_del_tip": "删除选中任务",
            "btn_load": "加载",
            "btn_load_tip": "从JSON文件加载任务",
            "btn_save": "保存",
            "btn_save_tip": "保存任务到JSON文件",
            "btn_task_save": "保存",
            "btn_task_save_tip": "保存当前参数到任务",
            "task_name_placeholder": "任务名称 / Task name",
            "dialog_new_task": "新任务",
            "dialog_task_name": "任务名称 / Task name:",
            "dialog_delete": "删除确认",
            "dialog_delete_msg": "删除任务 '{name}'?",
            "dialog_validation": "参数验证错误",
            # 方程编辑器
            "tab_equations": "ODE方程组 (dx_i/dt = ...)",
            "dim_label": "系统维数 / System dimension:",
            "bc_title": "边界条件 R[x(0), x(T)] = 0",
            "bc_placeholder": "每行一个条件。变量: x0_0, x1_0, ..., x0_T, x1_T, ...\n"
                             "示例 26.1:\nx0_T - 1.0738644361\nx2_T + 1.0995343576\n"
                             "示例 26.2:\nx0_0 - x3_0\nx0_T - x3_0\nx1_T",
            "iv_title": "初始条件 (已知=数值, 未知=留空)",
            "iv_known": "x{idx}(0) 已知 / known",
            "var_names_label": "变量名 (逗号分隔):",
            "tab_params": "参数与方法",
            "progress_title": "求解进度",
            # 按钮
            "btn_solve": "求解", "btn_plot": "绘图",
            "btn_export": "导出", "btn_clear": "清空",

            # 参数标签
            "param_T": "终端时间 T:",
            "param_guess": "初始猜测 (p0):",
            "param_eps": "精度 eps:",
            "param_method": "积分方法:",
            "param_solver": "求解方法:",
            "param_steps": "延续步数:",
            # 求解方法选项
            "solver_cont": "参数延续 / Continuation",
            "solver_shoot": "打靶法 / Shooting",
            # 状态
            "status_ready": "就绪",
            "status_solving": "正在求解...",
            "status_solved": "求解完成!",
            "status_error": "求解失败",
            "status_saved": "已保存: {name}",
            "err_no_solution": "无可用的解，请先求解。",
            "confirm_clear": "确定要清空所有数据?",
            "solve_success": "求解成功!\n\n方法: {method}\n"
                             "迭代次数: {iter}\n"
                             "残差范数: {res_norm:.4e}\n"
                             "完整状态 x(0): {full_state}\n"
                             "未知参数: {p_opt}",
        },
        "ru": {
            "window_title": "Универсальный решатель BVP",
            "menu_language": "Язык", "lang_chinese": "Китайский (Ctrl+1)",
            "lang_russian": "Русский (Ctrl+2)",
            "menu_theme": "Тема", "theme_bg": "Цвет фона...",
            "theme_frame": "Цвет рамки...",
            "menu_help": "Помощь", "help_doc": "Руководство",
            "help_about": "Об авторе",
            "help_title": "Руководство", "about_title": "О программе",
            "help_html": (
                "<h3>BVP Solver - Руководство</h3>"
                "<p><b>1. Создание задачи</b><br>Новая [+] или загрузка из JSON</p>"
                "<p><b>2. Уравнения ODE</b><br>dx/dt = ... с именами переменных</p>"
                "<p><b>3. Граничные условия</b><br>Одно на строку: x0_T - 1.07</p>"
                "<p><b>4. Начальные условия</b><br>OK = известно, пусто = неизвестно</p>"
                "<p><b>5. Начальное приближение</b><br>Формат: 2, 0, 2*pi, 2</p>"
                "<p><b>6. Решить и построить график</b><br>[Solve] -> [Plot]</p>"
            ),
            "about_html": (
                "<h3>Программа для решения краевой задачи "
                "методом продолжения по параметру</h3>"
                "<p>Преподаватели: Аввакумов С.Н., Орлов М.С.</p>"
                "<p>Автор: Ли Хунюй 313 группа, "
                "Факультет ВМК кафедра ОУ.</p>"
                "<p>2026г.</p>"
            ),
            # Библиотека задач
            "task_library": "Библиотека задач / Task Library",
            "task_col_num": "#",
            "task_col_name": "Название / Name",
            "task_col_status": "Статус / Status",
            "btn_add": "Добавить",
            "btn_add_tip": "Добавить новую задачу",
            "btn_del": "Удалить",
            "btn_del_tip": "Удалить выбранную задачу",
            "btn_load": "Загрузить",
            "btn_load_tip": "Загрузить задачи из JSON",
            "btn_save": "Сохранить",
            "btn_save_tip": "Сохранить задачи в JSON",
            "btn_task_save": "Save",
            "btn_task_save_tip": "Сохранить текущие параметры в выбранную задачу",
            "task_name_placeholder": "Имя задачи / Task name",
            "dialog_new_task": "Новая задача",
            "dialog_task_name": "Имя задачи / Task name:",
            "dialog_delete": "Удаление",
            "dialog_delete_msg": "Удалить задачу '{name}'?",
            "dialog_validation": "Validation Error",
            # Редактор уравнений
            "tab_equations": "Уравнения ODE (dx_i/dt = ...)",
            "dim_label": "Размерность системы / System dimension:",
            "bc_title": "Граничные условия (R[x(0), x(T)] = 0)",
            "bc_placeholder": "Одно условие на строку. Переменные: x0_0, x1_0, ..., x0_T, x1_T, ...\n"
                             "Пример 26.1:\nx0_T - 1.0738644361\nx2_T + 1.0995343576\n"
                             "Пример 26.2:\nx0_0 - x2_0\nx0_T - x2_0\nx1_T",
            "iv_title": "Начальные условия (известные=число, неизвестные=пусто)",
            "iv_known": "x{idx}(0) известно / known",
            "var_names_label": "Имена переменных (через запятую):",
            "tab_params": "Параметры и метод",
            "progress_title": "Progress",
            # Кнопки
            "btn_solve": "Решить", "btn_plot": "График",
            "btn_export": "Экспорт", "btn_clear": "Очистить",

            "param_T": "Время T:",
            "param_guess": "Начальное приближение (p0):",
            "param_eps": "Точность eps:",
            "param_method": "Метод интегрирования:",
            "param_solver": "Метод решения:",
            "param_steps": "Шаги продолжения:",
            "solver_cont": "Продолжение / Continuation",
            "solver_shoot": "Стрельба / Shooting",
            "status_ready": "Готово",
            "status_solving": "Решаю...",
            "status_solved": "Решение найдено!",
            "status_error": "Ошибка",
            "status_saved": "Сохранено: {name}",
            "err_no_solution": "Нет решения. Сначала нажмите 'Решить'.",
            "confirm_clear": "Очистить все данные?",
            "solve_success": "Решение найдено!\n\nМетод: {method}\n"
                             "Итераций: {iter}\n"
                             "‖Φ‖: {res_norm:.4e}\n"
                             "x(0)  = {full_state}\n"
                             "p_opt = {p_opt}",
        }
    }

    def __init__(self):
        super().__init__()
        self.current_lang = "ru"
        self.bg_color = "#f5f6fa"
        self.frame_color = "#2c3e50"
        self.panel_bg = "#ffffff"
        self.accent_color = "#2980b9"
        self.text_color = "#2c3e50"

        # ---- Библиотека задач -------------------------------------------
        self.tasks: List[Dataset] = []          # все задачи
        self.current_task_idx: int = -1          # текущая выбранная
        self._suppress_sync: bool = False       # флаг блокировки синхронизации

        # Хранение результатов
        self.last_result: Optional[dict] = None
        self.last_failed_result: Optional[dict] = None
        self.all_results: List[dict] = []
        self._task_status: Dict[int, str] = {}  # task_idx -> "solved" | "error"

        self.init_ui()
        # Загружаем встроенные примеры как задачи
        self._load_builtin_tasks()
        self._refresh_task_table()
        self.update_language(self.current_lang)
        self.apply_theme()

    def _load_builtin_tasks(self):
        """Built-in examples removed. User imports via Load button."""
        pass

    # ------------------------------------------------------------------
    # Управление библиотекой задач
    # ------------------------------------------------------------------

    def _refresh_task_table(self):
        """Обновляет таблицу задач."""
        self.task_table.setRowCount(len(self.tasks))
        for i, ds in enumerate(self.tasks):
            # Номер
            item_num = QTableWidgetItem(str(i + 1))
            item_num.setFlags(item_num.flags() & ~Qt.ItemIsEditable)
            self.task_table.setItem(i, 0, item_num)
            # Имя
            item_name = QTableWidgetItem(ds.name)
            item_name.setFlags(item_name.flags() & ~Qt.ItemIsEditable)
            self.task_table.setItem(i, 1, item_name)
            # Статус
            st = self._task_status.get(i, "")
            if st == "solved":
                status_text = "✓"
            elif st == "error":
                status_text = "✗"
            else:
                status_text = ""
            item_status = QTableWidgetItem(status_text)
            item_status.setFlags(item_status.flags() & ~Qt.ItemIsEditable)
            if st == "solved":
                item_status.setBackground(QColor(39, 174, 96, 60))
                item_status.setForeground(QColor(39, 174, 96))
            elif st == "error":
                item_status.setBackground(QColor(231, 76, 60, 60))
                item_status.setForeground(QColor(231, 76, 60))
            self.task_table.setItem(i, 2, item_status)

    def _on_task_selected(self):
        """При выборе задачи в таблице — загружаем её в редактор."""
        if self._suppress_sync:
            return
        selected = self.task_table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        if 0 <= row < len(self.tasks):
            self.current_task_idx = row
            ds = self.tasks[row]
            self.input_task_name.setText(ds.name)
            self._sync_task_to_editor(ds)

    def _sync_task_to_editor(self, ds: Dataset):
        """Загружает Dataset в редактор уравнений и параметры."""
        self._suppress_sync = True
        try:
            # --- Размерность ---
            n = ds.dim()
            self.eq_editor.spin_dim.setValue(n)
            # Перестраиваем редакторы
            self.eq_editor._on_dim_changed(n)

            # --- Уравнения ---
            for i, eq in enumerate(ds.equations):
                if i < len(self.eq_editor.eq_editors):
                    self.eq_editor.eq_editors[i].setText(eq)

            # --- Имена переменных ---
            if ds.var_names:
                self.eq_editor.input_varnames.setText(", ".join(ds.var_names))
            else:
                self.eq_editor.input_varnames.setText(", ".join(
                    [f"x{i}" for i in range(n)]
                ))

            # --- Граничные условия ---
            self.eq_editor.bc_editor.setPlainText(
                "\n".join(ds.boundary_conditions)
            )

            # --- Начальные значения ---
            for i in range(n):
                val = ds.initial_values.get(i)
                chk = self.eq_editor.iv_checkboxes[i]
                ed = self.eq_editor.iv_editors[i]
                if val is not None:
                    chk.setChecked(True)
                    ed.setText(str(val))
                else:
                    chk.setChecked(False)
                    ed.setText("")

            # --- Параметры ---
            self.input_T.setValue(ds.T)
            self.input_eps.setValue(ds.eps)
            self.combo_method.setCurrentText(ds.method)
            # solver_method
            idx = self.combo_solver.findData(ds.solver_method)
            if idx >= 0:
                self.combo_solver.setCurrentIndex(idx)
            self.input_steps.setValue(ds.continuation_steps)

            # --- Guess ---
            guess_str = ", ".join(str(g) for g in ds.guess)
            self.input_guess.setText(guess_str)

        finally:
            self._suppress_sync = False

    def _sync_editor_to_dataset(self, ds: Dataset) -> Dataset:
        """Собирает Dataset из текущего состояния GUI."""
        eqs = self.eq_editor.get_equations()
        bcs = self.eq_editor.get_boundary_conditions()
        iv = self.eq_editor.get_initial_values()
        known, unknown = self.eq_editor.get_known_unknown_indices()
        var_names = self.eq_editor.get_var_names()

        # Парсим guess: поддержка pi, 2π, 2pi, 2*pi и n-мерного вектора
        guess_str = self.input_guess.text().strip()
        raw_guess = []
        if guess_str:
            # Унифицируем разделители: запятые, точки с запятой
            for s in re.split(r"[,;，]", guess_str):
                s_clean = s.strip().lower()
                if not s_clean:
                    continue
                # 2π, 2pi → 2*pi (вставляем * между числом и pi)
                s_clean = re.sub(r"(\d)\s*π", r"\1*pi", s_clean)
                s_clean = re.sub(r"(\d)pi\b", r"\1*pi", s_clean)
                # π → pi
                s_clean = s_clean.replace("π", "pi")
                try:
                    raw_guess.append(float(s_clean))
                except ValueError:
                    try:
                        expr = parse_expr(s_clean, global_dict={
                            "pi": sp.pi, "e": sp.E,
                            "sin": sp.sin, "cos": sp.cos, "sqrt": sp.sqrt,
                            "Integer": sp.Integer, "Float": sp.Float,
                        }, transformations=standard_transformations, evaluate=True)
                        raw_guess.append(float(expr.evalf()))
                    except Exception:
                        pass
        n_dim = len(eqs)
        # Если guess длины n — полный вектор, извлекаем unknown
        if len(raw_guess) == n_dim and len(unknown) < n_dim:
            guess = [raw_guess[i] for i in unknown]
        else:
            guess = raw_guess
            while len(guess) < len(unknown):
                guess.append(0.0)
            guess = guess[:len(unknown)]

        ds.name = self.input_task_name.text().strip() or ds.name
        ds.equations = eqs
        ds.var_names = var_names
        ds.T = self.input_T.value()
        ds.initial_values = iv
        ds.boundary_conditions = bcs
        ds.guess = guess
        ds.eps = self.input_eps.value()
        ds.method = self.combo_method.currentText()
        ds.solver_method = self.combo_solver.currentData()
        ds.continuation_steps = self.input_steps.value()
        ds.known_indices = known
        ds.unknown_indices = unknown
        ds.t_star = 0.0
        return ds

    def _task_add_new(self):
        """Добавляет новую пустую задачу."""
        t = self.TRANSLATIONS[self.current_lang]
        name, ok = QInputDialog.getText(
            self, t["dialog_new_task"], t["dialog_task_name"]
        )
        if not ok:
            return
        ds = Dataset(
            name=name or f"Task {len(self.tasks) + 1}",
            equations=["x1", "-x0"],
            var_names=["x0", "x1"],
            T=np.pi,
            initial_values={0: None, 1: None},
            boundary_conditions=["x0_T"],
            guess=[1.0],
            unknown_indices=[0, 1],
        )
        self.tasks.append(ds)
        self._refresh_task_table()
        # Выбираем новую задачу
        self.task_table.selectRow(len(self.tasks) - 1)

    def _task_delete(self):
        """Удаляет выбранную задачу."""
        if self.current_task_idx < 0 or self.current_task_idx >= len(self.tasks):
            return
        reply = QMessageBox.question(
            self, "Удаление", f"Удалить задачу '{self.tasks[self.current_task_idx].name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.tasks.pop(self.current_task_idx)
            self.current_task_idx = -1
            self._refresh_task_table()
            self.input_task_name.setText("")

    def _task_save_current(self):
        """Сохраняет текущие параметры редактора в выбранную задачу."""
        if self.current_task_idx < 0 or self.current_task_idx >= len(self.tasks):
            # Нет выбранной задачи — создаём новую
            self._task_add_new()
            return
        ds = self.tasks[self.current_task_idx]
        self._sync_editor_to_dataset(ds)
        self._refresh_task_table()
        self.progress_detail.setText(f"已保存: {ds.name}")

    def _tasks_load_json(self):
        """Загружает задачи из JSON-файла."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить задачи", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    self.tasks.append(Dataset.from_dict(item))
            elif isinstance(data, dict):
                self.tasks.append(Dataset.from_dict(data))
            self._refresh_task_table()
            self.progress_detail.setText(f"已加载 {len(self.tasks)} 个任务")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки", str(e))

    def _tasks_save_json(self):
        """Сохраняет все задачи в JSON-файл."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить задачи", "tasks.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            data = [ds.to_dict() for ds in self.tasks]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.progress_detail.setText(f"已保存 {len(data)} 个任务")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка сохранения", str(e))

    def init_ui(self):
        self.create_menu_bar()
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # ===== Главный разделитель: Левая панель задач | Центр + Правая панель =====
        main_splitter = QSplitter(Qt.Horizontal)

        # --- ЛЕВАЯ ПАНЕЛЬ: Библиотека задач ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(8)

        # Заголовок
        self.label_task_library = QLabel()
        self.label_task_library.setStyleSheet(
            "font-weight: bold; font-size: 14px; color: #2c3e50;"
        )
        left_layout.addWidget(self.label_task_library)

        # Таблица задач
        self.task_table = QTableWidget()
        self.task_table.setColumnCount(3)
        self.task_table.setHorizontalHeaderLabels(
            ["#", "Название / Name", "Статус / Status"]
        )
        self.task_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.task_table.setSelectionMode(QTableWidget.SingleSelection)
        self.task_table.setColumnWidth(0, 35)
        self.task_table.setColumnWidth(1, 175)
        self.task_table.setColumnWidth(2, 75)
        self.task_table.setMinimumWidth(290)
        self.task_table.setMaximumWidth(310)
        self.task_table.horizontalHeader().setStretchLastSection(False)
        self.task_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.task_table.verticalHeader().setVisible(False)
        self.task_table.itemSelectionChanged.connect(self._on_task_selected)
        left_layout.addWidget(self.task_table, 1)

        # Кнопки управления задачами (2x2 grid, aligned with table)
        task_btn_layout = QGridLayout()
        task_btn_layout.setSpacing(4)
        self.btn_task_add = QPushButton("添加")
        self.btn_task_add.setToolTip("")
        self.btn_task_add.clicked.connect(self._task_add_new)
        self.btn_task_delete = QPushButton("删除")
        self.btn_task_delete.setToolTip("")
        self.btn_task_delete.clicked.connect(self._task_delete)
        self.btn_task_load = QPushButton("加载")
        self.btn_task_load.setToolTip("")
        self.btn_task_load.clicked.connect(self._tasks_load_json)
        self.btn_task_save = QPushButton("保存")
        self.btn_task_save.setToolTip("")
        self.btn_task_save.clicked.connect(self._tasks_save_json)
        task_btn_layout.addWidget(self.btn_task_add, 0, 0)
        task_btn_layout.addWidget(self.btn_task_delete, 0, 1)
        task_btn_layout.addWidget(self.btn_task_load, 1, 0)
        task_btn_layout.addWidget(self.btn_task_save, 1, 1)
        left_layout.addLayout(task_btn_layout)

        # Имя текущей задачи + кнопка сохранить
        name_layout = QHBoxLayout()
        self.input_task_name = QLineEdit()
        self.input_task_name.setPlaceholderText("")
        self.btn_task_save_current = QPushButton("Save")
        self.btn_task_save_current.setToolTip(
            "Сохранить текущие параметры в выбранную задачу"
        )
        self.btn_task_save_current.clicked.connect(self._task_save_current)
        name_layout.addWidget(self.input_task_name, 1)
        name_layout.addWidget(self.btn_task_save_current)
        left_layout.addLayout(name_layout)

        main_splitter.addWidget(left_panel)

        # --- ЦЕНТР + ПРАВАЯ ПАНЕЛЬ ---
        center_right_splitter = QSplitter(Qt.Horizontal)

        # Центр — редактор уравнений
        self.eq_editor = EquationEditor()
        self.eq_editor.changed.connect(self._on_equations_changed)
        center_right_splitter.addWidget(self.eq_editor)

        # Правая панель — параметры, прогресс, кнопки
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(10)

        # --- Параметры ---
        self.group_params = QGroupBox()
        params_grid = QGridLayout(self.group_params)
        params_grid.setVerticalSpacing(8)

        row = 0
        self.label_T = QLabel()
        self.input_T = QDoubleSpinBox()
        self.input_T.setRange(0.01, 1000)
        self.input_T.setDecimals(6)
        self.input_T.setValue(1.0)
        params_grid.addWidget(self.label_T, row, 0)
        params_grid.addWidget(self.input_T, row, 1)

        row += 1
        self.label_guess = QLabel()
        self.input_guess = QLineEdit("0")
        params_grid.addWidget(self.label_guess, row, 0)
        params_grid.addWidget(self.input_guess, row, 1)

        row += 1
        self.label_eps = QLabel()
        self.input_eps = QDoubleSpinBox()
        self.input_eps.setRange(1e-15, 1)
        self.input_eps.setDecimals(12)
        self.input_eps.setValue(1e-8)
        params_grid.addWidget(self.label_eps, row, 0)
        params_grid.addWidget(self.input_eps, row, 1)

        row += 1
        self.label_method = QLabel()
        self.combo_method = QComboBox()
        self.combo_method.addItems(
            ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]
        )
        params_grid.addWidget(self.label_method, row, 0)
        params_grid.addWidget(self.combo_method, row, 1)

        row += 1
        self.label_solver = QLabel()
        self.combo_solver = QComboBox()
        self.combo_solver.addItem("Продолжение / Continuation", "continuation")
        self.combo_solver.addItem("Стрельба / Shooting", "shooting")
        params_grid.addWidget(self.label_solver, row, 0)
        params_grid.addWidget(self.combo_solver, row, 1)

        row += 1
        self.label_steps = QLabel()
        self.input_steps = QSpinBox()
        self.input_steps.setRange(5, 500)
        self.input_steps.setValue(50)
        params_grid.addWidget(self.label_steps, row, 0)
        params_grid.addWidget(self.input_steps, row, 1)

        right_layout.addWidget(self.group_params)

        # --- Прогресс ---
        self.group_progress = QGroupBox()
        prog_layout = QVBoxLayout(self.group_progress)
        self.progress_bar_label = QLabel("Ready")
        self.progress_detail = QLabel("")
        self.progress_detail.setStyleSheet("color: #636e72; font-size: 12px;")
        prog_layout.addWidget(self.progress_bar_label)
        prog_layout.addWidget(self.progress_detail)
        right_layout.addWidget(self.group_progress)

        # --- Кнопки действий ---
        btn_layout = QHBoxLayout()
        self.btn_solve = QPushButton()
        self.btn_solve.setMinimumWidth(100)
        self.btn_solve.clicked.connect(self.on_solve)
        self.btn_plot = QPushButton()
        self.btn_plot.setMinimumWidth(100)
        self.btn_plot.clicked.connect(self.on_plot)
        self.btn_export = QPushButton()
        self.btn_export.setMinimumWidth(100)
        self.btn_export.clicked.connect(self.on_export)
        self.btn_clear = QPushButton()
        self.btn_clear.setMinimumWidth(100)
        self.btn_clear.clicked.connect(self.on_clear)
        btn_layout.addWidget(self.btn_solve)
        btn_layout.addWidget(self.btn_plot)
        btn_layout.addWidget(self.btn_export)
        btn_layout.addWidget(self.btn_clear)
        right_layout.addLayout(btn_layout)

        right_layout.addStretch()
        center_right_splitter.addWidget(right)
        center_right_splitter.setSizes([550, 400])
        main_splitter.addWidget(center_right_splitter)
        main_splitter.setSizes([260, 960])
        main_layout.addWidget(main_splitter, 1)

        # (状态栏已移除 — 状态通过进度标签显示)
        self.setMinimumSize(1250, 780)
        self.resize(1400, 850)
        self.setup_shortcuts()

    def create_menu_bar(self):
        menubar = self.menuBar()
        self.menu_language = menubar.addMenu("Language")
        self.action_zh = QAction("中文 (Ctrl+1)", self)
        self.action_zh.setShortcut("Ctrl+1")
        self.action_zh.triggered.connect(lambda: self.update_language("zh"))
        self.menu_language.addAction(self.action_zh)
        self.action_ru = QAction("Русский (Ctrl+2)", self)
        self.action_ru.setShortcut("Ctrl+2")
        self.action_ru.triggered.connect(lambda: self.update_language("ru"))
        self.menu_language.addAction(self.action_ru)

        self.menu_theme = menubar.addMenu("Theme")
        self.action_bg = QAction("Background Color...", self)
        self.action_bg.triggered.connect(self.pick_bg_color)
        self.menu_theme.addAction(self.action_bg)
        self.action_frame = QAction("Frame Color...", self)
        self.action_frame.triggered.connect(self.pick_frame_color)
        self.menu_theme.addAction(self.action_frame)

        self.menu_help = menubar.addMenu("Help")
        self.action_help_doc = QAction("Documentation", self)
        self.action_help_doc.triggered.connect(self.show_help)
        self.menu_help.addAction(self.action_help_doc)
        self.action_help_about = QAction("About", self)
        self.action_help_about.triggered.connect(self.show_about)
        self.menu_help.addAction(self.action_help_about)

    def show_help(self):
        t = self.TRANSLATIONS.get(self.current_lang, self.TRANSLATIONS["zh"])
        QMessageBox.information(self, t["help_title"], t["help_html"])

    def show_about(self):
        t = self.TRANSLATIONS.get(self.current_lang, self.TRANSLATIONS["zh"])
        QMessageBox.about(self, t["about_title"], t["about_html"])

    def setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+1"), self).activated.connect(
            lambda: self.update_language("zh"))
        QShortcut(QKeySequence("Ctrl+2"), self).activated.connect(
            lambda: self.update_language("ru"))

    def pick_bg_color(self):
        c = QColorDialog.getColor(QColor(self.bg_color), self, "Background")
        if c.isValid():
            self.bg_color = c.name()
            self.apply_theme()

    def pick_frame_color(self):
        c = QColorDialog.getColor(QColor(self.frame_color), self, "Frame")
        if c.isValid():
            self.frame_color = c.name()
            self.apply_theme()

    def apply_theme(self):
        style = f"""
            QMainWindow {{ background-color: {self.bg_color}; }}
            QWidget {{ background-color: {self.bg_color}; color: {self.text_color}; }}
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {self.frame_color};
                border-radius: 6px;
                margin-top: 8px;
                padding: 12px;
                background-color: {self.panel_bg};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px;
                color: {self.frame_color};
            }}
            QPushButton {{
                background-color: {self.accent_color};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 18px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self._darken(self.accent_color)};
            }}
            QPushButton:disabled {{
                background-color: #bdc3c7;
            }}
            QLineEdit, QTextEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
                padding: 5px;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                background-color: {self.panel_bg};
            }}
            QLineEdit:focus, QTextEdit:focus, QDoubleSpinBox:focus,
            QSpinBox:focus, QComboBox:focus {{
                border: 2px solid {self.accent_color};
            }}
            QLabel {{ color: {self.text_color}; }}
            QMenuBar {{
                background-color: {self.frame_color};
                color: white;
            }}
            QMenuBar::item:selected {{
                background-color: {self.accent_color};
            }}
        """
        self.setStyleSheet(style)

    def _darken(self, hex_color, factor=0.8):
        c = QColor(hex_color)
        c.setRed(int(c.red() * factor))
        c.setGreen(int(c.green() * factor))
        c.setBlue(int(c.blue() * factor))
        return c.name()

    def _build_dataset(self) -> Dataset:
        """Собирает Dataset из текущего состояния GUI.
        Использует deepcopy, чтобы не изменять оригинал в self.tasks.
        """
        import copy
        if self.current_task_idx < 0:
            ds = Dataset()
        else:
            ds = copy.deepcopy(self.tasks[self.current_task_idx])
        return self._sync_editor_to_dataset(ds)

    def _on_equations_changed(self):
        """Обновляет guess-подсказку при изменении уравнений."""
        _, unknown = self.eq_editor.get_known_unknown_indices()
        n_unknown = len(unknown)
        current = self.input_guess.text().strip()
        if not current:
            self.input_guess.setText(", ".join(["0.0"] * n_unknown)
            )

    # ------------------------------------------------------------------
    # Решение
    # ------------------------------------------------------------------

    def on_solve(self):
        t = self.TRANSLATIONS[self.current_lang]
        self.btn_solve.setEnabled(False)
        # 重置进度标签样式（清除之前的绿色）
        self.progress_bar_label.setStyleSheet("")
        self.progress_bar_label.setText(t["status_solving"])

        try:
            dataset = self._build_dataset()
            errors = dataset.validate()
            if errors:
                QMessageBox.warning(self, "Validation Error", "\n".join(errors))
                self.btn_solve.setEnabled(True)
                self.progress_bar_label.setText(t["status_ready"])
                return
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.btn_solve.setEnabled(True)
            return

        self.worker = SolverWorker(dataset)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_solve_done)
        self.worker.error.connect(self._on_solve_error)
        self.worker.start()

    def _on_progress(self, method: str, percent: int, message: str):
        self.progress_bar_label.setText(f"[{percent}%] {method}")
        self.progress_detail.setText(message)

    def _on_solve_done(self, result: dict):
        t = self.TRANSLATIONS[self.current_lang]
        self.btn_solve.setEnabled(True)

        if not is_result_acceptable(result):
            self.last_result = None
            self.last_failed_result = result
            self.progress_bar_label.setStyleSheet(
                "background-color: #e74c3c; color: white; "
                "padding: 4px 8px; border-radius: 4px; font-weight: bold;"
            )
            self.progress_bar_label.setText(t["status_error"])
            diagnostic = self._format_failed_result(result)
            self.progress_detail.setText(diagnostic)
            if self.current_task_idx >= 0:
                self._task_status[self.current_task_idx] = "error"
                self._refresh_task_table()
            QMessageBox.warning(self, "BVP result rejected", diagnostic)
            return

        # --- 计算辅助输出变量 ---
        result = self._compute_aux_outputs(result)

        self.last_failed_result = None
        self.last_result = result
        self.all_results.append(result)

        # --- 构建完整的 x(0) 状态向量 ---
        full_state = self._build_full_state(result["p_opt"])
        full_str = np.array2string(full_state, precision=6, separator=", ")

        p_opt_str = np.array2string(
            result["p_opt"], precision=6, separator=", "
        )
        msg = t["solve_success"].format(
            method=result["method"],
            iter=result["iterations"],
            res_norm=result["residual_norm"],
            p_opt=p_opt_str,
            full_state=full_str,
        )
        # 求解成功 — 进度标签变绿色，表格状态列标记 ✓
        self.progress_bar_label.setStyleSheet(
            "background-color: #27ae60; color: white; "
            "padding: 4px 8px; border-radius: 4px; font-weight: bold;"
        )
        self.progress_bar_label.setText(t["status_solved"])
        self.progress_detail.setText(
            f"‖Φ‖ = {result['residual_norm']:.4e}"
        )
        if self.current_task_idx >= 0:
            self._task_status[self.current_task_idx] = "solved"
            self._refresh_task_table()
        QMessageBox.information(self, "Result", msg)

    @staticmethod
    def _format_failed_result(result: dict) -> str:
        residual = np.asarray(result.get("boundary_residual", []), dtype=float)
        residual_text = np.array2string(residual, precision=6, separator=", ")
        residual_norm = result.get("boundary_residual_norm", float("inf"))
        boundary_atol = result.get("boundary_atol", "unknown")
        metadata = result.get("solver_metadata", {})
        diagnostic_keys = (
            "optimizer",
            "fallback_used",
            "root",
            "least_squares",
            "failure_step",
            "failure_reason",
        )
        metadata_summary = {
            key: metadata[key] for key in diagnostic_keys if key in metadata
        }
        return (
            f"method: {result.get('method', 'unknown')}\n"
            f"status: {result.get('status', 'unknown')}\n"
            f"message: {result.get('message', 'No diagnostic message')}\n"
            f"optimizer_success: {result.get('optimizer_success', False)}\n"
            f"ivp_success: {result.get('ivp_success', False)}\n"
            f"finite_success: {result.get('finite_success', False)}\n"
            f"boundary_success: {result.get('boundary_success', False)}\n"
            f"boundary_residual: {residual_text}\n"
            f"boundary_residual_norm: {residual_norm}\n"
            f"boundary_atol: {boundary_atol}\n"
            f"solver_metadata: {metadata_summary}"
        )

    def _build_full_state(self, p_opt: np.ndarray) -> np.ndarray:
        """从 p_opt 和 known 值构建完整的 x(0) 状态向量."""
        if self.current_task_idx < 0 or self.current_task_idx >= len(self.tasks):
            return p_opt
        ds = self.tasks[self.current_task_idx]
        n = ds.dim()
        state = np.zeros(n)
        # 填入 known 值
        for idx in ds.known_indices:
            val = ds.initial_values.get(idx)
            if val is not None:
                state[idx] = float(val)
        # 填入 unknown 值 (p_opt)
        for i, idx in enumerate(ds.unknown_indices):
            if i < len(p_opt):
                state[idx] = p_opt[i]
        return state

    def _compute_aux_outputs(self, result: dict) -> dict:
        """计算辅助输出变量并存储在结果中.
        
        对于 26.4 (lunula): u1/u2 从 ODE 方程反推,
        自动跟随方程中的 mu 值(用户可能修改了 mu).
        dx0/dt = T*(x1+u1) -> u1 = dx0/dt/T - x1
        dx1/dt = T*(-1.5*x0-0.25*x1+u2) -> u2 = dx1/dt/T + 1.5*x0 + 0.25*x1
        """
        if self.current_task_idx < 0 or self.current_task_idx >= len(self.tasks):
            return result
        ds = self.tasks[self.current_task_idx]
        try:
            parser = SymPyParser(ds.equations, ds.var_names)
            parser.parse()
            parser.lambdify_all()

            sol = result["sol"]
            t0, t1 = float(sol.t[0]), float(sol.t[-1])
            td = np.linspace(t0, t1, 500)
            if hasattr(sol, 'sol') and callable(sol.sol):
                yd = sol.sol(td)
            else:
                yd = np.array([np.interp(td, sol.t, sol.y[j])
                               for j in range(len(ds.var_names))])

            aux_data = {}
            aux_names = []

            # --- 26.4 lunula: 从 ODE 反推 u1, u2 ---
            is_lunula = ("26.4" in ds.name or "Лунка" in ds.name)
            if is_lunula and len(ds.var_names) >= 5:
                u1_vals = []
                u2_vals = []
                for i, t in enumerate(td):
                    f_vals = parser.f(t, yd[:, i])
                    T = yd[4, i]  # x4 = T
                    if abs(T) > 1e-12:
                        u1 = f_vals[0] / T - yd[1, i]
                        u2 = f_vals[1] / T + 1.5 * yd[0, i] + 0.25 * yd[1, i]
                    else:
                        u1 = u2 = 0.0
                    u1_vals.append(u1)
                    u2_vals.append(u2)
                aux_data["u1"] = np.array(u1_vals)
                aux_data["u2"] = np.array(u2_vals)
                aux_names.extend(["u1", "u2"])

            # --- 一般 aux_outputs ---
            if ds.aux_outputs:
                aux_fns = parser.lambdify_aux(ds.aux_outputs)
                for name, fn in aux_fns.items():
                    if name not in aux_data:  # 不覆盖 lunula 反推
                        vals = np.array([fn(t, *yd[:, i]) for i, t in enumerate(td)])
                        aux_data[name] = vals
                        aux_names.append(name)

            if aux_data:
                result["aux"] = aux_data
                result["aux_names"] = aux_names
        except Exception:
            pass
        return result

    def _on_solve_error(self, msg: str):
        t = self.TRANSLATIONS[self.current_lang]
        self.btn_solve.setEnabled(True)
        self.progress_bar_label.setStyleSheet(
            "background-color: #e74c3c; color: white; "
            "padding: 4px 8px; border-radius: 4px; font-weight: bold;"
        )
        self.progress_bar_label.setText(t["status_error"])
        self.progress_detail.setText(msg[:200])
        if self.current_task_idx >= 0:
            self._task_status[self.current_task_idx] = "error"
            self._refresh_task_table()
        QMessageBox.critical(self, "Error", msg)

    # ------------------------------------------------------------------
    # Построение графиков
    # ------------------------------------------------------------------

    def on_plot(self):
        t = self.TRANSLATIONS[self.current_lang]
        valid_results = [
            result for result in self.all_results if is_result_acceptable(result)
        ]
        if not valid_results:
            QMessageBox.warning(self, "Warning", t["err_no_solution"])
            return

        var_names = self.eq_editor.get_var_names()
        # 收集所有辅助变量名（去重）
        aux_names_set = set()
        for r in valid_results:
            aux_names_set.update(r.get("aux_names", []))
        aux_names = sorted(aux_names_set)
        if hasattr(self, '_plot_widget') and self._plot_widget is not None:
            try:
                self._plot_widget.update_data(var_names, valid_results, aux_names, lang=self.current_lang)
                self._plot_widget.show()
                self._plot_widget.raise_()
                self._plot_widget.activateWindow()
                return
            except RuntimeError:
                self._plot_widget = None
        self._plot_widget = IntegratedPlotWidget(var_names, valid_results, aux_names, lang=self.current_lang, parent=self)
        self._plot_widget.show()
        self._plot_widget.raise_()
        self._plot_widget.activateWindow()

    def _do_plot(self, all_results: List[dict], cfg: dict, var_names: List[str]):
        """
        Построение графика.
        Фазовый портрет: square figure + equal aspect (без tight_layout!).
        多Y叠加: 每个solution × 每个y_var 画一条曲线.
        """
        x_axis = cfg["x_axis"]
        y_axes = cfg.get("y_axes", [cfg.get("y_axis", var_names[0])])
        # 相图模式只取第一个Y
        if x_axis != "t":
            y_axes = [y_axes[0]]
        grid = cfg["grid"]
        equal = cfg["equal_aspect"]
        lw = cfg.get("line_width", 1.5)
        show_markers = cfg.get("show_markers", True)
        visible = set(cfg.get("visible_indices", list(range(len(all_results)))))

        colors = [(31/255,119/255,180/255),(255/255,127/255,14/255),
                  (44/255,160/255,44/255),(214/255,39/255,40/255),
                  (148/255,103/255,189/255),(140/255,86/255,75/255),
                  (227/255,119/255,194/255),(127/255,127/255,127/255),
                  (188/255,189/255,34/255),(23/255,190/255,207/255)]
        linestyles = ["-", "--", ":", "-."]

        is_phase = (x_axis != "t" and y_axes[0] != "t")

        def axis_data(name, t_vals, y_vals):
            if name == "t":
                return t_vals, "t"
            return y_vals[var_names.index(name)], name

        # ---- 1. Собираем данные ----
        # curves: (sol_idx, y_var_name, x_arr, y_arr)
        curves = []
        for i, r in enumerate(all_results):
            if i not in visible:
                continue
            try:
                sol = r["sol"]
                t0, t1 = float(sol.t[0]), float(sol.t[-1])
                td = np.linspace(t0, t1, 500)
                if hasattr(sol, 'sol') and callable(sol.sol):
                    yd = sol.sol(td)
                else:
                    yd = np.array([np.interp(td, sol.t, sol.y[j])
                                   for j in range(len(var_names))])
                xd, xl = axis_data(x_axis, td, yd)
                for y_name in y_axes:
                    try:
                        yd_arr, yl = axis_data(y_name, td, yd)
                        curves.append((i, y_name, np.asarray(xd), np.asarray(yd_arr)))
                    except Exception:
                        continue
            except Exception:
                continue

        if not curves:
            QMessageBox.warning(self, "Plot", "Нет данных")
            return

        # Диапазон
        all_x = np.concatenate([c[2] for c in curves])
        all_y = np.concatenate([c[3] for c in curves])
        x_min, x_max = float(all_x.min()), float(all_x.max())
        y_min, y_max = float(all_y.min()), float(all_y.max())
        x_pad = 0.1 * max(x_max - x_min, 1e-12)
        y_pad = 0.1 * max(y_max - y_min, 1e-12)

        # ---- 2. Figure ----
        figsize = (7, 7) if is_phase else (8, 6)
        fig, ax = plt.subplots(figsize=figsize)

        # Подписи
        _, x_lbl = axis_data(x_axis, None, np.zeros(len(var_names)))
        ax.set_xlabel(f"$\\mathit{{{x_lbl}}}$", fontsize=13)

        if is_phase:
            y_lbl = y_axes[0]
            title = f"${y_lbl}$ vs ${x_lbl}$"
            ax.set_ylabel(f"$\\mathit{{{y_lbl}}}$", fontsize=13)
        elif x_axis == "t":
            # 时间图: Y轴标注选中的变量名
            y_label_str = ", ".join(y_axes)
            ax.set_ylabel(f"$\\mathit{{{y_label_str}}}$", fontsize=12)
            if len(y_axes) == 1:
                title = f"${y_axes[0]}(t)$"
            else:
                title = "状态变量 $x_i(t)$"
        else:
            y_lbl = y_axes[0]
            title = f"${x_lbl}(t)$"
            ax.set_ylabel(f"$\\mathit{{{y_lbl}}}$", fontsize=13)
        ax.set_title(title, fontsize=14)

        # ---- 3. Рисуем (多Y叠加) ----
        # 颜色按 y_var 分配, 线型按 solution 分配
        n_y = len(y_axes)
        n_sol = len(set(c[0] for c in curves))
        for sol_idx, y_name, xd, yd in curves:
            y_idx = y_axes.index(y_name) if y_name in y_axes else 0
            c = colors[y_idx % len(colors)]
            ls = linestyles[sol_idx % len(linestyles)] if n_sol > 1 else "-"
            label = f"{y_name}" if n_sol == 1 else f"{y_name} #{sol_idx + 1}"
            ax.plot(xd, yd, ls=ls, color=c, lw=lw, label=label)
            if show_markers and is_phase:
                ax.plot(xd[0], yd[0], "o", color=c, ms=7,
                        mec="black", mew=0.5)
                ax.plot(xd[-1], yd[-1], "s", color=c, ms=7,
                        mec="black", mew=0.5)

        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        if grid:
            ax.grid(True, alpha=0.3, ls="--")

        # 相图交叉检测 (只取每条solution的第一个y_var)
        if is_phase and n_sol >= 2:
            phase_curves = [(c[0], c[2], c[3]) for c in curves if c[1] == y_axes[0]]
            if len(phase_curves) >= 2:
                self._draw_phase_with_crossing(phase_curves, ax, colors)

        # S/F 标记 (相图模式)
        if is_phase:
            sf_offsets = [(10, 8), (-14, -12), (10, -14), (-14, 8)]
            ff_offsets = [(10, -12), (-14, 8), (10, 10), (-14, -14)]
            seen_sol = set()
            for sol_idx, y_name, xd, yd in curves:
                if y_name != y_axes[0] or sol_idx in seen_sol:
                    continue
                seen_sol.add(sol_idx)
                y_idx = y_axes.index(y_name) if y_name in y_axes else 0
                c = colors[y_idx % len(colors)]
                ax.plot(xd[0], yd[0], "o", color=c, ms=7, mec="black", mew=0.5)
                ax.plot(xd[-1], yd[-1], "s", color=c, ms=7, mec="black", mew=0.5)
                idx = len(seen_sol) - 1
                ax.annotate("S", (xd[0], yd[0]), textcoords="offset points",
                           xytext=sf_offsets[idx % 4], fontsize=13,
                           fontweight="bold", color=c)
                ax.annotate("F", (xd[-1], yd[-1]), textcoords="offset points",
                           xytext=ff_offsets[idx % 4], fontsize=13,
                           fontweight="bold", color=c)

        ax.legend(loc="best", fontsize=11, framealpha=0.9)
        ax.tick_params(labelsize=11)

        # ---- 4. Финализация ----
        if is_phase and equal:
            ax.set_aspect("equal")
            x_c, y_c = (x_min + x_max) / 2, (y_min + y_max) / 2
            half = max(x_max - x_min, y_max - y_min) / 2 + max(x_pad, y_pad)
            ax.set_xlim(x_c - half, x_c + half)
            ax.set_ylim(y_c - half, y_c + half)
        else:
            fig.tight_layout()

        try:
            plt.show(block=False)
            plt.pause(0.001)
        except Exception:
            plt.show()

    def _draw_phase_with_crossing(self, curves, ax, colors):
        """
        Стиль методички рис. 26.1: пересечения — пунктир, снаружи — сплошная.
        """
        from scipy.spatial import cKDTree

        def find_crossing_segments(xa, ya, xb, yb, tol=0.08):
            pts_b = np.column_stack([xb, yb])
            tree = cKDTree(pts_b)
            dists, _ = tree.query(np.column_stack([xa, ya]), k=1)
            cross = dists < tol
            segs = []
            in_cross, start = False, 0
            for j in range(len(cross)):
                if cross[j] and not in_cross:
                    in_cross = True
                    if start < j:
                        segs.append((start, j, False))
                    start = j
                elif not cross[j] and in_cross:
                    in_cross = False
                    segs.append((start, j, True))
                    start = j
            if start < len(cross):
                segs.append((start, len(cross), in_cross))
            return segs

        # Рисуем каждую кривую с учётом пересечений
        for idx_a, (i_a, xa, ya) in enumerate(curves):
            segs = []
            for idx_b, (i_b, xb, yb) in enumerate(curves):
                if idx_a == idx_b:
                    continue
                segs_b = find_crossing_segments(xa, ya, xb, yb)
                segs.extend(segs_b)
            # Объединяем все пересечения
            if not segs:
                ax.plot(xa, ya, "-", color=colors[i_a % len(colors)], lw=1.5)
                continue
            # Маркируем точки как "в пересечении" если любая пара говорит да
            is_cross = np.zeros(len(xa), dtype=bool)
            for s, e, cross in segs:
                if cross:
                    is_cross[s:e] = True
            # Разбиваем на сегменты
            final_segs = []
            in_c, start = False, 0
            for j in range(len(is_cross)):
                if is_cross[j] and not in_c:
                    in_c = True
                    if start < j:
                        final_segs.append((start, j, False))
                    start = j
                elif not is_cross[j] and in_c:
                    in_c = False
                    final_segs.append((start, j, True))
                    start = j
            if start < len(is_cross):
                final_segs.append((start, len(is_cross), in_c))
            # Рисуем
            c = colors[i_a % len(colors)]
            for s, e, cross in final_segs:
                ls = "--" if cross else "-"
                if e - s > 1:
                    ax.plot(xa[s:e], ya[s:e], ls=ls, color=c, lw=1.5)

    # ------------------------------------------------------------------
    # Экспорт
    # ------------------------------------------------------------------

    def on_export(self):
        t = self.TRANSLATIONS[self.current_lang]
        if self.last_result is None or not is_result_acceptable(self.last_result):
            QMessageBox.warning(self, "Warning", t["err_no_solution"])
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export", "result.json", "JSON (*.json);;Text (*.txt)"
        )
        if not path:
            return

        result = self.last_result
        var_names = self.eq_editor.get_var_names()

        if path.endswith(".json"):
            export_data = {
                "method": result["method"],
                "iterations": result["iterations"],
                "residual_norm": result["residual_norm"],
                "p_opt": result["p_opt"].tolist(),
                "var_names": var_names,
                "t": result["t"].tolist(),
                "y": {name: result["y"][i].tolist()
                      for i, name in enumerate(var_names)},
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        else:
            with open(path, "w") as f:
                # Header
                header = "# t" + "".join([f"\t{name}" for name in var_names])
                f.write(header + "\n")
                for i in range(len(result["t"])):
                    line = f"{result['t'][i]:.8f}"
                    for j in range(len(var_names)):
                        line += f"\t{result['y'][j, i]:.8f}"
                    f.write(line + "\n")

        self.progress_detail.setText(f"已导出: {path}")

    def on_clear(self):
        t = self.TRANSLATIONS[self.current_lang]
        reply = QMessageBox.question(self, "Confirm", t["confirm_clear"])
        if reply == QMessageBox.Yes:
            self.last_result = None
            self.last_failed_result = None
            self.all_results = []
            self._task_status.clear()
            self._refresh_task_table()
            # 清除绿色样式
            self.progress_bar_label.setStyleSheet("")
            self.progress_bar_label.setText(t["status_ready"])
            self.progress_detail.setText("")

    # ------------------------------------------------------------------
    # Локализация
    # ------------------------------------------------------------------

    def update_language(self, lang: str):
        self.current_lang = lang
        t = self.TRANSLATIONS[lang]
        self.setWindowTitle(t["window_title"])
        self.menu_language.setTitle(t["menu_language"])
        self.menu_theme.setTitle(t["menu_theme"])
        self.action_zh.setText(t["lang_chinese"])
        self.action_ru.setText(t["lang_russian"])
        self.action_bg.setText(t["theme_bg"])
        self.action_frame.setText(t["theme_frame"])

        self.menu_help.setTitle(t["menu_help"])
        self.action_help_doc.setText(t["help_doc"])
        self.action_help_about.setText(t["help_about"])

        # ---- Task library ----
        self.label_task_library.setText(t["task_library"])
        self.task_table.setHorizontalHeaderLabels(
            [t["task_col_num"], t["task_col_name"], t["task_col_status"]]
        )
        self.btn_task_add.setText(t["btn_add"])
        self.btn_task_add.setToolTip(t["btn_add_tip"])
        self.btn_task_delete.setText(t["btn_del"])
        self.btn_task_delete.setToolTip(t["btn_del_tip"])
        self.btn_task_load.setText(t["btn_load"])
        self.btn_task_load.setToolTip(t["btn_load_tip"])
        self.btn_task_save.setText(t["btn_save"])
        self.btn_task_save.setToolTip(t["btn_save_tip"])
        self.btn_task_save_current.setText(t["btn_task_save"])
        self.btn_task_save_current.setToolTip(t["btn_task_save_tip"])
        self.input_task_name.setPlaceholderText(t["task_name_placeholder"])

        # ---- Equation editor ----
        self.eq_editor.group_eq.setTitle(t["tab_equations"])
        self.eq_editor.group_bc.setTitle(t["bc_title"])
        self.eq_editor.group_iv.setTitle(t["iv_title"])
        self.eq_editor.bc_editor.setPlaceholderText(t["bc_placeholder"])
        # Обновляем подписи чекбоксов начальных условий
        for i, chk in enumerate(self.eq_editor.iv_checkboxes):
            chk.setText(t["iv_known"].format(idx=i))
        # Обновляем метку "Размерность системы"
        # (первая метка в layout — обновляем через поиск)
        dim_label = self.eq_editor.layout().itemAt(0).layout().itemAt(0).widget()
        if isinstance(dim_label, QLabel):
            dim_label.setText(t["dim_label"])
        # Обновляем метку "Имена переменных"
        # Находим в layout последний элемент перед stretch
        for i in range(self.eq_editor.layout().count()):
            item = self.eq_editor.layout().itemAt(i)
            if item and item.layout() and item.layout().count() >= 2:
                w0 = item.layout().itemAt(0).widget()
                w1 = item.layout().itemAt(1).widget()
                if isinstance(w0, QLabel) and isinstance(w1, QLineEdit) and w1 == self.eq_editor.input_varnames:
                    w0.setText(t["var_names_label"])

        # ---- Parameters ----
        self.group_params.setTitle(t["tab_params"])
        self.label_T.setText(t["param_T"])
        self.label_guess.setText(t["param_guess"])
        self.label_eps.setText(t["param_eps"])
        self.label_method.setText(t["param_method"])
        self.label_solver.setText(t["param_solver"])
        self.label_steps.setText(t["param_steps"])

        # Обновляем тексты в combo_solver
        current_data = self.combo_solver.currentData()
        self.combo_solver.clear()
        self.combo_solver.addItem(t["solver_cont"], "continuation")
        self.combo_solver.addItem(t["solver_shoot"], "shooting")
        idx = self.combo_solver.findData(current_data)
        if idx >= 0:
            self.combo_solver.setCurrentIndex(idx)

        # ---- Progress ----
        self.group_progress.setTitle(t["progress_title"])

        # ---- Action buttons ----
        self.btn_solve.setText(t["btn_solve"])
        self.btn_plot.setText(t["btn_plot"])
        self.btn_export.setText(t["btn_export"])
        self.btn_clear.setText(t["btn_clear"])

        # (状态栏已移除)

        # ---- 同步进度标签语言 ----
        if self.last_result is not None:
            self.progress_bar_label.setText(t["status_solved"])
        elif self.current_task_idx >= 0 and self._task_status.get(self.current_task_idx) == "error":
            self.progress_bar_label.setText(t["status_error"])
        else:
            self.progress_bar_label.setText(t["status_ready"])

        # ---- 同步更新绘图窗口语言 ----
        if hasattr(self, '_plot_widget') and self._plot_widget is not None:
            try:
                self._plot_widget.set_lang(lang)
            except RuntimeError:
                self._plot_widget = None


# ---------------------------------------------------------------------------
# 8. Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = BvpSolverApp()
    window.show()
    sys.exit(app.exec_())
