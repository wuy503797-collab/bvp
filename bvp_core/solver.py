"""Headless numerical implementation of shooting and parameter continuation."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numpy.linalg import norm, solve as np_solve
from scipy.integrate import solve_ivp
from scipy.optimize import root

from .exceptions import IVPIntegrationError
from .expressions import SymPyParser
from .models import BVPProblem, SolverConfig
from .observability import RunContext, emit_solver_event, solver_event_enabled
from .performance import SolverCounters
from .requests import SolveCancelled
from .tolerances import ResolvedTolerances, evaluate_boundary_acceptance


@dataclass(frozen=True)
class _SolverInputs:
    """Immutable legacy-shaped view built only from public core models."""

    problem: BVPProblem
    config: SolverConfig
    equations: tuple[str, ...]
    var_names: tuple[str, ...]
    T: float
    initial_values: dict[int, float | None]
    boundary_conditions: tuple[str, ...]
    guess: tuple[float, ...]
    tolerances: ResolvedTolerances
    method: str
    solver_method: str
    continuation_steps: int
    known_indices: tuple[int, ...]
    unknown_indices: tuple[int, ...]
    t_star: float
    aux_outputs: dict[str, str]

    @classmethod
    def from_models(
        cls, problem: BVPProblem, config: SolverConfig
    ) -> "_SolverInputs":
        known = set(problem.known_indices)
        return cls(
            problem=problem,
            config=config,
            equations=problem.odes,
            var_names=problem.var_names,
            T=problem.t_end,
            initial_values={
                index: problem.known_values[index] if index in known else None
                for index in range(problem.state_dimension)
            },
            boundary_conditions=problem.boundary_conditions,
            guess=problem.initial_guess,
            tolerances=config.effective_tolerances(
                boundary_count=len(problem.boundary_conditions)
            ),
            method=config.ivp_method,
            solver_method=config.method,
            continuation_steps=config.continuation_steps,
            known_indices=problem.known_indices,
            unknown_indices=problem.unknown_indices,
            t_star=problem.t_start,
            aux_outputs=dict(problem.auxiliary_expressions),
        )

    def dim(self) -> int:
        return self.problem.state_dimension

    def validate(self) -> list[str]:
        return self.problem.validation_errors() + self.config.validation_errors()


class BVPSolver:
    """
    Универсальный решатель краевых задач.
    Реализует:
      - Метод стрельбы (shooting) через scipy.optimize.root
      - Метод продолжения по параметру (continuation / homotopy)
        см. разд. 7.25–7.26 методички.
    """

    def __init__(
        self,
        problem: BVPProblem,
        config: SolverConfig,
        parser: SymPyParser | None = None,
        cancellation_check: Callable[[], None] | None = None,
        run_context: RunContext | None = None,
    ) -> None:
        if not isinstance(problem, BVPProblem):
            raise TypeError("problem must be a BVPProblem instance")
        if not isinstance(config, SolverConfig):
            raise TypeError("config must be a SolverConfig instance")
        problem.validate()
        config.validate()
        self.problem = problem
        self.config = config
        self.ds = _SolverInputs.from_models(problem, config)
        self.tolerances = self.ds.tolerances
        self.run_context = run_context or RunContext.create(problem, config)
        self.counters = SolverCounters()
        self.core_elapsed_seconds: float | None = None
        self._local_phi_base: tuple[np.ndarray, np.ndarray] | None = None
        self.parser = parser or SymPyParser(list(problem.odes), list(problem.var_names))
        self.bc_residual = SymPyParser.parse_boundary_conditions(
            list(problem.boundary_conditions), list(problem.var_names)
        )
        self._cancellation_check = cancellation_check
        self._build_initial_state_mapper()

    def _check_cancelled(self) -> None:
        if self._cancellation_check is not None:
            self._cancellation_check()

    def performance_metadata(self) -> dict[str, Any]:
        """Return a plain snapshot suitable for result and exception metadata."""
        return {
            "performance_counters": self.counters.snapshot().to_dict(),
            "core_elapsed_seconds": self.core_elapsed_seconds,
        }

    def _set_local_phi_base(self, p: np.ndarray, residual: np.ndarray) -> None:
        """Stage one parameter-bound residual for the immediately next Jacobian."""
        self._local_phi_base = (
            np.asarray(p, dtype=float).copy(),
            np.asarray(residual, dtype=float).copy(),
        )

    def _take_local_phi_base(self, p: np.ndarray) -> np.ndarray | None:
        staged = self._local_phi_base
        self._local_phi_base = None
        if staged is None:
            return None
        parameters, residual = staged
        if not np.array_equal(parameters, np.asarray(p, dtype=float)):
            raise ValueError(
                "local Phi base may only be reused for the identical parameter vector"
            )
        return residual

    def _emit(
        self,
        event_name: str,
        *,
        phase: str,
        status: str | None = None,
        details: dict[str, Any] | None = None,
        level: int = logging.INFO,
    ) -> None:
        if not solver_event_enabled(level):
            return
        emit_solver_event(
            self.run_context,
            event_name,
            phase=phase,
            status=status,
            details=details,
            level=level,
        )

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

    def _solve_ivp(
        self,
        p: np.ndarray,
        t_span: List[float],
        dense_output: bool = False,
        *,
        purpose: str = "solver",
    ) -> Any:
        """Solve the inner IVP and preserve diagnostics for every failure."""
        self._check_cancelled()
        self.counters.record_ivp_start(purpose=purpose)
        self._emit(
            "ivp_started",
            phase="ivp",
            status="running",
            details={"method": self.ds.method, "dense_output": dense_output},
            level=logging.DEBUG,
        )
        x0 = self._p_to_state(p)
        try:
            sol = solve_ivp(
                self.parser.f, t_span, x0,
                method=self.ds.method,
                dense_output=dense_output,
                rtol=self.tolerances.ivp_rtol,
                atol=self.tolerances.ivp_atol,
            )
        except Exception as exc:
            self._emit(
                "ivp_failed",
                phase="ivp",
                status="exception",
                details={"error_type": type(exc).__name__, "message": str(exc)},
                level=logging.WARNING,
            )
            raise IVPIntegrationError(
                f"solve_ivp raised {type(exc).__name__}: {exc}", p=p
            ) from exc
        self.counters.record_ivp_result(sol, purpose=purpose)

        self._check_cancelled()

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
            self._emit(
                "ivp_failed",
                phase="ivp",
                status="failed",
                details={
                    "ivp_status": getattr(sol, "status", None),
                    "ivp_message": getattr(sol, "message", ""),
                    "ivp_t_final": t_final,
                    "state_finite": state_finite,
                },
                level=logging.WARNING,
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
        boundary_count = len(self.ds.boundary_conditions)
        acceptance_residual = (
            residual
            if residual_available and residual.size == boundary_count
            else np.full(boundary_count, np.nan, dtype=float)
        )
        boundary_check = evaluate_boundary_acceptance(
            acceptance_residual,
            boundary_atol=self.tolerances.boundary_atol,
            boundary_rtol=self.tolerances.boundary_rtol,
            boundary_scales=self.tolerances.boundary_scales,
        )
        finite_success = bool(
            finite_success
            and residual.size == boundary_count
            and boundary_check.finite
        )
        boundary_success = bool(
            ivp_success
            and finite_success
            and boundary_check.success
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
                f"{residual_norm:.6e}; maximum scaled component ratio "
                f"{boundary_check.max_scaled_ratio:.6e} <= 1."
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
                "meet component-wise acceptance; maximum scaled ratio="
                f"{boundary_check.max_scaled_ratio:.6e} > 1."
            )
        else:
            status = failure_status or "optimizer_failed"
            message = failure_message or "The BVP result did not pass final acceptance."

        metadata = dict(solver_metadata or {})
        # Effective values are authoritative and cannot be shadowed by
        # algorithm-specific metadata assembled along the solve path.
        metadata.update(self.tolerances.to_metadata())
        metadata.update(self.performance_metadata())
        metadata.update(
            {
                "run_id": self.run_context.run_id,
                "request_id": self.run_context.request_id,
                "problem_signature": self.run_context.problem_signature,
            }
        )
        metadata["boundary_thresholds"] = boundary_check.thresholds.tolist()
        self._emit(
            "final_validation_completed",
            phase="final_validation",
            status=status,
            details={
                "success": success,
                "ivp_success": ivp_success,
                "algorithm_success": algorithm_success,
                "boundary_success": boundary_success,
                "boundary_max_scaled_ratio": boundary_check.max_scaled_ratio,
            },
        )
        if method == "continuation" and not algorithm_success:
            self._emit(
                "continuation_failed",
                phase="continuation",
                status=status,
                details={"failure_reason": failure_message or message},
                level=logging.WARNING,
            )
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
            "boundary_atol": float(self.tolerances.boundary_atol),
            "boundary_rtol": float(self.tolerances.boundary_rtol),
            "boundary_scales": boundary_check.scales,
            "boundary_thresholds": boundary_check.thresholds,
            "boundary_component_success": boundary_check.component_success,
            "boundary_scaled_ratios": boundary_check.scaled_ratios,
            "boundary_max_scaled_ratio": boundary_check.max_scaled_ratio,
            "boundary_acceptance": (
                "abs(residual_i) <= boundary_atol + boundary_rtol * "
                "boundary_scales_i"
            ),
            "solver_metadata": metadata,
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
        self._check_cancelled()
        try:
            sol = self._solve_ivp(
                p,
                [self.ds.t_star, self.ds.T],
                dense_output=True,
                purpose="final_validation",
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

        self._check_cancelled()
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

        self._check_cancelled()
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
        self._check_cancelled()
        self.counters.phi_evaluations += 1
        sol = self._solve_ivp(
            p, [self.ds.t_star, self.ds.T], purpose="phi"
        )
        self._check_cancelled()
        x0_full = self._p_to_state(p)
        xT_full = sol.y[:, -1]
        residual = self.bc_residual(x0_full, xT_full)
        self._check_cancelled()
        return residual

    def _dPhi_dp(self, p: np.ndarray) -> np.ndarray:
        """Evaluate the existing forward-difference Jacobian.

        A solver-private, single-use base residual may be staged with its exact
        parameter snapshot. This keeps reuse local to one Newton iteration and
        prevents a residual from a previous parameter vector being cached by error.
        The perturbation direction and step are unchanged from the former primary
        path; only the unused variational IVP has been removed.
        """
        self._check_cancelled()
        parameters = np.asarray(p, dtype=float)
        base = self._take_local_phi_base(parameters)
        if base is None:
            base = self._Phi(parameters)

        self.counters.jacobian_evaluations += 1
        k = len(self.unknown)
        m = len(base)
        eps_jac = max(1e-7, self.tolerances.jacobian_relative_step)
        jac_matrix = np.zeros((m, k))
        for j in range(k):
            self._check_cancelled()
            p_perturb = parameters.copy()
            p_perturb[j] += eps_jac
            phi_perturb = self._Phi(p_perturb)
            jac_matrix[:, j] = (phi_perturb - base) / eps_jac
        return jac_matrix

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

        self._check_cancelled()
        p0 = np.array(self.ds.guess, dtype=float)

        def root_residual(p):
            self.counters.root_residual_calls += 1
            return self._Phi(p)

        def least_squares_residual(p):
            self.counters.least_squares_residual_calls += 1
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
                root_residual,
                p0,
                method="hybr",
                tol=self.tolerances.root_tol,
                options={"maxfev": 100 * len(p0)},
            )
            self._check_cancelled()
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
            self._emit(
                "root_completed",
                phase="root",
                status="succeeded" if optimizer_success else "failed",
                details=solver_metadata["root"],
            )
        except SolveCancelled:
            raise
        except IVPIntegrationError as exc:
            solver_metadata["root"] = {
                "success": False,
                "message": str(exc),
            }
            self._emit(
                "root_completed",
                phase="root",
                status="ivp_failed",
                details=solver_metadata["root"],
                level=logging.WARNING,
            )
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
            self._emit(
                "root_completed",
                phase="root",
                status="exception",
                details={
                    "error_type": type(exc).__name__,
                    "message": optimizer_message,
                },
                level=logging.WARNING,
            )

        # Попытка 2: least_squares. Нормальное завершение оптимизатора не
        # является критерием выполнения граничных условий.
        if not optimizer_success:
            self._check_cancelled()
            solver_metadata["fallback_used"] = True
            solver_metadata["optimizer"] = "least_squares"
            self._emit(
                "least_squares_started",
                phase="least_squares",
                status="running",
            )
            if callback:
                callback("shooting", 0, "hybr не сошёлся, пробуем least_squares...")
            try:
                ls_result = least_squares(
                    least_squares_residual,
                    p0,
                    ftol=self.tolerances.least_squares_ftol,
                    xtol=self.tolerances.least_squares_xtol,
                    gtol=self.tolerances.least_squares_gtol,
                    max_nfev=5000 * len(p0),
                )
                self._check_cancelled()
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
                    "njev": getattr(ls_result, "njev", None),
                }
                self._emit(
                    "least_squares_completed",
                    phase="least_squares",
                    status="succeeded" if optimizer_success else "failed",
                    details=solver_metadata["least_squares"],
                )
            except SolveCancelled:
                raise
            except IVPIntegrationError as exc:
                solver_metadata["least_squares"] = {
                    "success": False,
                    "message": str(exc),
                }
                self._emit(
                    "least_squares_completed",
                    phase="least_squares",
                    status="ivp_failed",
                    details=solver_metadata["least_squares"],
                    level=logging.WARNING,
                )
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
                self._emit(
                    "least_squares_completed",
                    phase="least_squares",
                    status="exception",
                    details={
                        "error_type": type(exc).__name__,
                        "message": optimizer_message,
                    },
                    level=logging.WARNING,
                )

        self._check_cancelled()
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
        self._check_cancelled()
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
        initial_boundary_check = evaluate_boundary_acceptance(
            Phi_p0,
            boundary_atol=self.tolerances.boundary_atol,
            boundary_rtol=self.tolerances.boundary_rtol,
            boundary_scales=self.tolerances.boundary_scales,
        )
        if initial_boundary_check.success:
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
            self._check_cancelled()
            self.counters.continuation_steps_attempted += 1
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
                self._check_cancelled()
                self.counters.newton_iterations += 1
                try:
                    Phi_current = self._Phi(p)
                except SolveCancelled:
                    raise
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
                if res_norm <= self.tolerances.continuation_residual_tol:
                    step_metadata["newton_converged"] = True
                    break

                try:
                    self._set_local_phi_base(p, Phi_current)
                    try:
                        dPhi = self._dPhi_dp(p)
                    finally:
                        self._local_phi_base = None
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
                except SolveCancelled:
                    raise
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
                    self._check_cancelled()
                    self.counters.damping_trials += 1
                    p_try = p - factor * delta
                    if not np.isfinite(p_try).all():
                        continue
                    try:
                        Phi_try = self._Phi(p_try)
                    except SolveCancelled:
                        raise
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
                        self.counters.newton_updates += 1
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
                        and final_step_norm
                        <= self.tolerances.continuation_residual_tol
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
            self.counters.continuation_steps_completed += 1
            self._emit(
                "continuation_step_completed",
                phase="continuation",
                status=(
                    "converged" if step_metadata["newton_converged"] else "failed"
                ),
                details={
                    "step": step,
                    "mu": mu,
                    "newton_updates": step_metadata["newton_updates"],
                    "residual_norm": step_metadata["residual_norm"],
                },
                level=logging.DEBUG,
            )
            if callback and step % max(1, N // 10) == 0:
                progress = int(100 * step / N)
                callback(
                    "continuation",
                    progress,
                    f"Шаг {step}/{N}: ‖Φ-target‖="
                    f"{step_metadata['residual_norm']:.4e}",
                )

        self._check_cancelled()
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
        """Dispatch to the method selected by the immutable core configuration."""
        started = time.perf_counter()
        result: dict | None = None
        try:
            self._check_cancelled()
            errors = self.ds.validate()
            if errors:
                raise ValueError("; ".join(errors))

            if self.ds.solver_method == "shooting":
                result = self.solve_shooting(callback)
            else:
                result = self.solve_continuation(callback)
            return result
        finally:
            self.core_elapsed_seconds = time.perf_counter() - started
            if result is not None:
                metadata = result.setdefault("solver_metadata", {})
                metadata.update(self.performance_metadata())
