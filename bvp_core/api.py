"""Public no-window solve entry point backed only by the headless core."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from .exceptions import ExpressionValidationError
from .expressions import SymPyParser
from .models import BVPProblem, BVPValidationError, SolverConfig
from .observability import (
    RunContext,
    attach_run_metadata,
    build_run_metadata,
    emit_solver_event,
    stable_problem_signature,
)
from .requests import SolveCancelled
from .results import BVPResult
from .solver import BVPSolver


def _solver_metadata_snapshot(
    legacy_result: dict[str, Any] | None,
    solver: BVPSolver | None,
) -> dict[str, Any]:
    metadata = dict((legacy_result or {}).get("solver_metadata", {}))
    if solver is not None:
        metadata.update(solver.performance_metadata())
    return metadata


def solve_bvp_problem(
    problem: BVPProblem,
    config: SolverConfig,
    *,
    cancellation_check: Callable[[], None] | None = None,
    callback: Callable[[str, int, str], None] | None = None,
    run_context: RunContext | None = None,
) -> BVPResult:
    """Solve one BVP without creating QApplication, windows, plots, or files.

    Definition errors raise ``BVPValidationError`` before numerical solving. Expected
    numerical failures such as IVP failure or unacceptable boundary residual return a
    structured ``BVPResult(success=False)`` with the legacy diagnostics preserved.
    """
    if not isinstance(problem, BVPProblem):
        raise BVPValidationError("problem must be a BVPProblem instance")
    if not isinstance(config, SolverConfig):
        raise BVPValidationError("config must be a SolverConfig instance")
    context = run_context or RunContext.create(problem, config)
    if not isinstance(context, RunContext):
        raise BVPValidationError("run_context must be a RunContext instance or None")
    if context.problem_signature != stable_problem_signature(problem):
        raise BVPValidationError("run_context problem_signature does not match problem")
    if context.solver_method != config.method or context.ivp_method != config.ivp_method:
        raise BVPValidationError("run_context solver methods do not match config")

    started_perf = time.perf_counter()
    legacy_result: dict[str, Any] | None = None
    solver: BVPSolver | None = None
    current_phase = "validation"
    emit_solver_event(context, "solve_requested", phase="request", status="accepted")
    try:
        problem.validate()
        config.validate()
        emit_solver_event(context, "solve_started", phase="solve", status="running")
        if cancellation_check is not None:
            cancellation_check()
        emit_solver_event(
            context,
            "expression_validation_started",
            phase="expression_validation",
            status="running",
        )
        current_phase = "expression_validation"
        parser = SymPyParser(list(problem.odes), list(problem.var_names))
        parser.lambdify_all()
        if cancellation_check is not None:
            cancellation_check()
        solver = BVPSolver(
            problem,
            config,
            parser,
            cancellation_check=cancellation_check,
            run_context=context,
        )
        current_phase = "solver"
        emit_solver_event(context, "solver_started", phase="solver", status="running")
        legacy_result = solver.solve(callback=callback)
        elapsed = time.perf_counter() - started_perf
        run_metadata = build_run_metadata(
            context,
            config,
            boundary_count=len(problem.boundary_conditions),
            final_status=str(legacy_result.get("status", "unknown")),
            elapsed_seconds=elapsed,
            solver_metadata=legacy_result.get("solver_metadata", {}),
            result_data=legacy_result,
        )
        result = BVPResult.from_legacy_dict(
            legacy_result, run_metadata=run_metadata
        )
        emit_solver_event(
            context,
            "solve_succeeded" if result.success else "solve_failed",
            phase="solve",
            status=result.status,
            details={"elapsed_seconds": elapsed},
        )
        return result
    except SolveCancelled as exc:
        elapsed = time.perf_counter() - started_perf
        metadata = build_run_metadata(
            context,
            config,
            boundary_count=len(problem.boundary_conditions),
            final_status="cancelled",
            elapsed_seconds=elapsed,
            solver_metadata=_solver_metadata_snapshot(legacy_result, solver),
            result_data=legacy_result,
        )
        attach_run_metadata(exc, metadata)
        emit_solver_event(
            context,
            "solve_cancelled",
            phase="solve",
            status="cancelled",
            details={"elapsed_seconds": elapsed},
        )
        raise
    except ExpressionValidationError as exc:
        elapsed = time.perf_counter() - started_perf
        metadata = build_run_metadata(
            context,
            config,
            boundary_count=len(problem.boundary_conditions),
            final_status="expression_validation_failed",
            elapsed_seconds=elapsed,
        )
        attach_run_metadata(exc, metadata)
        emit_solver_event(
            context,
            "expression_validation_failed",
            level=logging.WARNING,
            phase="expression_validation",
            status="rejected",
            details={
                "context": exc.context,
                "field_index": exc.field_index,
                "error_code": exc.error_code,
                "expression_preview": exc.expression_preview,
            },
        )
        emit_solver_event(
            context,
            "solve_failed",
            level=logging.WARNING,
            phase="solve",
            status="expression_validation_failed",
        )
        raise
    except BVPValidationError as exc:
        elapsed = time.perf_counter() - started_perf
        metadata = build_run_metadata(
            context,
            config,
            boundary_count=len(problem.boundary_conditions),
            final_status="validation_failed",
            elapsed_seconds=elapsed,
        )
        attach_run_metadata(exc, metadata)
        emit_solver_event(
            context,
            "solve_failed",
            level=logging.WARNING,
            phase="validation",
            status="validation_failed",
            details={"error_type": type(exc).__name__, "message": str(exc)},
        )
        raise
    except (TypeError, ValueError, SyntaxError) as exc:
        if current_phase != "expression_validation":
            elapsed = time.perf_counter() - started_perf
            metadata = build_run_metadata(
                context,
                config,
                boundary_count=len(problem.boundary_conditions),
                final_status="internal_error",
                elapsed_seconds=elapsed,
                solver_metadata=_solver_metadata_snapshot(legacy_result, solver),
                result_data=legacy_result,
            )
            attach_run_metadata(exc, metadata)
            emit_solver_event(
                context,
                "solve_failed",
                level=logging.ERROR,
                phase=current_phase,
                status="internal_error",
                details={"error_type": type(exc).__name__, "message": str(exc)},
            )
            raise
        wrapped = BVPValidationError(
            f"Problem expression validation failed: {type(exc).__name__}: {exc}"
        )
        elapsed = time.perf_counter() - started_perf
        metadata = build_run_metadata(
            context,
            config,
            boundary_count=len(problem.boundary_conditions),
            final_status="validation_failed",
            elapsed_seconds=elapsed,
        )
        attach_run_metadata(wrapped, metadata)
        emit_solver_event(
            context,
            "solve_failed",
            level=logging.WARNING,
            phase="expression_validation",
            status="validation_failed",
            details={"error_type": type(exc).__name__, "message": str(exc)},
        )
        raise wrapped from exc
    except Exception as exc:
        elapsed = time.perf_counter() - started_perf
        metadata = build_run_metadata(
            context,
            config,
            boundary_count=len(problem.boundary_conditions),
            final_status="internal_error",
            elapsed_seconds=elapsed,
            solver_metadata=_solver_metadata_snapshot(legacy_result, solver),
            result_data=legacy_result,
        )
        attach_run_metadata(exc, metadata)
        emit_solver_event(
            context,
            "solve_failed",
            level=logging.ERROR,
            phase="internal",
            status="internal_error",
            details={"error_type": type(exc).__name__, "message": str(exc)},
        )
        raise
