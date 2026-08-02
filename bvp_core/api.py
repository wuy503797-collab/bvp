"""Public no-window solve entry point backed only by the headless core."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .expressions import SymPyParser
from .models import BVPProblem, BVPValidationError, SolverConfig
from .results import BVPResult
from .solver import BVPSolver


def solve_bvp_problem(
    problem: BVPProblem,
    config: SolverConfig,
    *,
    cancellation_check: Callable[[], None] | None = None,
    callback: Callable[[str, int, str], None] | None = None,
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
    problem.validate()
    config.validate()

    if cancellation_check is not None:
        cancellation_check()
    try:
        parser = SymPyParser(list(problem.odes), list(problem.var_names))
        parser.lambdify_all()
        if cancellation_check is not None:
            cancellation_check()
        solver = BVPSolver(
            problem,
            config,
            parser,
            cancellation_check=cancellation_check,
        )
    except (TypeError, ValueError, SyntaxError) as exc:
        raise BVPValidationError(
            f"Problem expression validation failed: {type(exc).__name__}: {exc}"
        ) from exc

    legacy_result: dict[str, Any] = solver.solve(callback=callback)
    return BVPResult.from_legacy_dict(legacy_result)
