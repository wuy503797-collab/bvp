"""Public no-window solve entry point backed by the current legacy solver."""

from __future__ import annotations

from typing import Any

from .adapters import dataset_kwargs
from .models import BVPProblem, BVPValidationError, SolverConfig
from .results import BVPResult


def _legacy_solver_components() -> tuple[type, type, type]:
    """Load the existing implementation only when an actual solve is requested."""
    from main import BVPSolver, Dataset, SymPyParser

    return Dataset, SymPyParser, BVPSolver


def solve_bvp_problem(problem: BVPProblem, config: SolverConfig) -> BVPResult:
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

    Dataset, SymPyParser, BVPSolver = _legacy_solver_components()
    dataset = Dataset(**dataset_kwargs(problem, config))
    try:
        parser = SymPyParser(dataset.equations, dataset.var_names)
        parser.lambdify_all()
        solver = BVPSolver(dataset, parser)
    except (TypeError, ValueError, SyntaxError) as exc:
        raise BVPValidationError(
            f"Problem expression validation failed: {type(exc).__name__}: {exc}"
        ) from exc

    legacy_result: dict[str, Any] = solver.solve()
    return BVPResult.from_legacy_dict(legacy_result)
