"""Public, GUI-independent models and API for the BVP project."""

from .api import solve_bvp_problem
from .exceptions import ExpressionValidationError, IVPIntegrationError
from .expression_policy import ExpressionContext
from .expressions import SymPyParser
from .models import BVPProblem, BVPValidationError, SolverConfig
from .requests import (
    CancellationToken,
    GuiSolveState,
    PlotCompatibility,
    SolveCancelled,
    SolveOutcome,
    SolveOutcomeStatus,
    SolveRecord,
    SolveRequest,
    gui_transition_allowed,
    partition_plot_records,
    plot_compatibility,
)
from .results import BVPResult
from .solver import BVPSolver

__all__ = [
    "BVPProblem",
    "BVPResult",
    "BVPSolver",
    "BVPValidationError",
    "CancellationToken",
    "ExpressionContext",
    "ExpressionValidationError",
    "GuiSolveState",
    "IVPIntegrationError",
    "PlotCompatibility",
    "SolveCancelled",
    "SolveOutcome",
    "SolveOutcomeStatus",
    "SolveRecord",
    "SolveRequest",
    "SolverConfig",
    "SymPyParser",
    "gui_transition_allowed",
    "partition_plot_records",
    "plot_compatibility",
    "solve_bvp_problem",
]
