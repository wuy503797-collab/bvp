"""Public, GUI-independent models and API for the BVP project."""

from .api import solve_bvp_problem
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

__all__ = [
    "BVPProblem",
    "BVPResult",
    "BVPValidationError",
    "CancellationToken",
    "GuiSolveState",
    "PlotCompatibility",
    "SolveCancelled",
    "SolveOutcome",
    "SolveOutcomeStatus",
    "SolveRecord",
    "SolveRequest",
    "SolverConfig",
    "gui_transition_allowed",
    "partition_plot_records",
    "plot_compatibility",
    "solve_bvp_problem",
]
