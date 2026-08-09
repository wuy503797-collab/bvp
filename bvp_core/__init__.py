"""Public, GUI-independent models and API for the BVP project."""

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

from .api import solve_bvp_problem
from .exceptions import ExpressionValidationError, IVPIntegrationError
from .expression_policy import ExpressionContext
from .expressions import SymPyParser
from .models import BVPProblem, BVPValidationError, SolverConfig
from .performance import SolverCounters, SolverCounterSnapshot
from .observability import (
    RunContext,
    RunMetadata,
    SolverEvent,
    emit_solver_event,
    stable_problem_signature,
)
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
from .serialization import (
    EXPORT_SCHEMA_VERSION,
    CanonicalExportRecord,
    atomic_write_text,
    build_canonical_export_record,
    canonical_json_text,
    canonical_text_summary,
    export_canonical_record,
    normalize_json_value,
    write_canonical_json,
    write_canonical_text,
)
from .solver import BVPSolver
from .tolerances import BoundaryAcceptance, ResolvedTolerances

__all__ = [
    "BVPProblem",
    "BVPResult",
    "BVPSolver",
    "BVPValidationError",
    "CancellationToken",
    "CanonicalExportRecord",
    "BoundaryAcceptance",
    "ExpressionContext",
    "ExpressionValidationError",
    "EXPORT_SCHEMA_VERSION",
    "GuiSolveState",
    "IVPIntegrationError",
    "PlotCompatibility",
    "ResolvedTolerances",
    "RunContext",
    "RunMetadata",
    "SolveCancelled",
    "SolveOutcome",
    "SolveOutcomeStatus",
    "SolveRecord",
    "SolveRequest",
    "SolverConfig",
    "SolverCounters",
    "SolverCounterSnapshot",
    "SolverEvent",
    "SymPyParser",
    "gui_transition_allowed",
    "emit_solver_event",
    "atomic_write_text",
    "build_canonical_export_record",
    "canonical_json_text",
    "canonical_text_summary",
    "export_canonical_record",
    "partition_plot_records",
    "plot_compatibility",
    "normalize_json_value",
    "solve_bvp_problem",
    "stable_problem_signature",
    "write_canonical_json",
    "write_canonical_text",
]
