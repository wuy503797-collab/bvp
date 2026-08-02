"""Public, GUI-independent models and API for the BVP project."""

from .api import solve_bvp_problem
from .models import BVPProblem, BVPValidationError, SolverConfig
from .results import BVPResult

__all__ = [
    "BVPProblem",
    "BVPResult",
    "BVPValidationError",
    "SolverConfig",
    "solve_bvp_problem",
]
