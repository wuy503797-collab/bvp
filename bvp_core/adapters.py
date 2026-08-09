"""Explicit adapters between legacy Dataset objects and public core models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .expressions import SymPyParser
from .models import BVPProblem, BVPValidationError, SolverConfig
from .solver import BVPSolver as CoreBVPSolver


class _RejectedLegacySolver:
    """Preserve legacy construct-then-validate behavior for invalid Datasets."""

    def __init__(self, errors: tuple[str, ...]) -> None:
        self._errors = errors

    def solve(self, callback: Callable[..., Any] | None = None) -> dict[str, Any]:
        del callback
        raise ValueError("; ".join(self._errors))


def problem_from_dataset(dataset: Any) -> BVPProblem:
    """Convert a legacy Dataset-like object into the mathematical problem model."""
    equations = list(dataset.equations)
    initial_values = dict(dataset.initial_values)
    known_indices = list(dataset.known_indices)
    errors: list[str] = []
    if len(initial_values) != len(equations):
        errors.append(
            "Input validation: initial value entry count is "
            f"{len(initial_values)}, but state dimension is {len(equations)}"
        )

    known_values: dict[int, float] = {}
    for index in known_indices:
        if index not in initial_values or initial_values[index] is None:
            errors.append(
                f"Input validation: known initial index {index} has no finite value"
            )
        else:
            known_values[index] = initial_values[index]
    try:
        problem = BVPProblem(
            name=dataset.name,
            odes=equations,
            var_names=list(dataset.var_names),
            boundary_conditions=list(dataset.boundary_conditions),
            known_indices=known_indices,
            unknown_indices=list(dataset.unknown_indices),
            known_values=known_values,
            initial_guess=list(dataset.guess),
            t_start=dataset.t_star,
            t_end=dataset.T,
            auxiliary_expressions=dict(dataset.aux_outputs),
            metadata={"source": "legacy_dataset"},
        )
    except BVPValidationError as exc:
        errors.extend(exc.errors)
        raise BVPValidationError(errors) from exc
    if errors:
        raise BVPValidationError(errors)
    return problem


def config_from_dataset(dataset: Any) -> SolverConfig:
    """Convert legacy and explicit controls through the sole config resolver."""
    return SolverConfig(
        method=dataset.solver_method,
        ivp_method=dataset.method,
        eps=dataset.eps,
        boundary_atol=dataset.boundary_atol,
        boundary_rtol=dataset.boundary_rtol,
        continuation_steps=dataset.continuation_steps,
        ivp_rtol=getattr(dataset, "ivp_rtol", None),
        ivp_atol=getattr(dataset, "ivp_atol", None),
        root_tol=getattr(dataset, "root_tol", None),
        least_squares_ftol=getattr(dataset, "least_squares_ftol", None),
        least_squares_xtol=getattr(dataset, "least_squares_xtol", None),
        least_squares_gtol=getattr(dataset, "least_squares_gtol", None),
        continuation_residual_tol=getattr(
            dataset, "continuation_residual_tol", None
        ),
        jacobian_relative_step=getattr(dataset, "jacobian_relative_step", None),
        boundary_scales=getattr(dataset, "boundary_scales", None),
    )


def dataset_kwargs(problem: BVPProblem, config: SolverConfig) -> dict[str, Any]:
    """Build defensive legacy Dataset constructor arguments for the solver adapter."""
    problem.validate()
    config.validate()
    known = set(problem.known_indices)
    initial_values = {
        index: problem.known_values[index] if index in known else None
        for index in range(problem.state_dimension)
    }
    return {
        "name": problem.name,
        "equations": list(problem.odes),
        "var_names": list(problem.var_names),
        "T": problem.t_end,
        "initial_values": initial_values,
        "boundary_conditions": list(problem.boundary_conditions),
        "guess": list(problem.initial_guess),
        "eps": config.eps,
        "boundary_atol": config.boundary_atol,
        "boundary_rtol": config.boundary_rtol,
        "boundary_scales": (
            None if config.boundary_scales is None else list(config.boundary_scales)
        ),
        "ivp_rtol": config.explicit_tolerance_value("ivp_rtol"),
        "ivp_atol": config.explicit_tolerance_value("ivp_atol"),
        "root_tol": config.explicit_tolerance_value("root_tol"),
        "least_squares_ftol": config.explicit_tolerance_value(
            "least_squares_ftol"
        ),
        "least_squares_xtol": config.explicit_tolerance_value(
            "least_squares_xtol"
        ),
        "least_squares_gtol": config.explicit_tolerance_value(
            "least_squares_gtol"
        ),
        "continuation_residual_tol": config.explicit_tolerance_value(
            "continuation_residual_tol"
        ),
        "jacobian_relative_step": config.explicit_tolerance_value(
            "jacobian_relative_step"
        ),
        "method": config.ivp_method,
        "solver_method": config.method,
        "continuation_steps": config.continuation_steps,
        "known_indices": list(problem.known_indices),
        "unknown_indices": list(problem.unknown_indices),
        "t_star": problem.t_start,
        "aux_outputs": dict(problem.auxiliary_expressions),
    }


def solver_from_dataset(
    dataset: Any,
    parser: SymPyParser | None = None,
    cancellation_check: Callable[[], None] | None = None,
) -> CoreBVPSolver | _RejectedLegacySolver:
    """Build the unique core solver from a legacy GUI/JSON Dataset object."""
    try:
        problem = problem_from_dataset(dataset)
        config = config_from_dataset(dataset)
    except BVPValidationError as exc:
        return _RejectedLegacySolver(exc.errors)
    expression_parser = parser or SymPyParser(
        list(problem.odes), list(problem.var_names)
    )
    return CoreBVPSolver(
        problem,
        config,
        expression_parser,
        cancellation_check=cancellation_check,
    )
