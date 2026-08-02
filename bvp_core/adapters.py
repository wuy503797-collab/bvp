"""Explicit adapters between legacy Dataset objects and public core models."""

from __future__ import annotations

from typing import Any

from .models import BVPProblem, BVPValidationError, SolverConfig


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
    """Convert legacy solver controls without changing current eps semantics."""
    return SolverConfig(
        method=dataset.solver_method,
        ivp_method=dataset.method,
        eps=dataset.eps,
        boundary_atol=dataset.boundary_atol,
        boundary_rtol=dataset.boundary_rtol,
        continuation_steps=dataset.continuation_steps,
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
        "method": config.ivp_method,
        "solver_method": config.method,
        "continuation_steps": config.continuation_steps,
        "known_indices": list(problem.known_indices),
        "unknown_indices": list(problem.unknown_indices),
        "t_star": problem.t_start,
        "aux_outputs": dict(problem.auxiliary_expressions),
    }
