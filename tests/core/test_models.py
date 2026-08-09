"""Validation and immutability tests for public core input models."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from bvp_core import BVPProblem, BVPValidationError, SolverConfig


def _problem(**overrides) -> BVPProblem:
    values = {
        "name": "Exponential",
        "odes": ["x"],
        "var_names": ["x"],
        "boundary_conditions": ["x0_T - E"],
        "known_indices": [],
        "unknown_indices": [0],
        "known_values": {},
        "initial_guess": [0.5],
        "t_start": 0.0,
        "t_end": 1.0,
    }
    values.update(overrides)
    return BVPProblem(**values)


def test_problem_normalizes_shapes_and_does_not_share_mutable_defaults() -> None:
    first = _problem()
    second = _problem()

    assert first.state_dimension == 1
    assert first.unknown_count == 1
    assert first.odes == ("x",)
    assert first.unknown_indices == (0,)
    assert first.auxiliary_expressions == {}
    assert first.metadata == {}
    assert first.auxiliary_expressions is not second.auxiliary_expressions
    assert first.metadata is not second.metadata


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"known_indices": [0], "known_values": {0: 0.0}}, "overlap"),
        ({"unknown_indices": [1]}, "out of range"),
        ({"initial_guess": [0.5, 0.6]}, "initial guess count"),
        (
            {"boundary_conditions": ["x0_T - E", "x0_0 - 1"]},
            "boundary condition count",
        ),
    ],
)
def test_problem_rejects_invalid_dimensions_and_indices(overrides, message) -> None:
    with pytest.raises(BVPValidationError, match=message):
        _problem(**overrides)


def test_solver_config_defaults_and_legacy_eps_mapping_are_explicit() -> None:
    config = SolverConfig(eps=1e-6)

    assert config.ivp_rtol == 1e-6
    assert config.ivp_atol == 1e-7
    assert config.root_tol == 1e-6
    assert config.max_newton_iterations == 20
    with pytest.raises(FrozenInstanceError):
        config.eps = 1e-4


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"method": "unknown"}, "solver method"),
        ({"ivp_method": "unknown"}, "unsupported IVP"),
        ({"eps": "not-a-number"}, "must be numeric"),
        ({"boundary_atol": 0.0}, "boundary_atol"),
        ({"continuation_steps": 0}, "continuation_steps"),
    ],
)
def test_solver_config_rejects_invalid_controls(overrides, message) -> None:
    with pytest.raises(BVPValidationError, match=message):
        SolverConfig(**overrides)
