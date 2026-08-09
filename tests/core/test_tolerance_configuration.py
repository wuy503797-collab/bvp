"""Contracts for explicit tolerances and legacy ``eps`` compatibility."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from bvp_core import BVPProblem, BVPValidationError, SolveRequest, SolverConfig
from bvp_core.adapters import config_from_dataset, dataset_kwargs
from main import Dataset


EXPLICIT_VALUES = {
    "ivp_rtol": 2e-8,
    "ivp_atol": 3e-9,
    "root_tol": 4e-8,
    "least_squares_ftol": 5e-8,
    "least_squares_xtol": 6e-8,
    "least_squares_gtol": 7e-8,
    "continuation_residual_tol": 8e-8,
    "jacobian_relative_step": 9e-5,
}


def _problem() -> BVPProblem:
    return BVPProblem(
        name="Tolerance snapshot",
        odes=["x"],
        var_names=["x"],
        boundary_conditions=["x0_T - E"],
        known_indices=[],
        unknown_indices=[0],
        known_values={},
        initial_guess=[0.5],
    )


def test_legacy_eps_maps_to_every_effective_process_tolerance() -> None:
    config = SolverConfig(eps=1e-6)
    resolved = config.effective_tolerances(boundary_count=2)

    assert config.tolerance_mode == "legacy"
    assert resolved.ivp_rtol == 1e-6
    assert resolved.ivp_atol == 1e-7
    assert resolved.root_tol == 1e-6
    assert resolved.least_squares_ftol == 1e-6
    assert resolved.least_squares_xtol == 1e-6
    assert resolved.least_squares_gtol == 1e-6
    assert resolved.continuation_residual_tol == 1e-6
    assert resolved.jacobian_relative_step == pytest.approx(1e-3)
    assert resolved.jacobian_ivp_rtol == 1e-6
    assert resolved.jacobian_ivp_atol == 1e-8
    assert resolved.boundary_scales == (1.0, 1.0)
    assert set(resolved.sources.values()) <= {"legacy_eps", "explicit", "default"}


def test_partial_explicit_values_override_only_their_own_process() -> None:
    config = SolverConfig(eps=1e-6, ivp_rtol=1e-8, root_tol=2e-7)

    assert config.tolerance_mode == "explicit"
    assert config.ivp_rtol == 1e-8
    assert config.ivp_atol == 1e-7
    assert config.root_tol == 2e-7
    assert config.least_squares_ftol == 1e-6
    assert config.continuation_residual_tol == 1e-6
    assert config.tolerance_sources["ivp_rtol"] == "explicit"
    assert config.tolerance_sources["ivp_atol"] == "legacy_eps"


def test_fully_explicit_configuration_can_omit_eps() -> None:
    config = SolverConfig(eps=None, **EXPLICIT_VALUES)
    resolved = config.effective_tolerances(boundary_count=1)

    assert config.tolerance_mode == "explicit"
    assert resolved.legacy_eps is None
    for name, value in EXPLICIT_VALUES.items():
        assert getattr(resolved, name) == value
        assert resolved.sources[name] == "explicit"


def test_eps_none_requires_every_process_tolerance() -> None:
    incomplete = dict(EXPLICIT_VALUES)
    incomplete.pop("least_squares_gtol")

    with pytest.raises(BVPValidationError, match="least_squares_gtol"):
        SolverConfig(eps=None, **incomplete)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ivp_rtol", 0.0),
        ("ivp_atol", -1.0),
        ("root_tol", np.nan),
        ("least_squares_ftol", np.inf),
        ("least_squares_xtol", 0.0),
        ("least_squares_gtol", 0.0),
        ("continuation_residual_tol", -1e-8),
        ("jacobian_relative_step", np.nan),
    ],
)
def test_invalid_tolerances_name_the_field_and_actual_value(field, value) -> None:
    with pytest.raises(BVPValidationError) as captured:
        SolverConfig(**{field: value})

    message = str(captured.value)
    assert field in message
    assert "positive finite" in message


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("eps", "not-a-number"),
        ("ivp_rtol", "not-a-number"),
        ("boundary_atol", "not-a-number"),
        ("boundary_rtol", "not-a-number"),
        ("boundary_scales", ["not-a-number"]),
    ],
)
def test_non_numeric_tolerances_name_the_field_and_actual_value(field, value) -> None:
    expected_name = "legacy eps" if field == "eps" else field
    with pytest.raises(BVPValidationError) as captured:
        SolverConfig(**{field: value})

    message = str(captured.value)
    assert expected_name in message
    assert "actual=" in message


def test_solver_config_and_request_tolerance_snapshot_are_immutable() -> None:
    config = SolverConfig(eps=1e-6, ivp_rtol=2e-8)
    request = SolveRequest.create(
        problem=_problem(), config=config, source_task_id="tolerance-task"
    )

    with pytest.raises(FrozenInstanceError):
        config.root_tol = 1e-2
    assert request.config.ivp_rtol == 2e-8
    assert request.config.root_tol == 1e-6


def test_legacy_and_explicit_dataset_json_round_trip_preserves_inputs() -> None:
    legacy = Dataset.from_dict({"eps": 1e-6})
    assert legacy.eps == 1e-6
    assert legacy.ivp_rtol is None
    assert config_from_dataset(legacy).tolerance_mode == "legacy"

    explicit_data = {
        "eps": None,
        **EXPLICIT_VALUES,
        "boundary_atol": 1e-8,
        "boundary_rtol": 1e-3,
        "boundary_scales": [1000.0],
    }
    explicit = Dataset.from_dict(explicit_data)
    serialized = explicit.to_dict()
    config = config_from_dataset(explicit)

    for name, value in EXPLICIT_VALUES.items():
        assert serialized[name] == value
        assert getattr(config, name) == value
    assert serialized["eps"] is None
    assert serialized["boundary_scales"] == [1000.0]
    rebuilt = Dataset(**dataset_kwargs(_problem(), config))
    assert rebuilt.eps is None
    assert rebuilt.ivp_rtol == EXPLICIT_VALUES["ivp_rtol"]
    assert rebuilt.boundary_scales == [1000.0]


def test_dataset_mutation_does_not_change_request_config_snapshot() -> None:
    dataset = Dataset(eps=1e-6, ivp_rtol=2e-8)
    config = config_from_dataset(dataset)
    request = SolveRequest.create(
        problem=_problem(), config=config, source_task_id="snapshot"
    )

    dataset.eps = 1e-2
    dataset.ivp_rtol = 3e-2

    assert request.config.eps == 1e-6
    assert request.config.ivp_rtol == 2e-8
