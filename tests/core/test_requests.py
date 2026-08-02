"""Contract tests for solve snapshots, provenance, cancellation, and plotting."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from bvp_core import (
    BVPProblem,
    BVPResult,
    CancellationToken,
    GuiSolveState,
    SolveCancelled,
    SolveRecord,
    SolveRequest,
    SolverConfig,
    gui_transition_allowed,
    plot_compatibility,
)


def _problem(**overrides) -> BVPProblem:
    values = {
        "name": "Snapshot exponential",
        "odes": ["x"],
        "var_names": ["x"],
        "boundary_conditions": ["x0_T - E"],
        "known_indices": [],
        "unknown_indices": [0],
        "known_values": {},
        "initial_guess": [0.5],
        "auxiliary_expressions": {"double": "2*x"},
    }
    values.update(overrides)
    return BVPProblem(**values)


def _request(**problem_overrides) -> SolveRequest:
    return SolveRequest.create(
        problem=_problem(**problem_overrides),
        config=SolverConfig(method="shooting"),
        source_task_id="task-a",
        source_task_index=0,
    )


def _result(state_dimension: int = 1) -> BVPResult:
    return BVPResult(
        success=True,
        status="success",
        message="accepted",
        method="shooting",
        p_opt=np.array([1.0]),
        t=np.array([0.0, 1.0]),
        y=np.ones((state_dimension, 2)),
        ivp_success=True,
        optimizer_success=True,
        algorithm_success=True,
        finite_success=True,
        boundary_success=True,
        boundary_residual=np.array([0.0]),
        boundary_residual_norm=0.0,
    )


def test_request_ids_are_unique_and_snapshot_fields_are_immutable() -> None:
    first = _request()
    second = _request()

    assert first.request_id != second.request_id
    assert first.task_name == "Snapshot exponential"
    assert first.var_names == ("x",)
    assert first.known_indices == ()
    assert first.unknown_indices == (0,)
    assert first.auxiliary_expressions == {"double": "2*x"}
    with pytest.raises(FrozenInstanceError):
        first.source_task_id = "changed"


def test_request_rejects_gui_or_other_opaque_display_metadata() -> None:
    with pytest.raises(TypeError, match="opaque objects"):
        SolveRequest.create(
            problem=_problem(),
            config=SolverConfig(),
            source_task_id="task-a",
            display_metadata={"widget": object()},
        )


def test_request_builds_initial_state_only_from_its_problem_snapshot() -> None:
    request = SolveRequest.create(
        problem=BVPProblem(
            name="Two state",
            odes=["v", "-x"],
            var_names=["x", "v"],
            boundary_conditions=["x0_T - 1"],
            known_indices=[0],
            unknown_indices=[1],
            known_values={0: 2.0},
            initial_guess=[0.5],
        ),
        config=SolverConfig(),
        source_task_id="task-two",
    )

    np.testing.assert_array_equal(request.build_initial_state([3.0]), [2.0, 3.0])


def test_record_freezes_auxiliary_arrays_and_returns_legacy_copies() -> None:
    values = np.array([2.0, 3.0])
    record = SolveRecord(
        request=_request(),
        result=_result(),
        auxiliary_outputs={"double": values},
        auxiliary_sample_t=np.array([0.0, 1.0]),
    )
    values[0] = 99.0
    legacy = record.to_legacy_dict()
    legacy["aux"]["double"][0] = 88.0

    assert record.auxiliary_outputs["double"].flags.writeable is False
    np.testing.assert_array_equal(record.auxiliary_outputs["double"], [2.0, 3.0])
    assert record.request_id == legacy["request_id"]
    assert legacy["var_names"] == ["x"]


def test_record_rejects_result_with_wrong_state_dimension() -> None:
    with pytest.raises(ValueError, match="state dimension"):
        SolveRecord(request=_request(), result=_result(state_dimension=2))


def test_plot_compatibility_rejects_same_dimension_with_different_semantics() -> None:
    anchor = SolveRecord(request=_request(), result=_result())
    different_name = SolveRecord(
        request=_request(var_names=["temperature"]),
        result=_result(),
    )
    different_equation = SolveRecord(
        request=_request(odes=["-x"]),
        result=_result(),
    )
    two_state_request = SolveRequest.create(
        problem=BVPProblem(
            name="Oscillator",
            odes=["v", "-x"],
            var_names=["x", "v"],
            boundary_conditions=["x0_T - 1"],
            known_indices=[0],
            unknown_indices=[1],
            known_values={0: 0.0},
            initial_guess=[1.0],
        ),
        config=SolverConfig(method="shooting"),
        source_task_id="oscillator",
    )
    two_state = SolveRecord(two_state_request, _result(state_dimension=2))

    assert plot_compatibility(anchor, anchor).compatible is True
    assert plot_compatibility(anchor, different_name).reason == (
        "variable names or meanings differ"
    )
    assert plot_compatibility(anchor, different_equation).reason == (
        "problem equations or boundaries differ"
    )
    assert plot_compatibility(anchor, two_state).reason == "state dimensions differ"


def test_cancellation_token_raises_only_after_cancel_request() -> None:
    token = CancellationToken()
    token.raise_if_cancelled()
    token.cancel()

    assert token.is_cancelled() is True
    with pytest.raises(SolveCancelled):
        token.raise_if_cancelled()


@pytest.mark.parametrize(
    ("current", "target", "allowed"),
    [
        (GuiSolveState.IDLE, GuiSolveState.RUNNING, True),
        (GuiSolveState.RUNNING, GuiSolveState.CANCEL_REQUESTED, True),
        (GuiSolveState.CANCEL_REQUESTED, GuiSolveState.CANCELLED, True),
        (GuiSolveState.IDLE, GuiSolveState.COMPLETED, False),
    ],
)
def test_gui_state_transitions_are_explicit(current, target, allowed) -> None:
    assert gui_transition_allowed(current, target) is allowed
