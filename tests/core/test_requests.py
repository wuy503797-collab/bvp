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
    partition_plot_records,
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


def _request(
    *,
    source_task_id: str = "task-a",
    config: SolverConfig | None = None,
    **problem_overrides,
) -> SolveRequest:
    return SolveRequest.create(
        problem=_problem(**problem_overrides),
        config=config or SolverConfig(method="shooting"),
        source_task_id=source_task_id,
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
        request=_request(source_task_id="task-b", odes=["-x"]),
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
        "source tasks or problem definitions differ"
    )
    assert plot_compatibility(anchor, two_state).reason == "state dimensions differ"


def test_plot_compatibility_keeps_branches_from_different_initial_guesses() -> None:
    first = SolveRecord(request=_request(initial_guess=[-0.5]), result=_result())
    second = SolveRecord(request=_request(initial_guess=[0.5]), result=_result())

    assert first.request.problem_signature != second.request.problem_signature
    assert plot_compatibility(first, second).compatible is True
    compatible, rejected = partition_plot_records([first, second], second)
    assert compatible == (first, second)
    assert rejected == ()


@pytest.mark.parametrize(
    "problem_overrides",
    [
        {"initial_guess": [-0.75]},
        {"odes": ["2*x"]},
        {"boundary_conditions": ["x0_T - 3"]},
        {"t_start": -1.0, "t_end": 2.0},
        {"auxiliary_expressions": {"triple": "3*x"}},
    ],
)
def test_plot_compatibility_keeps_all_problem_edits_from_one_source_task(
    problem_overrides,
) -> None:
    anchor = SolveRecord(request=_request(), result=_result())
    candidate = SolveRecord(
        request=_request(**problem_overrides),
        result=_result(),
    )

    assert anchor.request.problem_signature != candidate.request.problem_signature
    assert plot_compatibility(anchor, candidate).compatible is True
    compatible, rejected = partition_plot_records([anchor, candidate], candidate)
    assert compatible == (anchor, candidate)
    assert rejected == ()


@pytest.mark.parametrize(
    "config",
    [
        SolverConfig(method="continuation", continuation_steps=25),
        SolverConfig(method="shooting", ivp_method="DOP853"),
        SolverConfig(method="shooting", eps=1e-6),
        SolverConfig(
            method="shooting",
            boundary_atol=1e-6,
            boundary_rtol=1e-4,
            boundary_scales=[2.0],
        ),
        SolverConfig(
            method="shooting",
            eps=None,
            ivp_rtol=2e-7,
            ivp_atol=3e-8,
            root_tol=4e-7,
            least_squares_ftol=4e-7,
            least_squares_xtol=4e-7,
            least_squares_gtol=4e-7,
            continuation_residual_tol=4e-7,
            jacobian_relative_step=5e-7,
        ),
    ],
)
def test_plot_compatibility_keeps_solver_control_variants(config) -> None:
    anchor = SolveRecord(request=_request(), result=_result())
    candidate = SolveRecord(request=_request(config=config), result=_result())

    assert plot_compatibility(anchor, candidate).compatible is True
    assert partition_plot_records([anchor, candidate], candidate)[0] == (
        anchor,
        candidate,
    )


def test_plot_compatibility_keeps_known_initial_value_variants() -> None:
    def two_state(known_value: float) -> SolveRecord:
        request = SolveRequest.create(
            problem=BVPProblem(
                name="Known-value variant",
                odes=["v", "0"],
                var_names=["x", "v"],
                boundary_conditions=["x0_T - 1"],
                known_indices=[0],
                unknown_indices=[1],
                known_values={0: known_value},
                initial_guess=[0.0],
            ),
            config=SolverConfig(method="shooting"),
            source_task_id="known-value-task",
        )
        return SolveRecord(request=request, result=_result(state_dimension=2))

    first = two_state(0.0)
    second = two_state(2.0)

    assert first.request.problem_signature != second.request.problem_signature
    assert plot_compatibility(first, second).compatible is True


def test_record_freezes_primary_curve_before_dense_solution_can_change() -> None:
    class MutableDenseSolution:
        def __init__(self) -> None:
            self.value = 1.0
            self.t = np.array([0.0, 1.0])
            self.y = np.array([[1.0, 1.0]])

        def sol(self, sample):
            return np.full((1, np.asarray(sample).size), self.value)

    dense = MutableDenseSolution()
    base = _result()
    result = BVPResult(
        **{
            **base.__dict__,
            "raw_solution": dense,
        }
    )
    record = SolveRecord(request=_request(), result=result)
    dense.value = 99.0
    dense.y[:] = 99.0

    assert record.primary_plot_t.flags.writeable is False
    assert record.primary_plot_y.flags.writeable is False
    np.testing.assert_allclose(record.primary_plot_y, 1.0)


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
