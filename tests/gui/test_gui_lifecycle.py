"""Offscreen tests for GUI request provenance and worker lifecycle."""

from __future__ import annotations

import os
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication, QMessageBox

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from bvp_core import (
    BVPProblem,
    BVPResult,
    CancellationToken,
    GuiSolveState,
    SolveOutcome,
    SolveOutcomeStatus,
    SolveRecord,
    SolveRequest,
    SolverConfig,
)
from bvp_core.adapters import config_from_dataset, problem_from_dataset
from main import BvpSolverApp, Dataset, SolverWorker


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, monkeypatch) -> BvpSolverApp:
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)
    target = BvpSolverApp()
    yield target
    if target.active_worker is not None and target.active_worker.isRunning():
        if target.active_cancellation_token is not None:
            target.active_cancellation_token.cancel()
        target.active_worker.wait(1000)
    target.deleteLater()
    qapp.processEvents()


def _dataset(name: str, variable: str = "x", auxiliary: str | None = None) -> Dataset:
    return Dataset(
        name=name,
        equations=["0"],
        var_names=[variable],
        T=1.0,
        initial_values={0: None},
        boundary_conditions=["x0_T"],
        guess=[0.0],
        solver_method="shooting",
        known_indices=[],
        unknown_indices=[0],
        aux_outputs={} if auxiliary is None else {"double": auxiliary},
    )


def _request(
    name: str,
    *,
    variable: str = "x",
    auxiliary: str | None = None,
    task_id: str | None = None,
) -> SolveRequest:
    dataset = _dataset(name, variable, auxiliary)
    return SolveRequest.create(
        problem=problem_from_dataset(dataset),
        config=config_from_dataset(dataset),
        source_task_id=task_id or f"task-{name}",
        source_task_index=0,
    )


def _successful_result(state_value: float = 1.0) -> BVPResult:
    time = np.array([0.0, 1.0])
    state = np.array([[state_value, state_value]])

    def dense(sample_time):
        return np.full((1, np.asarray(sample_time).size), state_value)

    raw_solution = SimpleNamespace(t=time, y=state, sol=dense, success=True)
    return BVPResult(
        success=True,
        status="success",
        message="accepted",
        method="shooting",
        p_opt=np.array([state_value]),
        t=time,
        y=state,
        raw_solution=raw_solution,
        ivp_success=True,
        optimizer_success=True,
        algorithm_success=True,
        finite_success=True,
        boundary_success=True,
        boundary_residual=np.array([0.0]),
        boundary_residual_norm=0.0,
    )


def _failed_result() -> BVPResult:
    return BVPResult(
        success=False,
        status="boundary_residual_too_large",
        message="residual rejected",
        method="shooting",
        p_opt=np.array([0.0]),
        t=np.array([0.0, 1.0]),
        y=np.array([[0.0, 0.0]]),
        ivp_success=True,
        optimizer_success=True,
        algorithm_success=True,
        finite_success=True,
        boundary_success=False,
        boundary_residual=np.array([1.0]),
        boundary_residual_norm=1.0,
    )


def _activate(window: BvpSolverApp, request: SolveRequest) -> None:
    window.active_request_id = request.request_id
    window.active_cancellation_token = CancellationToken()
    window.solve_state = GuiSolveState.RUNNING
    window._render_gui_state()


def _complete(window: BvpSolverApp, request: SolveRequest) -> SolveRecord:
    _activate(window, request)
    outcome = SolveOutcome(
        request=request,
        status=SolveOutcomeStatus.COMPLETED,
        result=_successful_result(),
    )
    window._on_solve_done(request.request_id, outcome)
    return window.solve_records[-1]


def test_request_snapshot_survives_original_dataset_mutation() -> None:
    dataset = _dataset("A", auxiliary="2*x")
    request = SolveRequest.create(
        problem=problem_from_dataset(dataset),
        config=config_from_dataset(dataset),
        source_task_id="task-a",
    )

    dataset.name = "edited"
    dataset.equations[0] = "x"
    dataset.var_names[0] = "changed"
    dataset.guess[0] = 99.0
    dataset.aux_outputs["double"] = "broken"

    assert request.task_name == "A"
    assert request.problem.odes == ("0",)
    assert request.var_names == ("x",)
    assert request.problem.initial_guess == (0.0,)
    assert request.auxiliary_expressions == {"double": "2*x"}


def test_gui_builds_request_from_one_editor_snapshot(window) -> None:
    dataset = _dataset("GUI snapshot", auxiliary="2*x")
    window.tasks = [dataset]
    window._task_ids = ["gui-task"]
    window.current_task_idx = 0
    window._sync_task_to_editor(dataset)

    request = window._build_solve_request()
    window.input_task_name.setText("edited later")
    window.eq_editor.eq_editors[0].setText("x")
    dataset.aux_outputs["double"] = "3*x"

    assert request.source_task_id == "gui-task"
    assert request.task_name == "GUI snapshot"
    assert request.problem.odes == ("0",)
    assert request.auxiliary_expressions == {"double": "2*x"}


def test_dangerous_ode_is_rejected_before_worker_creation(window, monkeypatch) -> None:
    dataset = _dataset("Unsafe ODE")
    window.tasks = [dataset]
    window._task_ids = ["unsafe-ode"]
    window.current_task_idx = 0
    window._sync_task_to_editor(dataset)
    window.eq_editor.eq_editors[0].setText('__import__("os")')
    warnings = []
    worker_created = False

    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )

    def unexpected_worker(*_args, **_kwargs):
        nonlocal worker_created
        worker_created = True
        raise AssertionError("unsafe expressions must be rejected before a worker starts")

    window.worker_factory = unexpected_worker

    assert window.on_solve() is False
    assert worker_created is False
    assert window.active_worker is None
    assert window.solve_state is GuiSolveState.IDLE
    diagnostic = " ".join(str(item) for warning in warnings for item in warning)
    assert "ODE" in diagnostic
    assert "forbidden_name" in diagnostic
    assert "__import__" in diagnostic


def test_dangerous_initial_guess_cannot_bypass_core_parser(window, monkeypatch) -> None:
    dataset = _dataset("Unsafe guess")
    window.tasks = [dataset]
    window._task_ids = ["unsafe-guess"]
    window.current_task_idx = 0
    window._sync_task_to_editor(dataset)
    window.input_guess.setText('__import__("os")')
    warnings = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )

    assert window.on_solve() is False
    assert window.active_worker is None
    diagnostic = " ".join(str(item) for warning in warnings for item in warning)
    assert "SCALAR" in diagnostic
    assert "forbidden_name" in diagnostic


def test_task_switch_does_not_pollute_result_or_auxiliary_snapshot(window) -> None:
    task_a = _dataset("A", auxiliary="2*x")
    task_b = _dataset("B", variable="b", auxiliary="3*b")
    window.tasks = [task_a, task_b]
    window._task_ids = ["task-a", "task-b"]
    request = _request("A", auxiliary="2*x", task_id="task-a")

    window.current_task_idx = 1
    record = _complete(window, request)

    assert record.request.task_name == "A"
    assert record.request.var_names == ("x",)
    assert set(record.auxiliary_outputs) == {"double"}
    np.testing.assert_allclose(record.auxiliary_outputs["double"], 2.0)
    assert window._task_status["task-a"] == "solved"
    assert "task-b" not in window._task_status


def test_editing_or_deleting_task_does_not_change_existing_record(window) -> None:
    task = _dataset("Original", auxiliary="2*x")
    window.tasks = [task]
    window._task_ids = ["task-original"]
    record = _complete(
        window,
        _request("Original", auxiliary="2*x", task_id="task-original"),
    )

    task.name = "Edited"
    task.equations[0] = "x"
    task.var_names[0] = "edited"
    window.tasks.clear()
    window._task_ids.clear()

    assert record.request.task_name == "Original"
    assert record.request.problem.odes == ("0",)
    assert record.request.var_names == ("x",)
    assert window.solve_records == [record]


def test_plot_uses_latest_record_metadata_and_excludes_incompatible_history(
    window, monkeypatch
) -> None:
    first = SolveRecord(_request("A", variable="x"), _successful_result())
    second = SolveRecord(_request("B", variable="temperature"), _successful_result())
    window.solve_records = [first, second]
    window.last_record = second
    captured = {}

    class FakePlotWidget:
        def __init__(self, records, lang, parent):
            captured["records"] = records

        def show(self):
            pass

        def raise_(self):
            pass

        def activateWindow(self):
            pass

    monkeypatch.setattr("main.IntegratedPlotWidget", FakePlotWidget)
    window.on_plot()

    assert captured["records"] == [second]
    assert captured["records"][0].request.var_names == ("temperature",)
    assert "excluded 1" in window.progress_detail.text()


def test_failed_and_cancelled_outcomes_never_enter_success_history(window) -> None:
    failed_request = _request("failed")
    _activate(window, failed_request)
    failed = SolveOutcome(
        request=failed_request,
        status=SolveOutcomeStatus.FAILED,
        result=_failed_result(),
    )
    window._on_solve_error(failed_request.request_id, failed)

    assert window.solve_records == []
    assert window.last_failed_record is failed
    assert window.solve_state is GuiSolveState.FAILED

    cancelled_request = _request("cancelled")
    _activate(window, cancelled_request)
    cancelled = SolveOutcome(
        request=cancelled_request,
        status=SolveOutcomeStatus.CANCELLED,
        message="cancelled at checkpoint",
    )
    window._on_solve_cancelled(cancelled_request.request_id, cancelled)

    assert window.solve_records == []
    assert window.last_cancelled_outcome is cancelled
    assert window.solve_state is GuiSolveState.CANCELLED


def test_cancel_request_wins_at_auxiliary_checkpoint(window) -> None:
    request = _request("cancel-before-aux", auxiliary="2*x")
    _activate(window, request)

    assert window.on_cancel() is True
    assert window.active_cancellation_token.is_cancelled() is True
    assert window.solve_state is GuiSolveState.CANCEL_REQUESTED

    completed = SolveOutcome(
        request=request,
        status=SolveOutcomeStatus.COMPLETED,
        result=_successful_result(),
    )
    window._on_solve_done(request.request_id, completed)

    assert window.solve_records == []
    assert window.solve_state is GuiSolveState.CANCELLED
    assert "before auxiliary" in window.last_cancelled_outcome.message


def test_duplicate_terminal_signal_adds_history_once(window) -> None:
    request = _request("duplicate")
    _activate(window, request)
    outcome = SolveOutcome(
        request=request,
        status=SolveOutcomeStatus.COMPLETED,
        result=_successful_result(),
    )

    window._on_solve_done(request.request_id, outcome)
    window._on_solve_done(request.request_id, outcome)

    assert len(window.solve_records) == 1
    assert "duplicate" in window._last_signal_diagnostic.lower()


def test_stale_worker_signal_does_not_override_active_request(window) -> None:
    old_request = _request("old")
    active_request = _request("active")
    _activate(window, active_request)
    stale_outcome = SolveOutcome(
        request=old_request,
        status=SolveOutcomeStatus.COMPLETED,
        result=_successful_result(),
    )

    window._on_solve_done(old_request.request_id, stale_outcome)

    assert window.solve_records == []
    assert window.active_request_id == active_request.request_id
    assert window.solve_state is GuiSolveState.RUNNING
    assert "stale" in window._last_signal_diagnostic.lower()


def test_repeated_solve_does_not_start_second_worker(window) -> None:
    class RunningWorker:
        def isRunning(self):
            return True

        def wait(self, _timeout):
            return True

    window.active_worker = RunningWorker()
    window.solve_state = GuiSolveState.RUNNING

    assert window.on_solve() is False
    assert "second worker was not started" in window.progress_detail.text()


def test_precancelled_worker_emits_cancelled_with_request_id() -> None:
    request = _request("worker-cancel")
    token = CancellationToken()
    token.cancel()
    worker = SolverWorker(request, token)
    captured = []
    worker.request_cancelled.connect(lambda request_id, outcome: captured.append((request_id, outcome)))

    worker.run()

    assert captured[0][0] == request.request_id
    assert captured[0][1].status is SolveOutcomeStatus.CANCELLED


def test_worker_success_signal_returns_structured_outcome_with_request_id() -> None:
    request = _request("worker-success")
    worker = SolverWorker(request)
    captured = []
    worker.request_finished.connect(
        lambda request_id, outcome: captured.append((request_id, outcome))
    )

    worker.run()

    assert captured[0][0] == request.request_id
    assert captured[0][1].status is SolveOutcomeStatus.COMPLETED
    assert captured[0][1].result.success is True


def test_worker_programming_failure_retains_traceback() -> None:
    request = SolveRequest.create(
        problem=BVPProblem(
            name="Invalid expression",
            odes=["x +"],
            var_names=["x"],
            boundary_conditions=["x0_T"],
            known_indices=[],
            unknown_indices=[0],
            known_values={},
            initial_guess=[0.0],
        ),
        config=SolverConfig(method="shooting"),
        source_task_id="invalid-expression",
    )
    worker = SolverWorker(request)
    captured = []
    worker.request_failed.connect(lambda request_id, outcome: captured.append((request_id, outcome)))

    worker.run()

    assert captured[0][0] == request.request_id
    assert captured[0][1].result is None
    assert "Traceback" in captured[0][1].technical_diagnostic


def test_invalid_auxiliary_expression_keeps_primary_result(window) -> None:
    record = _complete(window, _request("bad-aux", auxiliary="unknown("))

    assert record.result.success is True
    assert record.auxiliary_outputs == {}
    assert len(record.auxiliary_errors) == 1
    assert "double" in record.auxiliary_errors[0]


def test_dangerous_auxiliary_expression_is_isolated_after_primary_solve(window) -> None:
    record = _complete(
        window,
        _request("unsafe-aux", auxiliary='__import__("os")'),
    )

    assert record.result.success is True
    assert record.auxiliary_outputs == {}
    assert len(record.auxiliary_errors) == 1
    assert "forbidden_name" in record.auxiliary_errors[0]


def test_close_event_cooperatively_cancels_and_joins_worker(window) -> None:
    request = _request("close")
    token = CancellationToken()
    started = Event()

    class ControlledWorker(QThread):
        def run(self):
            started.set()
            token.wait()

    worker = ControlledWorker()
    worker.start()
    assert started.wait(1.0)
    window.active_worker = worker
    window.active_request_id = request.request_id
    window.active_cancellation_token = token
    window.solve_state = GuiSolveState.RUNNING
    event = SimpleNamespace(accepted=False, ignored=False)
    event.accept = lambda: setattr(event, "accepted", True)
    event.ignore = lambda: setattr(event, "ignored", True)

    window.closeEvent(event)

    assert token.is_cancelled() is True
    assert worker.isRunning() is False
    assert event.accepted is True
    assert event.ignored is False


def test_close_timeout_keeps_window_alive_without_force_termination(window) -> None:
    token = CancellationToken()

    class NonStoppingWorker:
        def isRunning(self):
            return True

        def wait(self, _timeout):
            return False

    window.active_worker = NonStoppingWorker()
    window.active_request_id = "still-running"
    window.active_cancellation_token = token
    window.solve_state = GuiSolveState.RUNNING
    event = SimpleNamespace(accepted=False, ignored=False)
    event.accept = lambda: setattr(event, "accepted", True)
    event.ignore = lambda: setattr(event, "ignored", True)

    window.closeEvent(event)

    assert token.is_cancelled() is True
    assert event.accepted is False
    assert event.ignored is True
    assert "timed out" in window.progress_detail.text()
    window.active_worker = None
