"""Offscreen tests for GUI request provenance and worker lifecycle."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtWidgets import QApplication, QMessageBox

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from bvp_core import (
    BVPProblem,
    BVPResult,
    CancellationToken,
    GuiSolveState,
    RunContext,
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
        solver_metadata={
            "tolerance_mode": "legacy",
            "legacy_eps": 1e-8,
            "effective_ivp_rtol": 1e-8,
            "effective_ivp_atol": 1e-9,
            "effective_root_tol": 1e-8,
            "effective_least_squares_ftol": 1e-8,
            "effective_least_squares_xtol": 1e-8,
            "effective_least_squares_gtol": 1e-8,
            "effective_continuation_residual_tol": 1e-8,
            "effective_jacobian_relative_step": 1e-4,
        },
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


def test_solver_selector_exposes_three_stable_method_identifiers_and_tooltips(
    window: BvpSolverApp,
) -> None:
    assert [window.combo_solver.itemData(index) for index in range(3)] == [
        "shooting",
        "continuation",
        "differential_continuation",
    ]

    window.update_language("zh")
    assert "离散 μ_k" in window.combo_solver.itemData(1, Qt.ToolTipRole)
    assert "μ=0" in window.combo_solver.itemData(2, Qt.ToolTipRole)

    window.update_language("ru")
    assert "Newton-коррекцию" in window.combo_solver.itemData(1, Qt.ToolTipRole)
    assert "μ=1" in window.combo_solver.itemData(2, Qt.ToolTipRole)


@pytest.mark.parametrize(
    ("method", "steps_enabled"),
    [
        ("shooting", False),
        ("continuation", True),
        ("differential_continuation", False),
    ],
)
def test_only_discrete_continuation_enables_step_count(
    window: BvpSolverApp, method: str, steps_enabled: bool
) -> None:
    window.combo_solver.setCurrentIndex(window.combo_solver.findData(method))
    assert window.input_steps.isEnabled() is steps_enabled
    assert window.label_steps.isEnabled() is steps_enabled


def test_solver_selection_builds_request_with_exact_core_method_key(
    window: BvpSolverApp,
) -> None:
    dataset = _dataset("GUI differential dispatch")
    window.tasks = [dataset]
    window._task_ids = ["gui-differential"]
    window.current_task_idx = 0
    window._sync_task_to_editor(dataset)

    for method in ("shooting", "continuation", "differential_continuation"):
        window.combo_solver.setCurrentIndex(window.combo_solver.findData(method))
        request = window._build_solve_request()
        assert request.config.method == method


def test_old_continuation_dataset_keeps_discrete_method_mapping(
    window: BvpSolverApp,
) -> None:
    dataset = _dataset("Legacy continuation")
    dataset.solver_method = "continuation"
    window._sync_task_to_editor(dataset)
    assert window.combo_solver.currentData() == "continuation"
    assert window.combo_solver.currentText() == "Дискретное продолжение"
    assert window._sync_editor_to_dataset(dataset).solver_method == "continuation"


def test_language_switch_preserves_differential_algorithm_key(
    window: BvpSolverApp,
) -> None:
    window.combo_solver.setCurrentIndex(
        window.combo_solver.findData("differential_continuation")
    )
    window.update_language("zh")
    assert window.combo_solver.currentData() == "differential_continuation"
    assert window.combo_solver.currentText() == "微分参数延拓"
    window.update_language("ru")
    assert window.combo_solver.currentData() == "differential_continuation"
    assert window.combo_solver.currentText() == "Дифференциальное продолжение"


def test_worker_dispatches_differential_method_through_public_core_api() -> None:
    dataset = _dataset("Worker differential")
    dataset.solver_method = "differential_continuation"
    request = SolveRequest.create(
        problem=problem_from_dataset(dataset),
        config=config_from_dataset(dataset),
        source_task_id="worker-differential",
    )
    worker = SolverWorker(request)
    completed = []
    worker.request_finished.connect(lambda _request_id, outcome: completed.append(outcome))

    worker.run()

    assert len(completed) == 1
    assert completed[0].result.success is True
    assert completed[0].result.method == "differential_continuation"
    assert completed[0].result.solver_metadata["differential_continuation"][
        "mu_reached_1"
    ] is True


def test_gui_26_1_discrete_and_differential_methods_pass_strict_acceptance(
    window: BvpSolverApp,
) -> None:
    task_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "tasks"
        / "26_1_two_body.json"
    )
    dataset = Dataset.from_dict(json.loads(task_path.read_text(encoding="utf-8"))[0])
    window.tasks = [dataset]
    window._task_ids = ["gui-26-1"]
    window.current_task_idx = 0
    window._sync_task_to_editor(dataset)

    residuals = {}
    for method in ("continuation", "differential_continuation"):
        window.combo_solver.setCurrentIndex(window.combo_solver.findData(method))
        request = window._build_solve_request()
        assert request.config.method == method
        worker = SolverWorker(request)
        outcomes = []
        worker.request_finished.connect(
            lambda _request_id, outcome: outcomes.append(outcome)
        )
        worker.run()
        assert len(outcomes) == 1
        result = outcomes[0].result
        assert result.success and result.boundary_success
        assert np.all(np.abs(result.boundary_residual) <= result.boundary_thresholds)
        residuals[method] = result.boundary_residual_norm

    assert residuals["continuation"] == pytest.approx(1.0192e-11, rel=5e-3)
    assert residuals["differential_continuation"] == pytest.approx(
        1.1810e-9, rel=5e-3
    )


def test_success_message_localizes_method_threshold_and_mu_diagnostics(
    window: BvpSolverApp,
) -> None:
    data = _successful_result().to_dict()
    data["method"] = "differential_continuation"
    data["iterations"] = 74
    data["boundary_thresholds"] = np.array([1e-8])
    data["solver_metadata"] = {
        **data["solver_metadata"],
        "differential_continuation": {"rhs_evaluations": 74},
    }
    result = BVPResult.from_legacy_dict(data)

    window.update_language("zh")
    zh = window._format_success_message(result, np.array([1.0]))
    assert "方法: 微分参数延拓" in zh
    assert "边界阈值: [1.e-08]" in zh
    assert "μ-RHS 调用次数: 74" in zh
    assert "迭代次数" not in zh

    window.update_language("ru")
    ru = window._format_success_message(result, np.array([1.0]))
    assert "Метод: Дифференциальное продолжение" in ru
    assert "Порог краевой невязки: [1.e-08]" in ru
    assert "Число вычислений правой части по μ: 74" in ru
    assert "Итераций" not in ru


def test_success_message_distinguishes_old_discrete_continuation(
    window: BvpSolverApp,
) -> None:
    data = _successful_result().to_dict()
    data["method"] = "continuation"
    result = BVPResult.from_legacy_dict(data)
    window.update_language("ru")
    message = window._format_success_message(result, np.array([1.0]))
    assert "Метод: Дискретное продолжение" in message
    assert "Итераций:" in message
    assert "Дифференциальное продолжение" not in message


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


def test_basic_gui_eps_input_builds_legacy_effective_tolerances(window) -> None:
    dataset = _dataset("Legacy GUI")
    dataset.eps = 1e-6
    window.tasks = [dataset]
    window._task_ids = ["legacy-gui"]
    window.current_task_idx = 0
    window._sync_task_to_editor(dataset)

    request = window._build_solve_request()

    assert request.config.tolerance_mode == "legacy"
    assert request.config.ivp_rtol == 1e-6
    assert request.config.ivp_atol == 1e-7
    assert request.config.root_tol == 1e-6


def test_fully_explicit_json_configuration_survives_gui_snapshot(window) -> None:
    raw = _dataset("Explicit GUI").to_dict()
    raw.update(
        {
            "eps": None,
            "ivp_rtol": 2e-8,
            "ivp_atol": 3e-9,
            "root_tol": 4e-8,
            "least_squares_ftol": 5e-8,
            "least_squares_xtol": 6e-8,
            "least_squares_gtol": 7e-8,
            "continuation_residual_tol": 8e-8,
            "jacobian_relative_step": 9e-5,
            "boundary_scales": [10.0],
        }
    )
    dataset = Dataset.from_dict(raw)
    window.tasks = [dataset]
    window._task_ids = ["explicit-gui"]
    window.current_task_idx = 0
    window._sync_task_to_editor(dataset)

    request = window._build_solve_request()

    assert request.config.eps is None
    assert request.config.tolerance_mode == "explicit"
    assert request.config.ivp_rtol == 2e-8
    assert request.config.boundary_scales == (10.0,)


def test_invalid_explicit_tolerance_does_not_start_worker(window, monkeypatch) -> None:
    dataset = _dataset("Invalid tolerance")
    dataset.ivp_rtol = 0.0
    window.tasks = [dataset]
    window._task_ids = ["invalid-tolerance"]
    window.current_task_idx = 0
    window._sync_task_to_editor(dataset)
    warnings = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )

    assert window.on_solve() is False
    assert window.active_worker is None
    assert "ivp_rtol" in " ".join(str(item) for row in warnings for item in row)


def test_incomplete_eps_none_configuration_does_not_fall_back_to_gui_eps(
    window, monkeypatch
) -> None:
    dataset = _dataset("Incomplete explicit tolerance")
    dataset.eps = None
    dataset.ivp_rtol = 1e-8
    window.tasks = [dataset]
    window._task_ids = ["incomplete-explicit"]
    window.current_task_idx = 0
    window._sync_task_to_editor(dataset)
    warnings = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )

    assert window.on_solve() is False
    assert dataset.eps is None
    assert window.active_worker is None
    assert "eps=None requires explicit values" in " ".join(
        str(item) for row in warnings for item in row
    )


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


def test_plot_keeps_numeric_variants_from_the_same_task(window, monkeypatch) -> None:
    first_request = _request("Variant", task_id="stable-task")
    second_request = SolveRequest.create(
        problem=BVPProblem(
            name="Variant",
            odes=["2*x"],
            var_names=["x"],
            boundary_conditions=["x0_T - 2"],
            known_indices=[],
            unknown_indices=[0],
            known_values={},
            initial_guess=[0.75],
            t_end=2.0,
        ),
        config=SolverConfig(
            method="continuation",
            ivp_method="DOP853",
            eps=1e-6,
            continuation_steps=25,
        ),
        source_task_id="stable-task",
        source_task_index=0,
    )
    first = SolveRecord(first_request, _successful_result(-1.0))
    second = SolveRecord(second_request, _successful_result(1.0))
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

    assert captured["records"] == [first, second]
    assert "excluded" not in window.progress_detail.text()


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


def test_failed_result_diagnostic_displays_effective_tolerances_and_thresholds() -> None:
    result = _failed_result().to_dict()
    result["solver_metadata"].update(
        {
            "tolerance_mode": "explicit",
            "effective_ivp_rtol": 2e-8,
            "effective_ivp_atol": 3e-9,
            "effective_root_tol": 4e-8,
        }
    )

    diagnostic = BvpSolverApp._format_failed_result(result)

    assert "tolerance_mode" in diagnostic
    assert "effective_ivp_rtol" in diagnostic
    assert "boundary_thresholds" in diagnostic
    assert "boundary_max_scaled_ratio" in diagnostic


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
    assert captured[0][1].run_id == worker.run_context.run_id
    assert captured[0][1].run_metadata.final_status == "cancelled"


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
    assert captured[0][1].run_id == worker.run_context.run_id
    assert captured[0][1].run_metadata.request_id == request.request_id


def test_gui_success_detail_displays_run_request_status_and_elapsed(window) -> None:
    request = _request("display-run")
    worker = SolverWorker(request)
    captured = []
    worker.request_finished.connect(lambda _request_id, outcome: captured.append(outcome))
    worker.run()
    _activate(window, request)
    window._run_contexts[request.request_id] = worker.run_context

    window._on_solve_done(request.request_id, captured[0])

    detail = window.progress_detail.text()
    assert worker.run_context.run_id[:8] in detail
    assert request.request_id[:8] in detail
    assert "status=success" in detail
    assert "elapsed_seconds=" in detail


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
    assert captured[0][1].run_id == worker.run_context.run_id
    assert captured[0][1].run_metadata.final_status in {
        "expression_validation_failed",
        "internal_error",
    }


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


def test_auxiliary_failure_emits_structured_warning(window, caplog) -> None:
    caplog.set_level(logging.WARNING, logger="bvp_core.events")
    request = _request("aux-event", auxiliary='__import__("os")')
    window._run_contexts[request.request_id] = RunContext.create(
        request.problem,
        request.config,
        request_id=request.request_id,
        source_task_id=request.source_task_id,
    )

    _complete(window, request)

    events = [
        record.solver_event
        for record in caplog.records
        if hasattr(record, "solver_event")
    ]
    auxiliary = [
        event for event in events if event["event_name"] == "auxiliary_output_failed"
    ]
    assert auxiliary
    assert auxiliary[-1]["run_id"] == window._run_contexts[request.request_id].run_id
    assert len(auxiliary[-1]["details"]["reason"]) <= 256


def test_json_export_contains_effective_tolerances_and_boundary_thresholds(
    window, monkeypatch, tmp_path
) -> None:
    record = _complete(window, _request("export-tolerances"))
    export_path = tmp_path / "result.json"
    monkeypatch.setattr(
        "main.QFileDialog.getSaveFileName",
        lambda *_args, **_kwargs: (str(export_path), "JSON (*.json)"),
    )

    window.on_export()

    exported = json.loads(export_path.read_text(encoding="utf-8"))
    assert exported["schema_version"] == "bvp-result-v1"
    assert exported["effective_tolerances"]["tolerance_mode"] == "legacy"
    assert exported["effective_tolerances"]["effective_ivp_rtol"] == 1e-8
    boundary = exported["diagnostics"]["boundary_acceptance"]
    assert boundary["thresholds"] == [1e-8]
    assert boundary["max_scaled_ratio"] == 0.0
    assert exported["problem"]["name"] == record.request.task_name


def test_json_export_uses_record_snapshot_and_is_byte_repeatable(
    window, monkeypatch, tmp_path
) -> None:
    record = _complete(window, _request("Original snapshot"))
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    paths = iter((str(first), str(second)))
    monkeypatch.setattr(
        "main.QFileDialog.getSaveFileName",
        lambda *_args, **_kwargs: (next(paths), "JSON (*.json)"),
    )
    window.tasks = [_dataset("Edited current task")]

    window.on_export()
    window.on_export()

    assert first.read_bytes() == second.read_bytes()
    exported = json.loads(first.read_text(encoding="utf-8"))
    assert exported["problem"]["name"] == "Original snapshot"
    assert exported["provenance"]["request_id"] == record.request_id


def test_text_export_uses_same_schema_and_key_diagnostics(
    window, monkeypatch, tmp_path
) -> None:
    record = _complete(window, _request("Text snapshot"))
    export_path = tmp_path / "result.txt"
    monkeypatch.setattr(
        "main.QFileDialog.getSaveFileName",
        lambda *_args, **_kwargs: (str(export_path), "Text (*.txt)"),
    )

    window.on_export()

    text = export_path.read_text(encoding="utf-8")
    assert "schema_version: bvp-result-v1" in text
    assert f"request_id: {record.request_id}" in text
    assert "name: Text snapshot" in text
    assert "effective_tolerances:" in text
    assert "boundary_thresholds:" in text


def test_stale_and_duplicate_signals_emit_structured_events(
    window, caplog
) -> None:
    caplog.set_level(logging.WARNING, logger="bvp_core.events")
    old_request = _request("event-old")
    active_request = _request("event-active")
    _activate(window, active_request)
    window._run_contexts[old_request.request_id] = RunContext.create(
        old_request.problem,
        old_request.config,
        request_id=old_request.request_id,
        source_task_id=old_request.source_task_id,
    )
    outcome = SolveOutcome(
        request=old_request,
        status=SolveOutcomeStatus.COMPLETED,
        result=_successful_result(),
    )

    window._on_solve_done(old_request.request_id, outcome)
    window._on_solve_done(old_request.request_id, outcome)

    names = {
        record.solver_event["event_name"]
        for record in caplog.records
        if hasattr(record, "solver_event")
    }
    assert "stale_signal_ignored" in names
    assert "duplicate_signal_ignored" in names


def test_export_emits_structured_start_and_success_without_path(
    window, monkeypatch, tmp_path, caplog
) -> None:
    caplog.set_level(logging.INFO, logger="bvp_core.events")
    record = _complete(window, _request("export-events"))
    context = RunContext.create(
        record.request.problem,
        record.request.config,
        request_id=record.request_id,
        source_task_id=record.request.source_task_id,
    )
    window._run_contexts[record.request_id] = context
    export_path = tmp_path / "result.json"
    monkeypatch.setattr(
        "main.QFileDialog.getSaveFileName",
        lambda *_args, **_kwargs: (str(export_path), "JSON (*.json)"),
    )

    window.on_export()

    events = [
        record.solver_event
        for record in caplog.records
        if hasattr(record, "solver_event") and record.run_id == context.run_id
    ]
    assert [event["event_name"] for event in events] == [
        "export_started",
        "export_succeeded",
    ]
    assert str(export_path) not in json.dumps(events)


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
