"""Warning and plotting regressions for the stable Qt toolbar setup."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import warnings

import numpy as np
from PyQt5.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from bvp_core import BVPProblem, BVPResult, SolveRecord, SolveRequest, SolverConfig
from main import IntegratedPlotWidget


REPOSITORY = Path(__file__).resolve().parents[2]


def _record(
    *,
    initial_guess: float = 0.0,
    state_value: float = 0.0,
    ode: str = "0",
    boundary_condition: str = "x0_T",
    t_end: float = 1.0,
    config: SolverConfig | None = None,
) -> SolveRecord:
    config = config or SolverConfig(method="shooting")
    request = SolveRequest.create(
        problem=BVPProblem(
            name="Plot smoke",
            odes=[ode],
            var_names=["x"],
            boundary_conditions=[boundary_condition],
            known_indices=[],
            unknown_indices=[0],
            known_values={},
            initial_guess=[initial_guess],
            t_end=t_end,
        ),
        config=config,
        source_task_id="plot-smoke",
    )
    time = np.array([0.0, t_end])
    state = np.array([[state_value, state_value]])
    raw_solution = SimpleNamespace(
        t=time,
        y=state,
        sol=lambda sample: np.full(
            (1, np.asarray(sample).size), state_value, dtype=float
        ),
        success=True,
    )
    result = BVPResult(
        success=True,
        status="success",
        message="accepted",
        method=config.method,
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
    return SolveRecord(request=request, result=result)


def test_import_main_is_clean_when_warnings_are_errors() -> None:
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    completed = subprocess.run(
        [sys.executable, "-W", "error", "-c", "import main"],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_plot_widget_uses_stable_toolbar_without_toolmanager_warning() -> None:
    application = QApplication.instance() or QApplication([])
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        widget = IntegratedPlotWidget([_record()])
        widget._refresh_plot()

    try:
        assert widget.toolbar is not None
        assert widget.canvas.figure.axes
        assert not any("experimental" in str(item.message).lower() for item in captured)
    finally:
        widget.close()
        widget.deleteLater()
        application.processEvents()


def test_repository_does_not_hide_warning_categories_globally() -> None:
    main_source = (REPOSITORY / "main.py").read_text(encoding="utf-8")
    assert 'rcParams["toolbar"] = "toolmanager"' not in main_source
    assert 'filterwarnings("ignore")' not in main_source
    assert 'simplefilter("ignore")' not in main_source


def test_plot_widget_keeps_different_guess_branches_checked_and_distinct() -> None:
    application = QApplication.instance() or QApplication([])
    widget = IntegratedPlotWidget(
        [
            _record(initial_guess=-0.5, state_value=-1.0),
            _record(initial_guess=0.5, state_value=1.0),
        ]
    )

    try:
        assert widget.sol_list.count() == 2
        assert all(
            widget.sol_list.item(index).checkState() != 0 for index in range(2)
        )
        labels = [widget.sol_list.item(index).text() for index in range(2)]
        assert "guess=[-0.5]" in labels[0]
        assert "guess=[0.5]" in labels[1]
        branch_lines = [
            line
            for line in widget.fig.axes[0].lines
            if line.get_label().startswith("x #")
        ]
        assert len(branch_lines) == 2
        assert not np.array_equal(
            branch_lines[0].get_ydata(), branch_lines[1].get_ydata()
        )
    finally:
        widget.close()
        widget.deleteLater()
        application.processEvents()


def test_plot_widget_keeps_all_numeric_input_variants_checked_and_distinct() -> None:
    application = QApplication.instance() or QApplication([])
    records = [
        _record(initial_guess=-0.5, state_value=-3.0),
        _record(initial_guess=0.5, state_value=-2.0),
        _record(ode="2*x", state_value=-1.0),
        _record(boundary_condition="x0_T - 2", state_value=0.0),
        _record(t_end=2.0, state_value=1.0),
        _record(
            config=SolverConfig(method="shooting", ivp_method="DOP853", eps=1e-6),
            state_value=2.0,
        ),
        _record(
            config=SolverConfig(method="continuation", continuation_steps=25),
            state_value=3.0,
        ),
    ]
    widget = IntegratedPlotWidget(records)

    try:
        assert widget.sol_list.count() == len(records)
        assert all(
            widget.sol_list.item(index).checkState() != 0
            for index in range(len(records))
        )
        variant_lines = [
            line
            for line in widget.fig.axes[0].lines
            if line.get_label().startswith("x #")
        ]
        assert len(variant_lines) == len(records)
        assert {
            float(line.get_ydata()[0]) for line in variant_lines
        } == {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0}
    finally:
        widget.close()
        widget.deleteLater()
        application.processEvents()


def test_plot_widget_uses_frozen_curve_after_dense_solution_object_changes() -> None:
    application = QApplication.instance() or QApplication([])
    record = _record(initial_guess=-0.5, state_value=-1.0)
    record.result.raw_solution.sol = lambda sample: np.full(
        (1, np.asarray(sample).size), 99.0
    )
    record.result.raw_solution.y[:] = 99.0
    widget = IntegratedPlotWidget([record])

    try:
        solution_lines = [
            line for line in widget.fig.axes[0].lines if line.get_label() == "x"
        ]
        assert len(solution_lines) == 1
        np.testing.assert_allclose(solution_lines[0].get_ydata(), -1.0)
    finally:
        widget.close()
        widget.deleteLater()
        application.processEvents()
