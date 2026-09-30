"""The GUI plots saved differential-continuation trajectories without solving again."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication, QFileDialog

from bvp_core import (
    BVPProblem, BVPResult, SolveRecord, SolveRequest, SolverConfig,
    solve_bvp_problem,
)
from main import IntegratedPlotWidget
from matplotlib import pyplot as plt
from scripts.run_differential_continuation_validation import load_case


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def close_test_figures():
    yield
    plt.close("all")


def _record(method="differential_continuation", *, trajectory=True, semantic=False):
    names = ["x", "y", "vx", "vy"] if semantic else ["a", "b"]
    unknown = [2, 3] if semantic else [0, 1]
    problem = BVPProblem(
        name="Trajectory test",
        odes=["0"] * len(names),
        var_names=names,
        boundary_conditions=["x0_T", "x1_T"],
        known_indices=[0, 1] if semantic else [],
        unknown_indices=unknown,
        known_values={0: 2.0, 1: 0.0} if semantic else {},
        initial_guess=[0.5, -0.5],
    )
    request = SolveRequest.create(
        problem=problem, config=SolverConfig(method=method), source_task_id="plot-trajectory"
    )
    mu = np.linspace(0.0, 1.0, 201)
    path = np.vstack((0.5 - 0.05 * mu**2, -0.5 + 0.2 * np.sin(mu)))
    metadata = (
        {"differential_continuation": {"trajectory": {
            "mu": mu.tolist(), "p": path.tolist(),
        }}}
        if trajectory else {}
    )
    result = BVPResult(
        success=True, status="success", message="accepted", method=method,
        p_opt=path[:, -1], t=[0.0, 1.0],
        y=np.repeat(np.arange(len(names), dtype=float)[:, None], 2, axis=1),
        ivp_success=True, optimizer_success=True, algorithm_success=True,
        finite_success=True, boundary_success=True,
        boundary_residual=[0.0, 0.0], boundary_residual_norm=0.0,
        solver_metadata=metadata,
    )
    return SolveRecord(request=request, result=result), mu, path


def _record_dimension(dimension: int):
    names = [f"x{i}" for i in range(dimension)]
    problem = BVPProblem(
        name=f"{dimension}-parameter trajectory", odes=["0"] * dimension,
        var_names=names,
        boundary_conditions=[f"x{i}_T" for i in range(dimension)],
        known_indices=[], unknown_indices=list(range(dimension)),
        known_values={}, initial_guess=[0.5] * dimension,
    )
    config = SolverConfig(method="differential_continuation")
    request = SolveRequest.create(problem=problem, config=config,
                                  source_task_id=f"dimension-{dimension}")
    mu = np.linspace(0.0, 1.0, 17)
    parameter_major = np.vstack([0.5 + (index + 1) * mu**2
                                 for index in range(dimension)])
    result = BVPResult(
        success=True, status="success", message="accepted",
        method="differential_continuation", p_opt=parameter_major[:, -1],
        t=[0.0, 1.0], y=np.zeros((dimension, 2)),
        ivp_success=True, optimizer_success=True, algorithm_success=True,
        finite_success=True, boundary_success=True,
        boundary_residual=np.zeros(dimension), boundary_residual_norm=0.0,
        solver_metadata={"differential_continuation": {"trajectory": {
            "mu": mu.tolist(), "p": parameter_major.tolist(),
        }}},
    )
    return SolveRecord(request=request, result=result), mu, parameter_major.T


def test_differential_result_exposes_parameter_plot(qapp):
    widget = IntegratedPlotWidget([_record()[0]], lang="zh")
    try:
        assert [widget.combo_plot_type.itemData(i) for i in range(widget.combo_plot_type.count())] == [
            "state", "parameter"
        ]
        assert widget.combo_plot_type.itemText(1) == "延拓参数 p(μ)"
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("method", ["shooting", "continuation"])
def test_other_methods_do_not_expose_parameter_plot(qapp, method):
    record, _, _ = _record(method)
    widget = IntegratedPlotWidget([record])
    try:
        assert widget.combo_plot_type.count() == 1
        assert widget.combo_plot_type.itemData(0) == "state"
        assert widget.combo_plot_type.isHidden()
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


def test_differential_without_trajectory_does_not_expose_parameter_plot(qapp):
    record, _, _ = _record(trajectory=False)
    widget = IntegratedPlotWidget([record])
    try:
        assert widget.combo_plot_type.count() == 1
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


def test_two_endpoint_values_are_not_presented_as_a_full_trajectory(qapp):
    record, _, _ = _record()
    data = record.result.to_dict()
    data["solver_metadata"]["differential_continuation"]["trajectory"] = {
        "mu": [0.0, 1.0], "p": [[0.5, 0.45], [-0.5, -0.3]],
    }
    short_result = BVPResult.from_legacy_dict(data)
    widget = IntegratedPlotWidget([SolveRecord(request=record.request, result=short_result)])
    try:
        assert widget.combo_plot_type.count() == 1
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("lang, expected", [
    ("zh", ("延拓参数 p(μ)", "微分参数延拓：p(μ)", "延拓参数 μ", "参数向量 p(μ) 的分量")),
    ("ru", ("Параметры p(μ)", "Дифференциальное продолжение: p(μ)",
            "Параметр продолжения μ", "Компоненты вектора p(μ)")),
])
def test_localized_parameter_labels(qapp, lang, expected):
    widget = IntegratedPlotWidget([_record()[0]], lang=lang)
    try:
        assert widget.combo_plot_type.itemText(1) == expected[0]
        widget.combo_plot_type.setCurrentIndex(1)
        ax = widget.fig.axes[0]
        assert (ax.get_title(), ax.get_xlabel(), ax.get_ylabel()) == expected[1:]
        assert "Unknown initial parameter" not in ax.get_ylabel()
        widget.set_lang("ru" if lang == "zh" else "zh")
        assert widget.combo_plot_type.currentData() == "parameter"
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


def test_parameter_plot_uses_all_saved_samples_and_state_plot_is_unchanged(qapp):
    record, mu, path = _record()
    widget = IntegratedPlotWidget([record])
    try:
        state_lines = [line for line in widget.fig.axes[0].lines if line.get_label() in {"a", "b"}]
        state_samples = [(line.get_label(), np.asarray(line.get_xdata()).copy(),
                          np.asarray(line.get_ydata()).copy()) for line in state_lines]
        widget.combo_plot_type.setCurrentIndex(1)
        parameter_lines = [line for line in widget.fig.axes[0].lines
                           if line.get_label().startswith("p")]
        assert len(parameter_lines) == 2
        for index, line in enumerate(parameter_lines):
            assert len(line.get_xdata()) == 201
            np.testing.assert_allclose(line.get_xdata(), mu)
            np.testing.assert_allclose(line.get_ydata(), path[index])
            assert not np.allclose(line.get_ydata(), np.linspace(path[index, 0], path[index, -1], 201))
        widget.combo_plot_type.setCurrentIndex(0)
        new_state_lines = [line for line in widget.fig.axes[0].lines if line.get_label() in {"a", "b"}]
        assert len(new_state_lines) == len(state_samples)
        for line, (label, x, y) in zip(new_state_lines, state_samples):
            assert line.get_label() == label
            np.testing.assert_array_equal(line.get_xdata(), x)
            np.testing.assert_array_equal(line.get_ydata(), y)
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


def test_textbook_26_1_parameter_names_only_when_state_meanings_are_known(qapp):
    semantic_record, _, _ = _record(semantic=True)
    widget = IntegratedPlotWidget([semantic_record])
    try:
        widget.combo_plot_type.setCurrentIndex(1)
        assert [line.get_label() for line in widget.fig.axes[0].lines] == [
            "p₁(μ) = vₓ(0)", "p₂(μ) = vᵧ(0)"
        ]
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


def test_mixed_plot_only_draws_saved_differential_trajectory(qapp):
    differential, _, _ = _record()
    shooting, _, _ = _record("shooting")
    widget = IntegratedPlotWidget([shooting, differential])
    try:
        widget.combo_plot_type.setCurrentIndex(1)
        assert len([line for line in widget.fig.axes[0].lines if line.get_label().startswith("p")]) == 2
        widget.sol_list.item(1).setCheckState(0)
        assert not widget.fig.axes[0].lines
        widget.update_data([shooting])
        assert widget.combo_plot_type.count() == 1
        assert widget.combo_plot_type.currentData() == "state"
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


def test_real_26_1_plot_uses_saved_dense_output_trajectory(qapp):
    problem, config = load_case()
    config = SolverConfig(method="differential_continuation", ivp_method=config.ivp_method,
                          eps=config.eps, continuation_steps=config.continuation_steps)
    result = solve_bvp_problem(problem, config)
    assert result.success
    request = SolveRequest.create(problem=problem, config=config, source_task_id="26-1-plot")
    widget = IntegratedPlotWidget([SolveRecord(request=request, result=result)])
    try:
        trajectory = result.solver_metadata["differential_continuation"]["trajectory"]
        assert len(trajectory["mu"]) == 201
        widget.combo_plot_type.setCurrentIndex(1)
        lines = widget.fig.axes[0].lines
        assert len(lines) == 2
        for index, line in enumerate(lines):
            np.testing.assert_allclose(line.get_xdata(), trajectory["mu"])
            np.testing.assert_allclose(line.get_ydata(), trajectory["p"][index])
        assert [line.get_label() for line in lines] == [
            "p₁(μ) = vₓ(0)", "p₂(μ) = vᵧ(0)"
        ]
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("dimension", [1, 2, 3, 4, 5])
def test_parameter_curve_count_and_sample_major_view_are_generic(qapp, dimension):
    record, mu, expected_path = _record_dimension(dimension)
    widget = IntegratedPlotWidget([record])
    try:
        actual_mu, actual_path = widget._parameter_trajectory(record)
        assert actual_path.shape == (len(mu), dimension)
        np.testing.assert_allclose(actual_mu, mu)
        np.testing.assert_allclose(actual_path, expected_path)
        actual_path[0, 0] = -999.0
        assert record.result.solver_metadata["differential_continuation"]["trajectory"]["p"][0][0] == 0.5
        widget.combo_plot_type.setCurrentIndex(1)
        lines = widget.fig.axes[0].lines
        assert len(lines) == dimension
        for index, line in enumerate(lines):
            np.testing.assert_allclose(line.get_xdata(), mu)
            np.testing.assert_allclose(line.get_ydata(), expected_path[:, index])
            assert line.get_label().startswith("p")
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


def test_open_and_toolbar_save_use_current_figure_without_solving(qapp, monkeypatch, tmp_path):
    record, _, _ = _record()
    widget = IntegratedPlotWidget([record])
    destination = tmp_path / "current_gui_plot.png"
    def unexpected_solve(*_args, **_kwargs):
        pytest.fail("plot or save attempted to rerun the solver")
    monkeypatch.setattr("main.solve_bvp_problem", unexpected_solve)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args, **_kwargs: (str(destination), "PNG"))
    try:
        widget.combo_plot_type.setCurrentIndex(1)
        widget._refresh_plot()
        widget.toolbar.save_figure()
        assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert len(widget.fig.axes[0].lines) == 2
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()
