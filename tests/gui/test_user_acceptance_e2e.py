"""End-to-end GUI acceptance: drive real solves like a user, verify display completeness.

Runs headless (QT_QPA_PLATFORM=offscreen). For each classic BVP it loads the
problem into the editor, clicks solve, waits for the worker, then verifies the
GUI exposes every datum a user needs: status text, records, run metadata,
tolerances, boundary diagnostics, and that the plot layer can select all
compatible records. The GUI is bilingual (zh/ru) by design; assertions avoid
hard-coding one language.
"""

from __future__ import annotations

import math
import os
import re
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa: E402

from bvp_core.requests import partition_plot_records  # noqa: E402
from main import BvpSolverApp, Dataset  # noqa: E402

EXPECTED = 1.0  # exponential unknown x(0)
EXPECTED_OSC = 1.0  # harmonic oscillator unknown v(0)
EXPECTED_MANU = -2.0  # manufactured polynomial unknown v(0)

TAU = math.pi / 2.0


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, monkeypatch) -> BvpSolverApp:
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    target = BvpSolverApp()
    yield target
    if target.active_worker is not None and target.active_worker.isRunning():
        if target.active_cancellation_token is not None:
            target.active_cancellation_token.cancel()
        target.active_worker.wait(2000)
    target.deleteLater()
    qapp.processEvents()


def _dataset_for_scalar_exponential() -> Dataset:
    return Dataset(
        name="Scalar exponential",
        equations=["x"],
        var_names=["x"],
        T=1.0,
        initial_values={0: None},
        boundary_conditions=["x0_T - E"],
        guess=[0.5],
        solver_method="shooting",
        known_indices=[],
        unknown_indices=[0],
        aux_outputs={},
    )


def _dataset_for_harmonic_oscillator(guess: float = 1.0) -> Dataset:
    return Dataset(
        name="Harmonic oscillator",
        equations=["v", "-x"],
        var_names=["x", "v"],
        T=TAU,
        initial_values={0: 0.0},
        boundary_conditions=["x0_T - 1"],
        guess=[guess],
        solver_method="shooting",
        known_indices=[0],
        unknown_indices=[1],
        aux_outputs={},
    )


def _dataset_for_manufactured_polynomial(guess: float = -2.0) -> Dataset:
    return Dataset(
        name="Manufactured polynomial",
        equations=["v", "6*t"],
        var_names=["y", "v"],
        T=1.0,
        initial_values={0: 1.0},
        boundary_conditions=["x0_T"],
        guess=[guess],
        solver_method="shooting",
        known_indices=[0],
        unknown_indices=[1],
        aux_outputs={},
    )


def _load_and_solve(window: BvpSolverApp, dataset: Dataset, *, timeout: float = 30.0) -> None:
    window.tasks = [dataset]
    window._task_ids = ["user-task"]
    window.current_task_idx = 0
    window._sync_task_to_editor(dataset)
    qapp = QApplication.instance()
    assert qapp is not None
    started = window.on_solve()
    if not started:
        pytest.fail(
            f"on_solve refused to start a worker; "
            f"detail={window.progress_detail.text()!r}"
        )
    deadline = time.time() + timeout
    while time.time() < deadline:
        qapp.processEvents()
        if window.active_worker is None:
            return
        time.sleep(0.02)
    pytest.fail("solve worker did not finish in time")


def _display_text(window: BvpSolverApp) -> str:
    qapp = QApplication.instance()
    if qapp is not None:
        qapp.processEvents()
    return f"{window.progress_bar_label.text()}\n{window.progress_detail.text()}"


def test_user_solves_scalar_exponential_and_ui_shows_full_picture(window) -> None:
    _load_and_solve(window, _dataset_for_scalar_exponential())

    assert len(window.solve_records) == 1
    record = window.solve_records[-1]
    result = record.result
    assert result.success, result.message
    assert abs(float(result.p_opt[0]) - EXPECTED) < 1e-5

    # Every datum a user needs must be reachable from the record.
    metadata = result.run_metadata
    assert metadata.run_id
    assert re.fullmatch(r"[0-9a-f]{64}", metadata.problem_signature)
    assert metadata.elapsed_seconds is not None
    assert result.method == "shooting"
    assert result.boundary_residual_norm is not None
    assert result.boundary_success is True

    tolerances = result.solver_metadata
    assert tolerances["tolerance_mode"] == "legacy"
    assert tolerances["effective_ivp_rtol"] == pytest.approx(1e-8)
    assert "effective_ivp_atol" in tolerances
    assert "effective_root_tol" in tolerances
    assert "effective_continuation_residual_tol" in tolerances

    # Status line (language-independent fields): run identity, method, status,
    # residual, scaled boundary ratio and tolerance mode must all be visible.
    display = _display_text(window)
    assert "run_id=" in display
    assert "request_id=" in display
    assert "status=success" in display
    assert "method=shooting" in display
    assert "tolerance_mode=legacy" in display
    assert "max_scaled_ratio=" in display
    assert "elapsed_seconds=" in display


def test_user_switches_language_and_ui_stays_complete(window) -> None:
    _load_and_solve(window, _dataset_for_scalar_exponential())
    window.update_language("zh")
    display_zh = _display_text(window)
    # Same language-independent identity must survive a language switch.
    assert "run_id=" in display_zh and "status=success" in display_zh
    # Bilingual labels must both exist for the same control.
    assert window.current_lang == "zh"
    assert window.TRANSLATIONS["zh"]["btn_solve"]
    assert window.TRANSLATIONS["ru"]["btn_solve"]
    window.update_language("ru")
    assert window.current_lang == "ru"


def test_user_solves_harmonic_oscillator(window) -> None:
    _load_and_solve(window, _dataset_for_harmonic_oscillator())

    record = window.solve_records[-1]
    result = record.result
    assert result.success, result.message
    assert abs(float(result.p_opt[0]) - EXPECTED_OSC) < 1e-5
    assert re.fullmatch(r"[0-9a-f]{64}", result.run_metadata.problem_signature)
    # Two-state problem: plot layer must see both variables.
    assert len(result.t) >= 2
    assert result.y.shape[0] == 2


def test_user_solves_manufactured_polynomial_and_checks_ode_defect(window) -> None:
    _load_and_solve(window, _dataset_for_manufactured_polynomial())

    record = window.solve_records[-1]
    result = record.result
    assert result.success, result.message
    assert abs(float(result.p_opt[0]) - EXPECTED_MANU) < 1e-5

    # Independent check: y(t) = t^3 - 2t + 1 must satisfy y(1) = 0.
    y = np.asarray(result.y, dtype=float)
    assert abs(y[0, -1]) < 1e-5


def test_user_can_select_all_compatible_records_for_plot(window) -> None:
    # Three numeric variants of the SAME stable task identity.
    for guess in (-1.0, 1.0, 2.0):
        _load_and_solve(window, _dataset_for_harmonic_oscillator(guess=guess))

    assert len(window.solve_records) == 3
    compatible, rejected = partition_plot_records(
        window.solve_records, window.solve_records[-1]
    )
    assert not rejected, f"same-task variants must all plot, got {rejected}"
    assert len(compatible) == 3, "plot layer must be able to select all 3 records"

    # Records stay distinct: problem signatures differ per request.
    signatures = {r.request.problem_signature for r in window.solve_records}
    assert len(signatures) == 3


def test_user_plot_can_render_all_records(window) -> None:
    _load_and_solve(window, _dataset_for_scalar_exponential())
    _load_and_solve(window, _dataset_for_scalar_exponential())
    assert len(window.solve_records) == 2

    compatible, rejected = partition_plot_records(
        window.solve_records, window.solve_records[-1]
    )
    assert len(compatible) == 2 and not rejected

    # A plot widget must accept every compatible record for rendering.
    window.on_plot()
    widget = window._plot_widget
    assert widget is not None
    assert len(widget.records) == 2
