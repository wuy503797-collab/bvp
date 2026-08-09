#!/usr/bin/env python3
"""
通用BVP求解器 v2 -- 集成式实时交互绘图
基于参数延拓法, PyQt5 + matplotlib 嵌入

改进:
  - IntegratedPlotWidget: matplotlib嵌入PyQt5, 实时控件交互
  - 无需弹对话框, 所有绘图参数在图像窗口直接调整
  - 支持多Y叠加, S/F标记, 等比例, 网格等实时切换
"""

#!/usr/bin/env python3
"""
Универсальный решатель краевых задач (BVP) с методом продолжения по параметру.
通用常微分方程边值问题求解器（含参数延续法）

Архитектура:
  - Dataset      : dataclass для хранения параметров задачи
  - bvp_core     : SymPy parsing and numerical BVP implementation
  - SolverWorker : QThread для неблокирующих вычислений
  - PlotDialog   : универсальный диалог выбора осей графика
  - BvpSolverApp : PyQt5 GUI

См. кафедру ОУ ВМиК МГУ — Метод продолжения по параметру.
"""

import sys
import json
import logging
import re
import time
import traceback
import warnings
from dataclasses import dataclass, field, asdict
from typing import List, Callable, Optional, Tuple, Dict, Any
from uuid import uuid4

import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLineEdit,
    QComboBox, QTableWidget, QTableWidgetItem, QGroupBox,
    QDoubleSpinBox, QSpinBox, QFrame, QAction, QShortcut,
    QHeaderView, QColorDialog, QFileDialog,
    QMessageBox, QInputDialog, QDialog, QCheckBox,
    QDialogButtonBox, QTextEdit, QSplitter, QTabWidget,
    QScrollArea, QFormLayout, QSizePolicy,
    QListWidget, QListWidgetItem,
    QRadioButton, QButtonGroup,
    QDockWidget, QActionGroup,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QKeySequence, QColor

from bvp_core import (
    RunContext,
    build_canonical_export_record,
    emit_solver_event,
    export_canonical_record,
    solve_bvp_problem,
)
from bvp_core.adapters import (
    config_from_dataset,
    problem_from_dataset,
    solver_from_dataset as BVPSolver,
)
from bvp_core.exceptions import ExpressionValidationError, IVPIntegrationError
from bvp_core.expressions import SymPyParser
from bvp_core.requests import (
    CancellationToken,
    GuiSolveState,
    SolveCancelled,
    SolveOutcome,
    SolveOutcomeStatus,
    SolveRecord,
    SolveRequest,
    gui_transition_allowed,
    partition_plot_records,
)
from bvp_core.results import BVPResult
from bvp_core.observability import build_run_metadata, metadata_context
from bvp_core.serialization import write_json_data_atomic


LOGGER = logging.getLogger(__name__)


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)

# ---------------------------------------------------------------------------
# 1. Dataset — централизованное хранение параметров задачи
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    """
    Универсальный контейнер для параметров краевой задачи.
    """
    name: str = "Untitled"
    # ODE: список строк правых частей dx_i/dt = f_i(t, x0, x1, ...)
    equations: List[str] = field(default_factory=list)
    # Имена переменных состояния (автоматически: x0, x1, ... если не заданы)
    var_names: List[str] = field(default_factory=list)
    # Время интегрирования
    T: float = 1.0
    # Начальные условия: {index: value} — известные; None — неизвестные (искомые)
    initial_values: Dict[int, Optional[float]] = field(default_factory=dict)
    # Граничные условия: список строк вида "expr = 0" или "expr"
    # Переменные: x0_0=x0(0), x1_0=x1(0), ..., x0_T=x0(T), x1_T=x1(T)
    boundary_conditions: List[str] = field(default_factory=list)
    # Начальное приближение для НЕИЗВЕСТНЫХ начальных значений (по порядку индексов)
    guess: List[float] = field(default_factory=list)
    # Параметры интегрирования
    eps: Optional[float] = 1e-8
    # Независимый критерий приёмки конечной граничной невязки.
    # Финальная приёмка использует явные покомпонентные масштабы.
    boundary_atol: float = 1e-8
    boundary_rtol: float = 0.0
    boundary_scales: Optional[List[float]] = None
    ivp_rtol: Optional[float] = None
    ivp_atol: Optional[float] = None
    root_tol: Optional[float] = None
    least_squares_ftol: Optional[float] = None
    least_squares_xtol: Optional[float] = None
    least_squares_gtol: Optional[float] = None
    continuation_residual_tol: Optional[float] = None
    jacobian_relative_step: Optional[float] = None
    method: str = "RK45"
    # Выбор метода решения: "shooting" | "continuation"
    solver_method: str = "continuation"
    # Параметры продолжения по параметру
    continuation_steps: int = 50
    # Индексы переменных с фиксированными начальными значениями (known)
    known_indices: List[int] = field(default_factory=list)
    # Индексы переменных с неизвестными начальными значениями (unknown = to solve for)
    unknown_indices: List[int] = field(default_factory=list)
    # t* — точка, где задаётся начальное условие (0 по умолчанию)
    t_star: float = 0.0
    # Вспомогательные выходные переменные: {name: expr_str}
    # Примеры: {"u": "0.5*(sqrt(1e-10+(x5/2+1)**2)-sqrt(1e-10+(x5/2-1)**2))"}
    aux_outputs: Dict[str, str] = field(default_factory=dict)

    # ---- сериализация (JSON-совместимая) --------------------------------

    def to_dict(self) -> dict:
        """Полная сериализация в JSON-словарь."""
        return {
            "name": self.name,
            "equations": self.equations,
            "var_names": self.var_names,
            "T": self.T,
            "initial_values": {str(k): v for k, v in self.initial_values.items()},
            "boundary_conditions": self.boundary_conditions,
            "guess": self.guess,
            "eps": self.eps,
            "boundary_atol": self.boundary_atol,
            "boundary_rtol": self.boundary_rtol,
            "boundary_scales": (
                None if self.boundary_scales is None else list(self.boundary_scales)
            ),
            "ivp_rtol": self.ivp_rtol,
            "ivp_atol": self.ivp_atol,
            "root_tol": self.root_tol,
            "least_squares_ftol": self.least_squares_ftol,
            "least_squares_xtol": self.least_squares_xtol,
            "least_squares_gtol": self.least_squares_gtol,
            "continuation_residual_tol": self.continuation_residual_tol,
            "jacobian_relative_step": self.jacobian_relative_step,
            "method": self.method,
            "solver_method": self.solver_method,
            "continuation_steps": self.continuation_steps,
            "known_indices": self.known_indices,
            "unknown_indices": self.unknown_indices,
            "t_star": self.t_star,
            "aux_outputs": dict(self.aux_outputs),
        }

    @staticmethod
    def from_dict(d: dict) -> "Dataset":
        """Десериализация из JSON-словаря."""
        # Конвертируем строковые ключи initial_values обратно в int
        iv = d.get("initial_values", {})
        if iv:
            initial_values = {}
            for k, v in iv.items():
                if v is not None:
                    initial_values[int(k)] = float(v)
                else:
                    initial_values[int(k)] = None
        else:
            initial_values = {}
        return Dataset(
            name=d.get("name", "Untitled"),
            equations=list(d.get("equations", [])),
            var_names=list(d.get("var_names", [])),
            T=float(d.get("T", 1.0)),
            initial_values=initial_values,
            boundary_conditions=list(d.get("boundary_conditions", [])),
            guess=list(d.get("guess", [])),
            eps=None if d.get("eps", 1e-8) is None else float(d.get("eps", 1e-8)),
            boundary_atol=float(d.get("boundary_atol", 1e-8)),
            boundary_rtol=float(d.get("boundary_rtol", 0.0)),
            boundary_scales=(
                None
                if d.get("boundary_scales") is None
                else [float(value) for value in d["boundary_scales"]]
            ),
            ivp_rtol=_optional_float(d.get("ivp_rtol")),
            ivp_atol=_optional_float(d.get("ivp_atol")),
            root_tol=_optional_float(d.get("root_tol")),
            least_squares_ftol=_optional_float(d.get("least_squares_ftol")),
            least_squares_xtol=_optional_float(d.get("least_squares_xtol")),
            least_squares_gtol=_optional_float(d.get("least_squares_gtol")),
            continuation_residual_tol=_optional_float(
                d.get("continuation_residual_tol")
            ),
            jacobian_relative_step=_optional_float(d.get("jacobian_relative_step")),
            method=d.get("method", "RK45"),
            solver_method=d.get("solver_method", "continuation"),
            continuation_steps=int(d.get("continuation_steps", 50)),
            known_indices=list(d.get("known_indices", [])),
            unknown_indices=list(d.get("unknown_indices", [])),
            t_star=float(d.get("t_star", 0.0)),
            aux_outputs=dict(d.get("aux_outputs", {})),
        )

    def dim(self) -> int:
        return len(self.equations)

    def num_unknown(self) -> int:
        return len(self.unknown_indices)

    def validate(self) -> List[str]:
        """Return core-model validation errors while preserving the legacy list API."""
        from bvp_core.adapters import config_from_dataset, problem_from_dataset
        from bvp_core.models import BVPValidationError

        errors: List[str] = []
        problem = None
        try:
            problem = problem_from_dataset(self)
        except BVPValidationError as exc:
            errors.extend(exc.errors)
        try:
            config_from_dataset(self)
        except BVPValidationError as exc:
            errors.extend(exc.errors)
        if problem is not None:
            try:
                parser = SymPyParser(list(problem.odes), list(problem.var_names))
                parser.lambdify_all()
                SymPyParser.parse_boundary_conditions(
                    list(problem.boundary_conditions), list(problem.var_names)
                )
            except ExpressionValidationError as exc:
                errors.extend(exc.errors)
        return errors


def is_result_acceptable(result: dict) -> bool:
    """Return whether a solver result may enter GUI history, plots, and export."""
    required_flags = ("success", "ivp_success", "finite_success", "boundary_success")
    return all(result.get(flag) is True for flag in required_flags)

# ---------------------------------------------------------------------------
# 2. SolverWorker — вычисления в отдельном потоке через публичный core API
# ---------------------------------------------------------------------------

class SolverWorker(QThread):
    """
    Поток для выполнения BVP-решения.
    Every signal carries the immutable request ID. Numerical failures retain a
    BVPResult, while programming failures retain a technical traceback.
    """
    request_started = pyqtSignal(str)
    progress = pyqtSignal(str, str, int, str)
    request_finished = pyqtSignal(str, object)
    request_failed = pyqtSignal(str, object)
    request_cancelled = pyqtSignal(str, object)

    def __init__(
        self,
        request: SolveRequest,
        cancellation_token: Optional[CancellationToken] = None,
    ):
        super().__init__()
        self.request = request
        self.cancellation_token = cancellation_token or CancellationToken()
        self.run_context = RunContext.create(
            request.problem,
            request.config,
            request_id=request.request_id,
            source_task_id=request.source_task_id,
        )

    def request_cancel(self) -> None:
        self.cancellation_token.cancel()
        emit_solver_event(
            self.run_context,
            "cancel_requested",
            phase="cancellation",
            status="requested",
        )

    def run(self):
        request_id = self.request.request_id
        started_perf = time.perf_counter()
        self.request_started.emit(request_id)
        try:
            self.cancellation_token.raise_if_cancelled()
            self.cancellation_token.raise_if_cancelled()
            result = solve_bvp_problem(
                self.request.problem,
                self.request.config,
                cancellation_check=self.cancellation_token.raise_if_cancelled,
                callback=self._on_progress,
                run_context=self.run_context,
            )
            self.cancellation_token.raise_if_cancelled()
            if result.success:
                outcome = SolveOutcome(
                    request=self.request,
                    status=SolveOutcomeStatus.COMPLETED,
                    result=result,
                    message=result.message,
                )
                self.request_finished.emit(request_id, outcome)
            else:
                outcome = SolveOutcome(
                    request=self.request,
                    status=SolveOutcomeStatus.FAILED,
                    result=result,
                    message=result.message,
                )
                self.request_failed.emit(request_id, outcome)
        except SolveCancelled as exc:
            run_metadata = exc.run_metadata or build_run_metadata(
                self.run_context,
                self.request.config,
                boundary_count=len(self.request.problem.boundary_conditions),
                final_status="cancelled",
                elapsed_seconds=time.perf_counter() - started_perf,
            )
            if exc.run_metadata is None:
                emit_solver_event(
                    self.run_context,
                    "solve_cancelled",
                    phase="solve",
                    status="cancelled",
                    details={"elapsed_seconds": run_metadata.elapsed_seconds},
                )
            outcome = SolveOutcome(
                request=self.request,
                status=SolveOutcomeStatus.CANCELLED,
                message=str(exc),
                run_metadata=run_metadata,
            )
            self.request_cancelled.emit(request_id, outcome)
        except Exception as exc:
            technical_diagnostic = traceback.format_exc()
            run_metadata = getattr(exc, "run_metadata", None) or build_run_metadata(
                self.run_context,
                self.request.config,
                boundary_count=len(self.request.problem.boundary_conditions),
                final_status="internal_error",
                elapsed_seconds=time.perf_counter() - started_perf,
            )
            LOGGER.debug(
                "Technical traceback for run %s",
                self.run_context.run_id,
                exc_info=True,
            )
            outcome = SolveOutcome(
                request=self.request,
                status=SolveOutcomeStatus.FAILED,
                message=f"{type(exc).__name__}: {exc}",
                technical_diagnostic=technical_diagnostic,
                run_metadata=run_metadata,
            )
            self.request_failed.emit(request_id, outcome)

    def _on_progress(self, method: str, percent: int, message: str):
        self.cancellation_token.raise_if_cancelled()
        self.progress.emit(self.request.request_id, method, percent, message)


# ---------------------------------------------------------------------------
# 5. IntegratedPlotWidget -- 集成式实时交互绘图
# ---------------------------------------------------------------------------

class IntegratedPlotWidget(QWidget):
    """
    集成式绘图窗口: matplotlib 嵌入 PyQt5 + 实时控制面板。
    所有控件变化立即重绘图像, 无需弹对话框。
    """

    TRANSLATIONS = {
        "zh": {
            "plot_controls": "绘图控制",
            "x_axis": "X 轴:",
            "y_axes": "Y 轴:",
            "line_width": "线宽:",
            "grid": "网格",
            "sf_markers": "S/F 标记",
            "equal_aspect": "等比例",
            "solutions": "解",
            "ls_all_solid": "全部实线",
            "ls_aux_solid": "辅实状虚",
            "ls_aux_dash": "辅虚实实",
            "ls_tooltip": "循环切换线型模式",
        },
        "ru": {
            "plot_controls": "Управление графиком",
            "x_axis": "Ось X:",
            "y_axes": "Оси Y:",
            "line_width": "Толщина:",
            "grid": "Сетка",
            "sf_markers": "Маркеры S/F",
            "equal_aspect": "Равные пропорции",
            "solutions": "Решения",
            "ls_all_solid": "Все сплошные",
            "ls_aux_solid": "Вспом. сплошн.",
            "ls_aux_dash": "Вспом. пунктир",
            "ls_tooltip": "Переключить стиль линии",
        },
    }

    def __init__(
        self,
        records: List[SolveRecord],
        lang: str = "zh",
        parent=None,
    ):
        super().__init__(parent)
        self._set_records(records)
        self.lang = lang
        self._visible = set(range(len(records)))
        self._ls_mode = 0
        self.plot_diagnostics: tuple[str, ...] = ()
        self.setWindowFlags(Qt.Window)
        self.resize(950, 680)
        self._build_ui()
        self._refresh_plot()

    def _set_records(self, records: List[SolveRecord]) -> None:
        if not records:
            raise ValueError("plotting requires at least one SolveRecord")
        _, rejected = partition_plot_records(records, records[-1])
        if rejected:
            reasons = "; ".join(
                f"{record.request_id}: {reason}" for record, reason in rejected
            )
            raise ValueError(f"incompatible SolveRecord plot set: {reasons}")
        self.records = list(records)
        self.var_names = list(records[0].request.var_names)
        self.all_results = [record.to_legacy_dict() for record in records]
        self.aux_names = sorted(
            {
                name
                for record in records
                for name in record.auxiliary_outputs
            }
        )

    def _solution_label(self, index: int) -> str:
        record = self.records[index]
        result = record.result
        config = record.request.config
        guess = np.array2string(
            np.asarray(record.request.problem.initial_guess, dtype=float),
            precision=4,
            separator=", ",
        )
        tolerance = config.eps if config.eps is not None else config.ivp_rtol
        return (
            f"#{index + 1} {record.request.task_name} "
            f"[{record.request_id[:8]}/{record.request.problem_signature[:8]}] "
            f"guess={guess} ({result.method}/{config.ivp_method}, "
            f"tol={tolerance:.1e}, "
            f"‖Φ‖={result.residual_norm:.2e})"
        )

    def update_data(self, records: List[SolveRecord], lang: str = None):
        """Update the plot from provenance-compatible records."""
        self._set_records(records)
        if lang is not None:
            self.lang = lang
        self._visible = set(range(len(records)))
        self._rebuild_controls()
        self._refresh_plot()

    def set_lang(self, lang: str):
        """切换界面语言."""
        self.lang = lang
        self._update_ui_texts()
        self._refresh_plot()

    def _t(self, key: str) -> str:
        """获取当前语言的翻译."""
        return self.TRANSLATIONS.get(self.lang, self.TRANSLATIONS["zh"]).get(key, key)

    def _build_ui(self):
        """构建 UI: 左侧控制面板 + 右侧图像区."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # ===== 左侧控制面板 =====
        self.panel = QGroupBox()
        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setSpacing(6)

        self.form_layout = QFormLayout()

        # --- X 轴 ---
        self.combo_x = QComboBox()
        for name in self.var_names:
            self.combo_x.addItem(name, name)
        self.combo_x.addItem("t", "t")
        if len(self.var_names) > 2:
            self.combo_x.setCurrentIndex(self.combo_x.count() - 1)
        self.combo_x.currentIndexChanged.connect(self._refresh_plot)
        self._label_x = QLabel()
        self.form_layout.addRow(self._label_x, self.combo_x)

        # --- Y 轴 (多选) ---
        self.y_list = QListWidget()
        self.y_list.setSelectionMode(QListWidget.MultiSelection)
        self._populate_y_list()
        self.y_list.itemSelectionChanged.connect(self._refresh_plot)
        self.y_list.setMaximumHeight(150)
        self._label_y = QLabel()
        self.form_layout.addRow(self._label_y, self.y_list)

        # --- 线宽 ---
        self.spin_lw = QDoubleSpinBox()
        self.spin_lw.setRange(0.5, 5.0)
        self.spin_lw.setDecimals(1)
        self.spin_lw.setValue(1.5)
        self.spin_lw.setSingleStep(0.5)
        self.spin_lw.valueChanged.connect(self._refresh_plot)
        self._label_lw = QLabel()
        self.form_layout.addRow(self._label_lw, self.spin_lw)

        panel_layout.addLayout(self.form_layout)

        # --- 选项 ---
        self.check_grid = QCheckBox()
        self.check_grid.setChecked(True)
        self.check_grid.stateChanged.connect(self._refresh_plot)
        panel_layout.addWidget(self.check_grid)

        self.check_markers = QCheckBox()
        self.check_markers.setChecked(True)
        self.check_markers.stateChanged.connect(self._refresh_plot)
        panel_layout.addWidget(self.check_markers)

        self.check_equal = QCheckBox()
        self.check_equal.setChecked(False)
        self.check_equal.stateChanged.connect(self._refresh_plot)
        panel_layout.addWidget(self.check_equal)

        # 线型模式切换按钮
        self._ls_mode = 0
        self._ls_label_keys = ["ls_all_solid", "ls_aux_solid", "ls_aux_dash"]
        self.btn_linestyle = QPushButton()
        self.btn_linestyle.clicked.connect(self._toggle_linestyle)
        panel_layout.addWidget(self.btn_linestyle)

        panel_layout.addStretch()

        # Solutions 选择
        self.sol_group = QGroupBox()
        sol_layout = QVBoxLayout(self.sol_group)
        self.sol_list = QListWidget()
        self._colors_rgb = [(31,119,180),(255,127,14),(44,160,44),
                            (214,39,40),(148,103,189),(140,86,75),
                            (227,119,194),(127,127,127)]
        for i, r in enumerate(self.all_results):
            item = QListWidgetItem(self._solution_label(i))
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, i)
            c = self._colors_rgb[i % len(self._colors_rgb)]
            item.setBackground(QColor(c[0], c[1], c[2], 40))
            self.sol_list.addItem(item)
        self.sol_list.itemChanged.connect(self._on_sol_toggled)
        sol_layout.addWidget(self.sol_list)
        panel_layout.addWidget(self.sol_group)

        self._update_ui_texts()
        main_layout.addWidget(self.panel, 0)

        # ===== 右侧图像区 =====
        right = QVBoxLayout()
        self.fig = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right.addWidget(self.toolbar)
        self.diagnostic_label = QLabel("")
        self.diagnostic_label.setWordWrap(True)
        self.diagnostic_label.setStyleSheet("color: #c0392b; font-size: 11px;")
        right.addWidget(self.diagnostic_label)
        main_layout.addLayout(right, 1)

    def _populate_y_list(self):
        """填充 Y 轴列表: 状态变量 + [辅助变量]."""
        self.y_list.clear()
        for name in self.var_names:
            self.y_list.addItem(name)
        # 辅助变量（如果有）— 用不同前缀标识
        if self.aux_names:
            for name in self.aux_names:
                item = QListWidgetItem(f"[{name}]")
                item.setData(Qt.UserRole, name)  # 存储原始名
                item.setForeground(QColor(148, 103, 189))
                self.y_list.addItem(item)
        self.y_list.addItem("t")
        # 默认选中前 min(3, n_vars) 个状态变量
        n_vars = len(self.var_names)
        for r in range(min(3, n_vars)):
            self.y_list.item(r).setSelected(True)

    def _rebuild_controls(self):
        """数据更新后重建控件."""
        self._populate_y_list()

        cur_x = self.combo_x.currentData()
        self.combo_x.clear()
        for name in self.var_names:
            self.combo_x.addItem(name, name)
        self.combo_x.addItem("t (time)", "t")
        if len(self.var_names) > 2:
            self.combo_x.setCurrentIndex(self.combo_x.count() - 1)

        self.sol_list.clear()
        for i, r in enumerate(self.all_results):
            item = QListWidgetItem(self._solution_label(i))
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, i)
            c = self._colors_rgb[i % len(self._colors_rgb)]
            item.setBackground(QColor(c[0], c[1], c[2], 40))
            self.sol_list.addItem(item)
        self._visible = set(range(len(self.all_results)))

    def _on_sol_toggled(self, item):
        idx = item.data(Qt.UserRole)
        if item.checkState() == Qt.Checked:
            self._visible.add(idx)
        else:
            self._visible.discard(idx)
        self._refresh_plot()

    def _get_y_axes(self) -> List[str]:
        y_axes = []
        for item in self.y_list.selectedItems():
            txt = item.text()
            if txt == "t":
                y_axes.append("t")
            elif txt.startswith("[") and txt.endswith("]"):
                # 辅助变量: [name] -> name
                y_axes.append(item.data(Qt.UserRole) or txt[1:-1])
            else:
                y_axes.append(txt)
        if not y_axes:
            y_axes = [self.var_names[0]] if self.var_names else ["t"]
        return y_axes

    def _refresh_plot(self):
        """核心: 根据当前控件状态重绘图像."""
        if not self.all_results:
            return

        x_axis = self.combo_x.currentData() or "t"
        y_axes = self._get_y_axes()
        if x_axis != "t":
            y_axes = [y_axes[0]]

        lw = self.spin_lw.value()
        show_grid = self.check_grid.isChecked()
        show_markers = self.check_markers.isChecked()
        equal = self.check_equal.isChecked()

        colors = [(31/255,119/255,180/255),(255/255,127/255,14/255),
                  (44/255,160/255,44/255),(214/255,39/255,40/255),
                  (148/255,103/255,189/255),(140/255,86/255,75/255),
                  (227/255,119/255,194/255),(127/255,127/255,127/255)]
        linestyles = ["-", "--", ":", "-."]
        is_phase = (x_axis != "t" and y_axes[0] != "t")

        def axis_data(name, t_vals, y_vals):
            if name == "t":
                return t_vals, "t"
            return y_vals[self.var_names.index(name)], name

        # 收集数据（状态变量 + 辅助变量）
        aux_set = set(self.aux_names)
        curves = []
        diagnostics: list[str] = []
        for i, record in enumerate(self.records):
            if i not in self._visible:
                continue
            try:
                td = record.primary_plot_t
                yd = record.primary_plot_y
                xd, xl = axis_data(x_axis, td, yd)
                # 辅助变量数据
                aux_data = record.auxiliary_outputs
                for y_name in y_axes:
                    if y_name in aux_set:
                        aux_vals = aux_data.get(y_name)
                        if aux_vals is None:
                            diagnostics.append(
                                f"request {record.request_id}: auxiliary {y_name!r} "
                                "is unavailable"
                            )
                            continue
                        aux_t = record.auxiliary_sample_t
                        if x_axis == "t":
                            aux_x = aux_t
                        else:
                            state_index = self.var_names.index(x_axis)
                            aux_x = np.interp(aux_t, td, yd[state_index])
                        curves.append(
                            (i, y_name, np.asarray(aux_x), np.asarray(aux_vals))
                        )
                    else:
                        yd_arr, yl = axis_data(y_name, td, yd)
                        curves.append(
                            (i, y_name, np.asarray(xd), np.asarray(yd_arr))
                        )
            except (KeyError, ValueError, TypeError, IndexError, AttributeError) as exc:
                diagnostics.append(
                    f"request {record.request_id}: "
                    f"{type(exc).__name__}: {exc}"
                )

        self.plot_diagnostics = tuple(diagnostics)
        self.diagnostic_label.setText("\n".join(diagnostics))
        for diagnostic in diagnostics:
            LOGGER.warning("Plot diagnostic: %s", diagnostic)

        if not curves:
            if not diagnostics:
                self.diagnostic_label.setText("No compatible plot curves are available.")
            return

        # 重绘
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        n_y = len(y_axes)
        n_sol = len(set(c[0] for c in curves))

        # 线型模式: 0=全部实线, 1=辅实状虚(教材26.4), 2=全部虚线
        ls_mode = getattr(self, '_ls_mode', 0)
        for sol_idx, y_name, xd, yd in curves:
            y_idx = y_axes.index(y_name) if y_name in y_axes else 0
            c = colors[y_idx % len(colors)]
            is_aux = y_name in aux_set
            if n_sol > 1:
                base_ls = linestyles[sol_idx % len(linestyles)]
                if ls_mode == 0:
                    ls = base_ls  # 全部实线(按解区分)
                elif ls_mode == 1:
                    ls = "-" if is_aux else "--"  # 辅实状虚
                else:
                    ls = "--" if is_aux else "-"  # 辅虚实实
            else:
                if ls_mode == 0:
                    ls = "-"  # 全部实线
                elif ls_mode == 1:
                    ls = "-" if is_aux else "--"  # 辅实状虚
                else:
                    ls = "--" if is_aux else "-"  # 辅虚实实
            label = f"{y_name}" if n_sol == 1 else f"{y_name} #{sol_idx+1}"
            ax.plot(xd, yd, ls=ls, color=c, lw=lw, label=label)

        # 标签
        _, x_lbl = axis_data(x_axis, None, np.zeros(len(self.var_names)))
        ax.set_xlabel(f"$\\mathit{{{x_lbl}}}$", fontsize=13)
        if is_phase:
            y_lbl = y_axes[0]
            ax.set_ylabel(f"$\\mathit{{{y_lbl}}}$", fontsize=13)
            ax.set_title(f"${y_lbl}$ vs ${x_lbl}$", fontsize=14)
        elif x_axis == "t":
            y_label_str = ", ".join(y_axes)
            ax.set_ylabel(f"$\\mathit{{{y_label_str}}}$", fontsize=12)
            ax.set_title("$x_i(t)$" if len(y_axes) > 1 else f"${y_axes[0]}(t)$", fontsize=14)
        else:
            ax.set_ylabel(f"$\\mathit{{{y_axes[0]}}}$", fontsize=13)

        # S/F 标记
        if show_markers and is_phase:
            seen = set()
            sf_off = [(10, 8), (-14, -12), (10, -14), (-14, 8)]
            ff_off = [(10, -12), (-14, 8), (10, 10), (-14, -14)]
            for sol_idx, y_name, xd, yd in curves:
                if y_name != y_axes[0] or sol_idx in seen:
                    continue
                seen.add(sol_idx)
                y_idx = y_axes.index(y_name) if y_name in y_axes else 0
                c = colors[y_idx % len(colors)]
                ax.plot(xd[0], yd[0], "o", color=c, ms=7, mec="black", mew=0.5)
                ax.plot(xd[-1], yd[-1], "s", color=c, ms=7, mec="black", mew=0.5)
                idx = len(seen) - 1
                ax.annotate("S", (xd[0], yd[0]), textcoords="offset points",
                           xytext=sf_off[idx % 4], fontsize=13, fontweight="bold", color=c)
                ax.annotate("F", (xd[-1], yd[-1]), textcoords="offset points",
                           xytext=ff_off[idx % 4], fontsize=13, fontweight="bold", color=c)

        if show_grid:
            ax.grid(True, alpha=0.3, ls="--")
        ax.legend(loc="best", fontsize=10, framealpha=0.9)
        ax.tick_params(labelsize=11)
        # 零轴 x=0, y=0: 加粗实线, 像传统坐标系
        ax.axhline(y=0, color="black", linewidth=1.5, zorder=0)
        ax.axvline(x=0, color="black", linewidth=1.5, zorder=0)
        if is_phase and equal:
            ax.set_aspect("equal")
        self.fig.tight_layout()
        self.canvas.draw()

    def _update_ui_texts(self):
        """根据当前语言更新所有控件文本."""
        self.panel.setTitle(self._t("plot_controls"))
        self._label_x.setText(self._t("x_axis"))
        self._label_y.setText(self._t("y_axes"))
        self._label_lw.setText(self._t("line_width"))
        self.check_grid.setText(self._t("grid"))
        self.check_markers.setText(self._t("sf_markers"))
        self.check_equal.setText(self._t("equal_aspect"))
        self.sol_group.setTitle(self._t("solutions"))
        self.btn_linestyle.setText(self._t(self._ls_label_keys[self._ls_mode]))
        self.btn_linestyle.setToolTip(self._t("ls_tooltip"))
        # 更新窗口标题
        t = BvpSolverApp.TRANSLATIONS.get(self.lang, BvpSolverApp.TRANSLATIONS["zh"])
        self.setWindowTitle(t.get("btn_plot", "绘图"))

    def _toggle_linestyle(self):
        """循环切换线型模式: 0→1→2→0."""
        self._ls_mode = (self._ls_mode + 1) % 3
        self.btn_linestyle.setText(self._t(self._ls_label_keys[self._ls_mode]))
        self._refresh_plot()

    def closeEvent(self, event):
        event.accept()
        if self.parent() and hasattr(self.parent(), '_plot_widget'):
            self.parent()._plot_widget = None

class EquationEditor(QWidget):
    """
    Вкладка для ввода уравнений ODE, граничных условий и начальных значений.
    """

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # --- Размерность ---
        h = QHBoxLayout()
        self.label_dim = QLabel("Размерность системы / System dimension:")
        h.addWidget(self.label_dim)
        self.spin_dim = QSpinBox()
        self.spin_dim.setRange(1, 20)
        self.spin_dim.setValue(2)
        self.spin_dim.valueChanged.connect(self._on_dim_changed)
        h.addWidget(self.spin_dim)
        h.addStretch()
        layout.addLayout(h)

        # --- Уравнения ---
        self.group_eq = QGroupBox()
        self.eq_layout = QVBoxLayout(self.group_eq)
        self.eq_editors: List[QLineEdit] = []
        layout.addWidget(self.group_eq)

        # --- Граничные условия ---
        self.group_bc = QGroupBox()
        bc_layout = QVBoxLayout(self.group_bc)
        self.bc_editor = QTextEdit()
        self.bc_editor.setPlaceholderText("")
        self.bc_editor.setMaximumHeight(100)
        bc_layout.addWidget(self.bc_editor)
        layout.addWidget(self.group_bc)

        # --- Начальные условия ---
        self.group_iv = QGroupBox()
        self.iv_layout = QGridLayout(self.group_iv)
        self.iv_editors: List[QLineEdit] = []
        self.iv_checkboxes: List[QCheckBox] = []
        layout.addWidget(self.group_iv)

        # --- Имена переменных ---
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Имена переменных (через запятую):"))
        self.input_varnames = QLineEdit("x, y, vx, vy")
        h2.addWidget(self.input_varnames)
        layout.addLayout(h2)

        layout.addStretch()
        self._on_dim_changed(4)

    def _on_dim_changed(self, n: int):
        """Перестраивает редакторы уравнений и начальных значений."""
        # Уравнения: скрываем и удаляем старые редакторы
        for old in self.eq_editors:
            old.hide()
            old.deleteLater()
        self.eq_editors.clear()
        while self.eq_layout.count():
            item = self.eq_layout.takeAt(0)
            if item.widget():
                w = item.widget()
                w.hide()
                w.deleteLater()
            elif item.layout():
                # Рекурсивно очищаем вложенные layout
                sub_layout = item.layout()
                while sub_layout.count():
                    sub = sub_layout.takeAt(0)
                    if sub.widget():
                        sub.widget().hide()
                        sub.widget().deleteLater()

        defaults = self._default_equations(n)
        for i in range(n):
            h = QHBoxLayout()
            h.addWidget(QLabel(f"d/dt [{i}]:"))
            ed = QLineEdit(defaults[i] if i < len(defaults) else "0")
            ed.textChanged.connect(self.changed.emit)
            self.eq_editors.append(ed)
            h.addWidget(ed)
            self.eq_layout.addLayout(h)

        # Начальные значения: скрываем и удаляем старые контролы
        for old in self.iv_editors:
            old.hide()
            old.deleteLater()
        self.iv_editors.clear()
        for old in self.iv_checkboxes:
            old.hide()
            old.deleteLater()
        self.iv_checkboxes.clear()
        while self.iv_layout.count():
            item = self.iv_layout.takeAt(0)
            if item.widget():
                w = item.widget()
                w.hide()
                w.deleteLater()
            elif item.layout():
                sub_layout = item.layout()
                while sub_layout.count():
                    sub = sub_layout.takeAt(0)
                    if sub.widget():
                        sub.widget().hide()
                        sub.widget().deleteLater()

        iv_defaults = self._default_initial_values(n)
        for i in range(n):
            chk = QCheckBox(f"x{i}(0)")
            ed = QLineEdit()
            if i < len(iv_defaults) and iv_defaults[i] is not None:
                ed.setText(str(iv_defaults[i]))
                chk.setChecked(True)
            else:
                chk.setChecked(False)
            chk.stateChanged.connect(self.changed.emit)
            ed.textChanged.connect(self.changed.emit)
            self.iv_editors.append(ed)
            self.iv_checkboxes.append(chk)
            self.iv_layout.addWidget(chk, i, 0)
            self.iv_layout.addWidget(ed, i, 1)

    def _default_equations(self, n: int) -> List[str]:
        """默认方程全部填0, 由用户手动输入."""
        return ["0"] * n

    def _default_initial_values(self, n: int) -> List[Optional[float]]:
        return [None] * n

    def get_equations(self) -> List[str]:
        return [ed.text() for ed in self.eq_editors]

    def get_boundary_conditions(self) -> List[str]:
        text = self.bc_editor.toPlainText().strip()
        if not text:
            return []
        return [line.strip() for line in text.split("\n") if line.strip()]

    def get_initial_values(self) -> Dict[int, Optional[float]]:
        result = {}
        for i, (chk, ed) in enumerate(zip(self.iv_checkboxes, self.iv_editors)):
            if chk.isChecked() and ed.text().strip():
                try:
                    result[i] = float(ed.text().strip())
                except ValueError:
                    result[i] = None
            else:
                result[i] = None
        return result

    def get_known_unknown_indices(self) -> Tuple[List[int], List[int]]:
        known = []
        unknown = []
        for i, chk in enumerate(self.iv_checkboxes):
            if chk.isChecked():
                known.append(i)
            else:
                unknown.append(i)
        return known, unknown

    def get_var_names(self) -> List[str]:
        text = self.input_varnames.text().strip()
        if text:
            names = [s.strip() for s in text.split(",")]
            n = self.spin_dim.value()
            # Дополняем автоматически
            while len(names) < n:
                names.append(f"x{len(names)}")
            return names[:n]
        return [f"x{i}" for i in range(self.spin_dim.value())]

# ---------------------------------------------------------------------------
# 7. BvpSolverApp — главное окно PyQt5
# ---------------------------------------------------------------------------


class BvpSolverApp(QMainWindow):
    """
    Универсальный GUI для решения краевых задач ODE.
    """

    TRANSLATIONS = {
        "zh": {
            "window_title": "通用BVP求解器",
            "menu_language": "语言", "lang_chinese": "中文 (Ctrl+1)",
            "lang_russian": "俄文 (Ctrl+2)",
            "menu_theme": "主题", "theme_bg": "背景颜色...",
            "theme_frame": "框架颜色...",
            "menu_help": "帮助", "help_doc": "使用说明",
            "help_about": "关于作者",
            "help_title": "使用指南", "about_title": "关于",
            "help_html": (
                "<h3>BVP 求解器 - 使用指南</h3>"
                "<p><b>1. 新建任务</b><br>点击 [+] 创建新任务或加载已保存的 JSON</p>"
                "<p><b>2. 输入 ODE</b><br>dx/dt = ... 用变量名 (x0, x1, x2, ...)</p>"
                "<p><b>3. 边界条件</b><br>每行一个: x0_T - 1.0 (x0_T = x0 at t=T)</p>"
                "<p><b>4. 初始条件</b><br>勾选=已知数值, 不勾选=未知(由求解器找)</p>"
                "<p><b>5. 初始猜测</b><br>格式: 0, 6.28 (逗号分隔, 支持 pi)</p>"
                "<p><b>6. 求解 & 绘图</b><br>[Solve] -> [Plot] 选 X/Y 轴变量</p>"
            ),
            "about_html": (
                "<h3>Программа для решения краевой задачи "
                "методом продолжения по параметру</h3>"
                "<p>Преподаватели: Аввакумов С.Н., Орлов М.С.</p>"
                "<p>Автор: Ли Хунюй 313 группа, "
                "Факультет ВМК кафедра ОУ.</p>"
                "<p>2026г.</p>"
            ),
            # 任务库
            "task_library": "任务库 / Task Library",
            "task_col_num": "#",
            "task_col_name": "名称 / Name",
            "task_col_status": "状态 / Status",
            "btn_add": "添加",
            "btn_add_tip": "添加新任务",
            "btn_del": "删除",
            "btn_del_tip": "删除选中任务",
            "btn_load": "加载",
            "btn_load_tip": "从JSON文件加载任务",
            "btn_save": "保存",
            "btn_save_tip": "保存任务到JSON文件",
            "btn_task_save": "保存",
            "btn_task_save_tip": "保存当前参数到任务",
            "task_name_placeholder": "任务名称 / Task name",
            "dialog_new_task": "新任务",
            "dialog_task_name": "任务名称 / Task name:",
            "dialog_delete": "删除确认",
            "dialog_delete_msg": "删除任务 '{name}'?",
            "dialog_validation": "参数验证错误",
            # 方程编辑器
            "tab_equations": "ODE方程组 (dx_i/dt = ...)",
            "dim_label": "系统维数 / System dimension:",
            "bc_title": "边界条件 R[x(0), x(T)] = 0",
            "bc_placeholder": "每行一个条件。变量: x0_0, x1_0, ..., x0_T, x1_T, ...\n"
                             "示例 26.1:\nx0_T - 1.0738644361\nx2_T + 1.0995343576\n"
                             "示例 26.2:\nx0_0 - x3_0\nx0_T - x3_0\nx1_T",
            "iv_title": "初始条件 (已知=数值, 未知=留空)",
            "iv_known": "x{idx}(0) 已知 / known",
            "var_names_label": "变量名 (逗号分隔):",
            "tab_params": "参数与方法",
            "progress_title": "求解进度",
            # 按钮
            "btn_solve": "求解", "btn_cancel": "取消",
            "btn_plot": "绘图",
            "btn_export": "导出", "btn_clear": "清空",

            # 参数标签
            "param_T": "终端时间 T:",
            "param_guess": "初始猜测 (p0):",
            "param_eps": "精度 eps:",
            "param_method": "积分方法:",
            "param_solver": "求解方法:",
            "param_steps": "延续步数:",
            # 求解方法选项
            "solver_cont": "参数延续 / Continuation",
            "solver_shoot": "打靶法 / Shooting",
            # 状态
            "status_ready": "就绪",
            "status_solving": "正在求解...",
            "status_cancel_requested": "正在取消...",
            "status_cancelled": "已取消",
            "status_solved": "求解完成!",
            "status_error": "求解失败",
            "status_saved": "已保存: {name}",
            "err_no_solution": "无可用的解，请先求解。",
            "confirm_clear": "确定要清空结果历史和绘图吗?",
            "solve_success": "求解成功!\n\n方法: {method}\n"
                             "迭代次数: {iter}\n"
                             "残差范数: {res_norm:.4e}\n"
                             "完整状态 x(0): {full_state}\n"
                             "未知参数: {p_opt}",
        },
        "ru": {
            "window_title": "Универсальный решатель BVP",
            "menu_language": "Язык", "lang_chinese": "Китайский (Ctrl+1)",
            "lang_russian": "Русский (Ctrl+2)",
            "menu_theme": "Тема", "theme_bg": "Цвет фона...",
            "theme_frame": "Цвет рамки...",
            "menu_help": "Помощь", "help_doc": "Руководство",
            "help_about": "Об авторе",
            "help_title": "Руководство", "about_title": "О программе",
            "help_html": (
                "<h3>BVP Solver - Руководство</h3>"
                "<p><b>1. Создание задачи</b><br>Новая [+] или загрузка из JSON</p>"
                "<p><b>2. Уравнения ODE</b><br>dx/dt = ... с именами переменных</p>"
                "<p><b>3. Граничные условия</b><br>Одно на строку: x0_T - 1.07</p>"
                "<p><b>4. Начальные условия</b><br>OK = известно, пусто = неизвестно</p>"
                "<p><b>5. Начальное приближение</b><br>Формат: 2, 0, 2*pi, 2</p>"
                "<p><b>6. Решить и построить график</b><br>[Solve] -> [Plot]</p>"
            ),
            "about_html": (
                "<h3>Программа для решения краевой задачи "
                "методом продолжения по параметру</h3>"
                "<p>Преподаватели: Аввакумов С.Н., Орлов М.С.</p>"
                "<p>Автор: Ли Хунюй 313 группа, "
                "Факультет ВМК кафедра ОУ.</p>"
                "<p>2026г.</p>"
            ),
            # Библиотека задач
            "task_library": "Библиотека задач / Task Library",
            "task_col_num": "#",
            "task_col_name": "Название / Name",
            "task_col_status": "Статус / Status",
            "btn_add": "Добавить",
            "btn_add_tip": "Добавить новую задачу",
            "btn_del": "Удалить",
            "btn_del_tip": "Удалить выбранную задачу",
            "btn_load": "Загрузить",
            "btn_load_tip": "Загрузить задачи из JSON",
            "btn_save": "Сохранить",
            "btn_save_tip": "Сохранить задачи в JSON",
            "btn_task_save": "Save",
            "btn_task_save_tip": "Сохранить текущие параметры в выбранную задачу",
            "task_name_placeholder": "Имя задачи / Task name",
            "dialog_new_task": "Новая задача",
            "dialog_task_name": "Имя задачи / Task name:",
            "dialog_delete": "Удаление",
            "dialog_delete_msg": "Удалить задачу '{name}'?",
            "dialog_validation": "Validation Error",
            # Редактор уравнений
            "tab_equations": "Уравнения ODE (dx_i/dt = ...)",
            "dim_label": "Размерность системы / System dimension:",
            "bc_title": "Граничные условия (R[x(0), x(T)] = 0)",
            "bc_placeholder": "Одно условие на строку. Переменные: x0_0, x1_0, ..., x0_T, x1_T, ...\n"
                             "Пример 26.1:\nx0_T - 1.0738644361\nx2_T + 1.0995343576\n"
                             "Пример 26.2:\nx0_0 - x2_0\nx0_T - x2_0\nx1_T",
            "iv_title": "Начальные условия (известные=число, неизвестные=пусто)",
            "iv_known": "x{idx}(0) известно / known",
            "var_names_label": "Имена переменных (через запятую):",
            "tab_params": "Параметры и метод",
            "progress_title": "Progress",
            # Кнопки
            "btn_solve": "Решить", "btn_cancel": "Отмена",
            "btn_plot": "График",
            "btn_export": "Экспорт", "btn_clear": "Очистить",

            "param_T": "Время T:",
            "param_guess": "Начальное приближение (p0):",
            "param_eps": "Точность eps:",
            "param_method": "Метод интегрирования:",
            "param_solver": "Метод решения:",
            "param_steps": "Шаги продолжения:",
            "solver_cont": "Продолжение / Continuation",
            "solver_shoot": "Стрельба / Shooting",
            "status_ready": "Готово",
            "status_solving": "Решаю...",
            "status_cancel_requested": "Отмена запрошена...",
            "status_cancelled": "Отменено",
            "status_solved": "Решение найдено!",
            "status_error": "Ошибка",
            "status_saved": "Сохранено: {name}",
            "err_no_solution": "Нет решения. Сначала нажмите 'Решить'.",
            "confirm_clear": "Очистить историю результатов и график?",
            "solve_success": "Решение найдено!\n\nМетод: {method}\n"
                             "Итераций: {iter}\n"
                             "‖Φ‖: {res_norm:.4e}\n"
                             "x(0)  = {full_state}\n"
                             "p_opt = {p_opt}",
        }
    }

    def __init__(self):
        super().__init__()
        self.current_lang = "ru"
        self.bg_color = "#f5f6fa"
        self.frame_color = "#2c3e50"
        self.panel_bg = "#ffffff"
        self.accent_color = "#2980b9"
        self.text_color = "#2c3e50"

        # ---- Библиотека задач -------------------------------------------
        self.tasks: List[Dataset] = []          # все задачи
        self._task_ids: List[str] = []          # stable GUI identities
        self.current_task_idx: int = -1          # текущая выбранная
        self._suppress_sync: bool = False       # флаг блокировки синхронизации

        # Request lifecycle and provenance-backed result history.
        self.solve_records: List[SolveRecord] = []
        self.last_record: Optional[SolveRecord] = None
        self.last_failed_record: Optional[SolveOutcome] = None
        self.last_cancelled_outcome: Optional[SolveOutcome] = None
        self.solve_state = GuiSolveState.IDLE
        self.active_request_id: Optional[str] = None
        self.active_worker: Optional[SolverWorker] = None
        self.active_cancellation_token: Optional[CancellationToken] = None
        self._processed_request_ids: set[str] = set()
        self._run_contexts: Dict[str, RunContext] = {}
        self._last_signal_diagnostic = ""
        self._closing = False
        self.close_wait_timeout_ms = 2000
        self.worker_factory = SolverWorker
        self._task_status: Dict[str, str] = {}

        self.init_ui()
        # Загружаем встроенные примеры как задачи
        self._load_builtin_tasks()
        self._refresh_task_table()
        self.update_language(self.current_lang)
        self.apply_theme()
        self._render_gui_state()

    @property
    def all_results(self) -> List[dict]:
        """Legacy boundary view; SolveRecord remains the history source of truth."""
        return [record.to_legacy_dict() for record in self.solve_records]

    @property
    def last_result(self) -> Optional[dict]:
        return None if self.last_record is None else self.last_record.to_legacy_dict()

    @property
    def last_failed_result(self) -> Optional[dict]:
        outcome = self.last_failed_record
        if outcome is None or outcome.result is None:
            return None
        return outcome.result.to_dict()

    def _load_builtin_tasks(self):
        """Built-in examples removed. User imports via Load button."""
        pass

    def _ensure_task_ids(self) -> None:
        while len(self._task_ids) < len(self.tasks):
            self._task_ids.append(uuid4().hex)
        if len(self._task_ids) > len(self.tasks):
            del self._task_ids[len(self.tasks):]

    def _task_id_for_index(self, index: int) -> Optional[str]:
        self._ensure_task_ids()
        if 0 <= index < len(self._task_ids):
            return self._task_ids[index]
        return None

    def _task_index_for_id(self, task_id: str) -> Optional[int]:
        self._ensure_task_ids()
        try:
            return self._task_ids.index(task_id)
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Управление библиотекой задач
    # ------------------------------------------------------------------

    def _refresh_task_table(self):
        """Обновляет таблицу задач."""
        self._ensure_task_ids()
        self.task_table.setRowCount(len(self.tasks))
        for i, ds in enumerate(self.tasks):
            # Номер
            item_num = QTableWidgetItem(str(i + 1))
            item_num.setFlags(item_num.flags() & ~Qt.ItemIsEditable)
            self.task_table.setItem(i, 0, item_num)
            # Имя
            item_name = QTableWidgetItem(ds.name)
            item_name.setFlags(item_name.flags() & ~Qt.ItemIsEditable)
            self.task_table.setItem(i, 1, item_name)
            # Статус
            st = self._task_status.get(self._task_ids[i], "")
            if st == "solved":
                status_text = "✓"
            elif st == "error":
                status_text = "✗"
            elif st == "cancelled":
                status_text = "–"
            else:
                status_text = ""
            item_status = QTableWidgetItem(status_text)
            item_status.setFlags(item_status.flags() & ~Qt.ItemIsEditable)
            if st == "solved":
                item_status.setBackground(QColor(39, 174, 96, 60))
                item_status.setForeground(QColor(39, 174, 96))
            elif st == "error":
                item_status.setBackground(QColor(231, 76, 60, 60))
                item_status.setForeground(QColor(231, 76, 60))
            elif st == "cancelled":
                item_status.setBackground(QColor(127, 140, 141, 50))
                item_status.setForeground(QColor(127, 140, 141))
            self.task_table.setItem(i, 2, item_status)

    def _on_task_selected(self):
        """При выборе задачи в таблице — загружаем её в редактор."""
        if self._suppress_sync:
            return
        selected = self.task_table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        if 0 <= row < len(self.tasks):
            self.current_task_idx = row
            ds = self.tasks[row]
            self.input_task_name.setText(ds.name)
            self._sync_task_to_editor(ds)

    def _sync_task_to_editor(self, ds: Dataset):
        """Загружает Dataset в редактор уравнений и параметры."""
        self._suppress_sync = True
        try:
            # --- Размерность ---
            n = ds.dim()
            self.eq_editor.spin_dim.setValue(n)
            # Перестраиваем редакторы
            self.eq_editor._on_dim_changed(n)

            # --- Уравнения ---
            for i, eq in enumerate(ds.equations):
                if i < len(self.eq_editor.eq_editors):
                    self.eq_editor.eq_editors[i].setText(eq)

            # --- Имена переменных ---
            if ds.var_names:
                self.eq_editor.input_varnames.setText(", ".join(ds.var_names))
            else:
                self.eq_editor.input_varnames.setText(", ".join(
                    [f"x{i}" for i in range(n)]
                ))

            # --- Граничные условия ---
            self.eq_editor.bc_editor.setPlainText(
                "\n".join(ds.boundary_conditions)
            )

            # --- Начальные значения ---
            for i in range(n):
                val = ds.initial_values.get(i)
                chk = self.eq_editor.iv_checkboxes[i]
                ed = self.eq_editor.iv_editors[i]
                if val is not None:
                    chk.setChecked(True)
                    ed.setText(str(val))
                else:
                    chk.setChecked(False)
                    ed.setText("")

            # --- Параметры ---
            self.input_T.setValue(ds.T)
            self.input_eps.setValue(1e-8 if ds.eps is None else ds.eps)
            self.combo_method.setCurrentText(ds.method)
            # solver_method
            idx = self.combo_solver.findData(ds.solver_method)
            if idx >= 0:
                self.combo_solver.setCurrentIndex(idx)
            self.input_steps.setValue(ds.continuation_steps)

            # --- Guess ---
            guess_str = ", ".join(str(g) for g in ds.guess)
            self.input_guess.setText(guess_str)

        finally:
            self._suppress_sync = False

    def _sync_editor_to_dataset(self, ds: Dataset) -> Dataset:
        """Собирает Dataset из текущего состояния GUI."""
        eqs = self.eq_editor.get_equations()
        bcs = self.eq_editor.get_boundary_conditions()
        iv = self.eq_editor.get_initial_values()
        known, unknown = self.eq_editor.get_known_unknown_indices()
        var_names = self.eq_editor.get_var_names()

        # Парсим guess: поддержка pi, 2π, 2pi, 2*pi и n-мерного вектора
        guess_str = self.input_guess.text().strip()
        raw_guess = []
        if guess_str:
            # Унифицируем разделители: запятые, точки с запятой
            for s in re.split(r"[,;，]", guess_str):
                s_clean = s.strip().lower()
                if not s_clean:
                    continue
                # 2π, 2pi → 2*pi (вставляем * между числом и pi)
                s_clean = re.sub(r"(\d)\s*π", r"\1*pi", s_clean)
                s_clean = re.sub(r"(\d)pi\b", r"\1*pi", s_clean)
                # π → pi
                s_clean = s_clean.replace("π", "pi")
                try:
                    raw_guess.append(float(s_clean))
                except ValueError:
                    try:
                        expr = SymPyParser.parse_scalar(s_clean)
                        raw_guess.append(float(expr.evalf()))
                    except ExpressionValidationError:
                        raise
                    except (TypeError, ValueError, OverflowError):
                        pass
        n_dim = len(eqs)
        # Если guess длины n — полный вектор, извлекаем unknown
        if len(raw_guess) == n_dim and len(unknown) < n_dim:
            guess = [raw_guess[i] for i in unknown]
        else:
            guess = raw_guess
            while len(guess) < len(unknown):
                guess.append(0.0)
            guess = guess[:len(unknown)]

        ds.name = self.input_task_name.text().strip() or ds.name
        ds.equations = eqs
        ds.var_names = var_names
        ds.T = self.input_T.value()
        ds.initial_values = iv
        ds.boundary_conditions = bcs
        ds.guess = guess
        # The basic control edits legacy datasets.  An explicit JSON/API task keeps
        # eps=None so an incomplete explicit configuration cannot be silently
        # converted into a valid-looking legacy request.
        if ds.eps is not None:
            ds.eps = self.input_eps.value()
        ds.method = self.combo_method.currentText()
        ds.solver_method = self.combo_solver.currentData()
        ds.continuation_steps = self.input_steps.value()
        ds.known_indices = known
        ds.unknown_indices = unknown
        ds.t_star = 0.0
        return ds

    def _task_add_new(self):
        """Добавляет новую пустую задачу."""
        t = self.TRANSLATIONS[self.current_lang]
        name, ok = QInputDialog.getText(
            self, t["dialog_new_task"], t["dialog_task_name"]
        )
        if not ok:
            return
        ds = Dataset(
            name=name or f"Task {len(self.tasks) + 1}",
            equations=["x1", "-x0"],
            var_names=["x0", "x1"],
            T=np.pi,
            initial_values={0: None, 1: None},
            boundary_conditions=["x0_T"],
            guess=[1.0],
            unknown_indices=[0, 1],
        )
        self.tasks.append(ds)
        self._task_ids.append(uuid4().hex)
        self._refresh_task_table()
        # Выбираем новую задачу
        self.task_table.selectRow(len(self.tasks) - 1)

    def _task_delete(self):
        """Удаляет выбранную задачу."""
        if self.current_task_idx < 0 or self.current_task_idx >= len(self.tasks):
            return
        reply = QMessageBox.question(
            self, "Удаление", f"Удалить задачу '{self.tasks[self.current_task_idx].name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            task_id = self._task_id_for_index(self.current_task_idx)
            self.tasks.pop(self.current_task_idx)
            self._task_ids.pop(self.current_task_idx)
            if task_id is not None:
                self._task_status.pop(task_id, None)
            self.current_task_idx = -1
            self._refresh_task_table()
            self.input_task_name.setText("")

    def _task_save_current(self):
        """Сохраняет текущие параметры редактора в выбранную задачу."""
        if self.current_task_idx < 0 or self.current_task_idx >= len(self.tasks):
            # Нет выбранной задачи — создаём новую
            self._task_add_new()
            return
        ds = self.tasks[self.current_task_idx]
        self._sync_editor_to_dataset(ds)
        self._refresh_task_table()
        self.progress_detail.setText(f"已保存: {ds.name}")

    def _tasks_load_json(self):
        """Загружает задачи из JSON-файла."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить задачи", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    self.tasks.append(Dataset.from_dict(item))
                    self._task_ids.append(uuid4().hex)
            elif isinstance(data, dict):
                self.tasks.append(Dataset.from_dict(data))
                self._task_ids.append(uuid4().hex)
            self._refresh_task_table()
            self.progress_detail.setText(f"已加载 {len(self.tasks)} 个任务")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки", str(e))

    def _tasks_save_json(self):
        """Сохраняет все задачи в JSON-файл."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить задачи", "tasks.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            data = [ds.to_dict() for ds in self.tasks]
            write_json_data_atomic(path, data)
            self.progress_detail.setText(f"已保存 {len(data)} 个任务")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка сохранения", str(e))

    def init_ui(self):
        self.create_menu_bar()
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # ===== Главный разделитель: Левая панель задач | Центр + Правая панель =====
        main_splitter = QSplitter(Qt.Horizontal)

        # --- ЛЕВАЯ ПАНЕЛЬ: Библиотека задач ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(8)

        # Заголовок
        self.label_task_library = QLabel()
        self.label_task_library.setStyleSheet(
            "font-weight: bold; font-size: 14px; color: #2c3e50;"
        )
        left_layout.addWidget(self.label_task_library)

        # Таблица задач
        self.task_table = QTableWidget()
        self.task_table.setColumnCount(3)
        self.task_table.setHorizontalHeaderLabels(
            ["#", "Название / Name", "Статус / Status"]
        )
        self.task_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.task_table.setSelectionMode(QTableWidget.SingleSelection)
        self.task_table.setColumnWidth(0, 35)
        self.task_table.setColumnWidth(1, 175)
        self.task_table.setColumnWidth(2, 75)
        self.task_table.setMinimumWidth(290)
        self.task_table.setMaximumWidth(310)
        self.task_table.horizontalHeader().setStretchLastSection(False)
        self.task_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.task_table.verticalHeader().setVisible(False)
        self.task_table.itemSelectionChanged.connect(self._on_task_selected)
        left_layout.addWidget(self.task_table, 1)

        # Кнопки управления задачами (2x2 grid, aligned with table)
        task_btn_layout = QGridLayout()
        task_btn_layout.setSpacing(4)
        self.btn_task_add = QPushButton("添加")
        self.btn_task_add.setToolTip("")
        self.btn_task_add.clicked.connect(self._task_add_new)
        self.btn_task_delete = QPushButton("删除")
        self.btn_task_delete.setToolTip("")
        self.btn_task_delete.clicked.connect(self._task_delete)
        self.btn_task_load = QPushButton("加载")
        self.btn_task_load.setToolTip("")
        self.btn_task_load.clicked.connect(self._tasks_load_json)
        self.btn_task_save = QPushButton("保存")
        self.btn_task_save.setToolTip("")
        self.btn_task_save.clicked.connect(self._tasks_save_json)
        task_btn_layout.addWidget(self.btn_task_add, 0, 0)
        task_btn_layout.addWidget(self.btn_task_delete, 0, 1)
        task_btn_layout.addWidget(self.btn_task_load, 1, 0)
        task_btn_layout.addWidget(self.btn_task_save, 1, 1)
        left_layout.addLayout(task_btn_layout)

        # Имя текущей задачи + кнопка сохранить
        name_layout = QHBoxLayout()
        self.input_task_name = QLineEdit()
        self.input_task_name.setPlaceholderText("")
        self.btn_task_save_current = QPushButton("Save")
        self.btn_task_save_current.setToolTip(
            "Сохранить текущие параметры в выбранную задачу"
        )
        self.btn_task_save_current.clicked.connect(self._task_save_current)
        name_layout.addWidget(self.input_task_name, 1)
        name_layout.addWidget(self.btn_task_save_current)
        left_layout.addLayout(name_layout)

        main_splitter.addWidget(left_panel)

        # --- ЦЕНТР + ПРАВАЯ ПАНЕЛЬ ---
        center_right_splitter = QSplitter(Qt.Horizontal)

        # Центр — редактор уравнений
        self.eq_editor = EquationEditor()
        self.eq_editor.changed.connect(self._on_equations_changed)
        center_right_splitter.addWidget(self.eq_editor)

        # Правая панель — параметры, прогресс, кнопки
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(10)

        # --- Параметры ---
        self.group_params = QGroupBox()
        params_grid = QGridLayout(self.group_params)
        params_grid.setVerticalSpacing(8)

        row = 0
        self.label_T = QLabel()
        self.input_T = QDoubleSpinBox()
        self.input_T.setRange(0.01, 1000)
        self.input_T.setDecimals(6)
        self.input_T.setValue(1.0)
        params_grid.addWidget(self.label_T, row, 0)
        params_grid.addWidget(self.input_T, row, 1)

        row += 1
        self.label_guess = QLabel()
        self.input_guess = QLineEdit("0")
        params_grid.addWidget(self.label_guess, row, 0)
        params_grid.addWidget(self.input_guess, row, 1)

        row += 1
        self.label_eps = QLabel()
        self.input_eps = QDoubleSpinBox()
        self.input_eps.setRange(1e-15, 1)
        self.input_eps.setDecimals(12)
        self.input_eps.setValue(1e-8)
        params_grid.addWidget(self.label_eps, row, 0)
        params_grid.addWidget(self.input_eps, row, 1)

        row += 1
        self.label_method = QLabel()
        self.combo_method = QComboBox()
        self.combo_method.addItems(
            ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]
        )
        params_grid.addWidget(self.label_method, row, 0)
        params_grid.addWidget(self.combo_method, row, 1)

        row += 1
        self.label_solver = QLabel()
        self.combo_solver = QComboBox()
        self.combo_solver.addItem("Продолжение / Continuation", "continuation")
        self.combo_solver.addItem("Стрельба / Shooting", "shooting")
        params_grid.addWidget(self.label_solver, row, 0)
        params_grid.addWidget(self.combo_solver, row, 1)

        row += 1
        self.label_steps = QLabel()
        self.input_steps = QSpinBox()
        self.input_steps.setRange(5, 500)
        self.input_steps.setValue(50)
        params_grid.addWidget(self.label_steps, row, 0)
        params_grid.addWidget(self.input_steps, row, 1)

        right_layout.addWidget(self.group_params)

        # --- Прогресс ---
        self.group_progress = QGroupBox()
        prog_layout = QVBoxLayout(self.group_progress)
        self.progress_bar_label = QLabel("Ready")
        self.progress_detail = QLabel("")
        self.progress_detail.setStyleSheet("color: #636e72; font-size: 12px;")
        prog_layout.addWidget(self.progress_bar_label)
        prog_layout.addWidget(self.progress_detail)
        right_layout.addWidget(self.group_progress)

        # --- Кнопки действий ---
        btn_layout = QHBoxLayout()
        self.btn_solve = QPushButton()
        self.btn_solve.setMinimumWidth(100)
        self.btn_solve.clicked.connect(self.on_solve)
        self.btn_cancel = QPushButton()
        self.btn_cancel.setMinimumWidth(100)
        self.btn_cancel.clicked.connect(self.on_cancel)
        self.btn_plot = QPushButton()
        self.btn_plot.setMinimumWidth(100)
        self.btn_plot.clicked.connect(self.on_plot)
        self.btn_export = QPushButton()
        self.btn_export.setMinimumWidth(100)
        self.btn_export.clicked.connect(self.on_export)
        self.btn_clear = QPushButton()
        self.btn_clear.setMinimumWidth(100)
        self.btn_clear.clicked.connect(self.on_clear)
        btn_layout.addWidget(self.btn_solve)
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_plot)
        btn_layout.addWidget(self.btn_export)
        btn_layout.addWidget(self.btn_clear)
        right_layout.addLayout(btn_layout)

        right_layout.addStretch()
        center_right_splitter.addWidget(right)
        center_right_splitter.setSizes([550, 400])
        main_splitter.addWidget(center_right_splitter)
        main_splitter.setSizes([260, 960])
        main_layout.addWidget(main_splitter, 1)

        # (状态栏已移除 — 状态通过进度标签显示)
        self.setMinimumSize(1250, 780)
        self.resize(1400, 850)
        self.setup_shortcuts()

    def create_menu_bar(self):
        menubar = self.menuBar()
        self.menu_language = menubar.addMenu("Language")
        self.action_zh = QAction("中文 (Ctrl+1)", self)
        self.action_zh.setShortcut("Ctrl+1")
        self.action_zh.triggered.connect(lambda: self.update_language("zh"))
        self.menu_language.addAction(self.action_zh)
        self.action_ru = QAction("Русский (Ctrl+2)", self)
        self.action_ru.setShortcut("Ctrl+2")
        self.action_ru.triggered.connect(lambda: self.update_language("ru"))
        self.menu_language.addAction(self.action_ru)

        self.menu_theme = menubar.addMenu("Theme")
        self.action_bg = QAction("Background Color...", self)
        self.action_bg.triggered.connect(self.pick_bg_color)
        self.menu_theme.addAction(self.action_bg)
        self.action_frame = QAction("Frame Color...", self)
        self.action_frame.triggered.connect(self.pick_frame_color)
        self.menu_theme.addAction(self.action_frame)

        self.menu_help = menubar.addMenu("Help")
        self.action_help_doc = QAction("Documentation", self)
        self.action_help_doc.triggered.connect(self.show_help)
        self.menu_help.addAction(self.action_help_doc)
        self.action_help_about = QAction("About", self)
        self.action_help_about.triggered.connect(self.show_about)
        self.menu_help.addAction(self.action_help_about)

    def show_help(self):
        t = self.TRANSLATIONS.get(self.current_lang, self.TRANSLATIONS["zh"])
        QMessageBox.information(self, t["help_title"], t["help_html"])

    def show_about(self):
        t = self.TRANSLATIONS.get(self.current_lang, self.TRANSLATIONS["zh"])
        QMessageBox.about(self, t["about_title"], t["about_html"])

    def setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+1"), self).activated.connect(
            lambda: self.update_language("zh"))
        QShortcut(QKeySequence("Ctrl+2"), self).activated.connect(
            lambda: self.update_language("ru"))

    def pick_bg_color(self):
        c = QColorDialog.getColor(QColor(self.bg_color), self, "Background")
        if c.isValid():
            self.bg_color = c.name()
            self.apply_theme()

    def pick_frame_color(self):
        c = QColorDialog.getColor(QColor(self.frame_color), self, "Frame")
        if c.isValid():
            self.frame_color = c.name()
            self.apply_theme()

    def apply_theme(self):
        style = f"""
            QMainWindow {{ background-color: {self.bg_color}; }}
            QWidget {{ background-color: {self.bg_color}; color: {self.text_color}; }}
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {self.frame_color};
                border-radius: 6px;
                margin-top: 8px;
                padding: 12px;
                background-color: {self.panel_bg};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px;
                color: {self.frame_color};
            }}
            QPushButton {{
                background-color: {self.accent_color};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 18px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self._darken(self.accent_color)};
            }}
            QPushButton:disabled {{
                background-color: #bdc3c7;
            }}
            QLineEdit, QTextEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
                padding: 5px;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                background-color: {self.panel_bg};
            }}
            QLineEdit:focus, QTextEdit:focus, QDoubleSpinBox:focus,
            QSpinBox:focus, QComboBox:focus {{
                border: 2px solid {self.accent_color};
            }}
            QLabel {{ color: {self.text_color}; }}
            QMenuBar {{
                background-color: {self.frame_color};
                color: white;
            }}
            QMenuBar::item:selected {{
                background-color: {self.accent_color};
            }}
        """
        self.setStyleSheet(style)

    def _darken(self, hex_color, factor=0.8):
        c = QColor(hex_color)
        c.setRed(int(c.red() * factor))
        c.setGreen(int(c.green() * factor))
        c.setBlue(int(c.blue() * factor))
        return c.name()

    def _build_dataset(self) -> Dataset:
        """Собирает Dataset из текущего состояния GUI.
        Использует deepcopy, чтобы не изменять оригинал в self.tasks.
        """
        import copy
        if self.current_task_idx < 0:
            ds = Dataset()
        else:
            ds = copy.deepcopy(self.tasks[self.current_task_idx])
        return self._sync_editor_to_dataset(ds)

    def _on_equations_changed(self):
        """Обновляет guess-подсказку при изменении уравнений."""
        _, unknown = self.eq_editor.get_known_unknown_indices()
        n_unknown = len(unknown)
        current = self.input_guess.text().strip()
        if not current:
            self.input_guess.setText(", ".join(["0.0"] * n_unknown)
            )

    # ------------------------------------------------------------------
    # Решение
    # ------------------------------------------------------------------

    def _set_gui_state(
        self, state: GuiSolveState, detail: Optional[str] = None
    ) -> None:
        if not gui_transition_allowed(self.solve_state, state):
            raise RuntimeError(
                f"Invalid GUI solve transition: {self.solve_state.value} -> "
                f"{state.value}"
            )
        self.solve_state = state
        if detail is not None:
            self.progress_detail.setText(detail)
        self._render_gui_state()

    def _render_gui_state(self) -> None:
        if not hasattr(self, "btn_solve"):
            return
        t = self.TRANSLATIONS[self.current_lang]
        active = self.solve_state in {
            GuiSolveState.RUNNING,
            GuiSolveState.CANCEL_REQUESTED,
        }
        self.btn_solve.setEnabled(not active)
        self.btn_cancel.setEnabled(self.solve_state is GuiSolveState.RUNNING)
        self.btn_plot.setEnabled(bool(self.solve_records))
        self.btn_export.setEnabled(self.last_record is not None)
        self.btn_clear.setEnabled(
            bool(
                self.solve_records
                or self.last_failed_record
                or self.last_cancelled_outcome
            )
        )

        # Task viewing and editing remain available because active requests are
        # immutable snapshots. Their availability is still centralized here.
        for widget in (
            self.btn_task_add,
            self.btn_task_delete,
            self.btn_task_load,
            self.btn_task_save,
            self.btn_task_save_current,
            self.task_table,
            self.eq_editor,
        ):
            widget.setEnabled(True)

        status_key = {
            GuiSolveState.IDLE: "status_ready",
            GuiSolveState.RUNNING: "status_solving",
            GuiSolveState.CANCEL_REQUESTED: "status_cancel_requested",
            GuiSolveState.COMPLETED: "status_solved",
            GuiSolveState.FAILED: "status_error",
            GuiSolveState.CANCELLED: "status_cancelled",
        }[self.solve_state]
        self.progress_bar_label.setText(t[status_key])
        if self.solve_state is GuiSolveState.COMPLETED:
            style = "background-color: #27ae60; color: white;"
        elif self.solve_state is GuiSolveState.FAILED:
            style = "background-color: #e74c3c; color: white;"
        elif self.solve_state is GuiSolveState.CANCEL_REQUESTED:
            style = "background-color: #f39c12; color: white;"
        elif self.solve_state is GuiSolveState.CANCELLED:
            style = "background-color: #7f8c8d; color: white;"
        else:
            style = ""
        if style:
            style += " padding: 4px 8px; border-radius: 4px; font-weight: bold;"
        self.progress_bar_label.setStyleSheet(style)

    def _build_solve_request(self) -> SolveRequest:
        dataset = self._build_dataset()
        errors = dataset.validate()
        if errors:
            raise ValueError("\n".join(errors))
        problem = problem_from_dataset(dataset)
        config = config_from_dataset(dataset)
        source_index = (
            self.current_task_idx
            if 0 <= self.current_task_idx < len(self.tasks)
            else None
        )
        source_task_id = (
            self._task_id_for_index(source_index)
            if source_index is not None
            else f"unsaved-{uuid4().hex}"
        )
        return SolveRequest.create(
            problem=problem,
            config=config,
            source_task_id=source_task_id,
            source_task_index=source_index,
            display_metadata={"language": self.current_lang},
        )

    def on_solve(self) -> bool:
        if self.solve_state in {
            GuiSolveState.RUNNING,
            GuiSolveState.CANCEL_REQUESTED,
        } or (self.active_worker is not None and self.active_worker.isRunning()):
            self._last_signal_diagnostic = (
                "A solve request is already active; a second worker was not started."
            )
            self.progress_detail.setText(self._last_signal_diagnostic)
            LOGGER.info(self._last_signal_diagnostic)
            return False

        try:
            request = self._build_solve_request()
        except Exception as exc:
            QMessageBox.warning(self, "Validation Error", str(exc))
            self.progress_detail.setText(str(exc))
            self._render_gui_state()
            return False
        return self._start_solve_request(request)

    def _start_solve_request(self, request: SolveRequest) -> bool:
        if self.active_worker is not None and self.active_worker.isRunning():
            return False
        token = CancellationToken()
        worker = self.worker_factory(request, token)
        worker.request_started.connect(self._on_request_started)
        worker.progress.connect(self._on_progress)
        worker.request_finished.connect(self._on_solve_done)
        worker.request_failed.connect(self._on_solve_error)
        worker.request_cancelled.connect(self._on_solve_cancelled)
        worker.finished.connect(
            lambda request_id=request.request_id, target=worker:
            self._on_worker_thread_finished(request_id, target)
        )
        self.active_request_id = request.request_id
        self.active_cancellation_token = token
        self.active_worker = worker
        context = getattr(worker, "run_context", None)
        if isinstance(context, RunContext):
            self._run_contexts[request.request_id] = context
        self._set_gui_state(
            GuiSolveState.RUNNING,
            f"request_id={request.request_id} task={request.task_name}",
        )
        LOGGER.info("Solve request %s started for %s", request.request_id, request.task_name)
        worker.start()
        return True

    def on_cancel(self) -> bool:
        if (
            self.solve_state is not GuiSolveState.RUNNING
            or self.active_request_id is None
            or self.active_cancellation_token is None
        ):
            return False
        request_id = self.active_request_id
        if self.active_worker is not None and hasattr(
            self.active_worker, "request_cancel"
        ):
            self.active_worker.request_cancel()
        else:
            self.active_cancellation_token.cancel()
            context = self._run_contexts.get(request_id)
            if context is not None:
                emit_solver_event(
                    context,
                    "cancel_requested",
                    phase="cancellation",
                    status="requested",
                )
        self._set_gui_state(
            GuiSolveState.CANCEL_REQUESTED,
            f"Cancellation requested for {request_id}; waiting for a checkpoint.",
        )
        LOGGER.info("Cancellation requested for solve %s", request_id)
        return True

    def _on_request_started(self, request_id: str) -> None:
        LOGGER.info("Worker acknowledged solve request %s", request_id)

    def _on_progress(
        self, request_id: str, method: str, percent: int, message: str
    ) -> None:
        if request_id != self.active_request_id or self._closing:
            context = self._run_contexts.get(request_id)
            if context is not None:
                emit_solver_event(
                    context,
                    "stale_signal_ignored",
                    phase="gui",
                    status="ignored",
                    details={"signal": "progress"},
                    level=logging.WARNING,
                )
            LOGGER.info("Ignored stale progress signal for request %s", request_id)
            return
        self.progress_bar_label.setText(f"[{percent}%] {method}")
        self.progress_detail.setText(message)

    def _accept_terminal_signal(self, request_id: str) -> bool:
        if request_id in self._processed_request_ids:
            self._last_signal_diagnostic = (
                f"Ignored duplicate terminal signal for request {request_id}."
            )
            LOGGER.warning(self._last_signal_diagnostic)
            context = self._run_contexts.get(request_id)
            if context is not None:
                emit_solver_event(
                    context,
                    "duplicate_signal_ignored",
                    phase="gui",
                    status="ignored",
                    details={"signal": "terminal"},
                    level=logging.WARNING,
                )
            return False
        if request_id != self.active_request_id:
            self._processed_request_ids.add(request_id)
            self._last_signal_diagnostic = (
                f"Ignored stale terminal signal for request {request_id}; "
                f"active_request_id={self.active_request_id}."
            )
            LOGGER.warning(self._last_signal_diagnostic)
            context = self._run_contexts.get(request_id)
            if context is not None:
                emit_solver_event(
                    context,
                    "stale_signal_ignored",
                    phase="gui",
                    status="ignored",
                    details={"signal": "terminal"},
                    level=logging.WARNING,
                )
            return False
        self._processed_request_ids.add(request_id)
        return True

    def _release_active_request(self, request_id: str) -> None:
        if self.active_request_id == request_id:
            self.active_request_id = None
            self.active_cancellation_token = None

    def _set_request_task_status(self, request: SolveRequest, status: str) -> None:
        if self._task_index_for_id(request.source_task_id) is None:
            return
        self._task_status[request.source_task_id] = status
        self._refresh_task_table()

    def _on_solve_done(self, request_id: str, outcome: SolveOutcome) -> None:
        if self._closing or not self._accept_terminal_signal(request_id):
            return
        if (
            outcome.request_id != request_id
            or outcome.status is not SolveOutcomeStatus.COMPLETED
            or outcome.result is None
        ):
            raise RuntimeError("Worker emitted an invalid completed outcome")

        if (
            self.active_cancellation_token is not None
            and self.active_cancellation_token.is_cancelled()
        ):
            result_metadata = outcome.result.run_metadata
            cancelled_metadata = (
                build_run_metadata(
                    metadata_context(result_metadata),
                    outcome.request.config,
                    boundary_count=len(outcome.request.problem.boundary_conditions),
                    final_status="cancelled",
                    elapsed_seconds=result_metadata.elapsed_seconds,
                    solver_metadata=outcome.result.solver_metadata,
                    result_data=outcome.result.to_dict(),
                )
                if result_metadata is not None
                else outcome.run_metadata
            )
            cancelled = SolveOutcome(
                request=outcome.request,
                status=SolveOutcomeStatus.CANCELLED,
                message="Cancellation was observed before auxiliary output processing.",
                run_metadata=cancelled_metadata,
            )
            context = self._run_contexts.get(request_id)
            if context is not None:
                emit_solver_event(
                    context,
                    "solve_cancelled",
                    phase="solve",
                    status="cancelled",
                    details={"checkpoint": "before_auxiliary_outputs"},
                )
            self.last_cancelled_outcome = cancelled
            self._set_request_task_status(outcome.request, "cancelled")
            self._release_active_request(request_id)
            self._set_gui_state(
                GuiSolveState.CANCELLED,
                f"request_id={request_id}: {cancelled.message}",
            )
            return

        aux_data, aux_t, aux_errors = self._compute_aux_outputs(
            outcome.request, outcome.result
        )
        record = SolveRecord(
            request=outcome.request,
            result=outcome.result,
            auxiliary_outputs=aux_data,
            auxiliary_sample_t=aux_t,
            auxiliary_errors=aux_errors,
            completed_at=outcome.completed_at,
        )
        self.solve_records.append(record)
        self.last_record = record
        self._set_request_task_status(outcome.request, "solved")
        self._release_active_request(request_id)

        result = outcome.result
        full_state = outcome.request.build_initial_state(result.p_opt)
        full_str = np.array2string(full_state, precision=6, separator=", ")
        p_opt_str = np.array2string(result.p_opt, precision=6, separator=", ")
        message = self.TRANSLATIONS[self.current_lang]["solve_success"].format(
            method=result.method,
            iter=result.iterations,
            res_norm=result.residual_norm,
            p_opt=p_opt_str,
            full_state=full_str,
        )
        mode = result.solver_metadata.get("tolerance_mode", "unknown")
        run_id_short = (
            result.run_metadata.run_id[:8] if result.run_metadata else "unknown"
        )
        elapsed_seconds = (
            result.run_metadata.elapsed_seconds if result.run_metadata else 0.0
        )
        detail = (
            f"run_id={run_id_short}; "
            f"request_id={request_id[:8]}; problem={outcome.request.task_name}; "
            f"method={result.method}; status={result.status}; "
            f"‖Φ‖ = {result.residual_norm:.4e}; "
            f"max_scaled_ratio={result.boundary_max_scaled_ratio:.4e}; "
            f"tolerance_mode={mode}; elapsed_seconds={elapsed_seconds:.6f}"
        )
        if aux_errors:
            detail += "; auxiliary warnings: " + " | ".join(aux_errors)
        self._set_gui_state(GuiSolveState.COMPLETED, detail)
        LOGGER.info("Solve request %s completed", request_id)
        QMessageBox.information(self, "Result", message)

    def _on_solve_error(self, request_id: str, outcome: SolveOutcome) -> None:
        if self._closing or not self._accept_terminal_signal(request_id):
            return
        if outcome.request_id != request_id or outcome.status is not SolveOutcomeStatus.FAILED:
            raise RuntimeError("Worker emitted an invalid failed outcome")
        self.last_failed_record = outcome
        self._set_request_task_status(outcome.request, "error")
        self._release_active_request(request_id)
        if outcome.result is not None:
            diagnostic = self._format_failed_result(outcome.result)
            QMessageBox.warning(self, "BVP result rejected", diagnostic)
        else:
            diagnostic = outcome.message
            QMessageBox.critical(self, "Solve error", diagnostic)
        if outcome.technical_diagnostic:
            LOGGER.debug(
                "Technical diagnostic for request %s:\n%s",
                request_id,
                outcome.technical_diagnostic,
            )
        self._set_gui_state(GuiSolveState.FAILED, diagnostic)

    def _on_solve_cancelled(self, request_id: str, outcome: SolveOutcome) -> None:
        if self._closing or not self._accept_terminal_signal(request_id):
            return
        if outcome.request_id != request_id or outcome.status is not SolveOutcomeStatus.CANCELLED:
            raise RuntimeError("Worker emitted an invalid cancelled outcome")
        self.last_cancelled_outcome = outcome
        self._set_request_task_status(outcome.request, "cancelled")
        self._release_active_request(request_id)
        self._set_gui_state(
            GuiSolveState.CANCELLED,
            f"run_id={outcome.run_id[:8] if outcome.run_id else 'unknown'}; "
            f"request_id={request_id[:8]}; status=cancelled; {outcome.message}",
        )
        LOGGER.info("Solve request %s cancelled", request_id)

    def _on_worker_thread_finished(
        self, request_id: str, worker: SolverWorker
    ) -> None:
        LOGGER.info("Worker thread finished for request %s", request_id)
        if self.active_worker is worker:
            self.active_worker = None

    @staticmethod
    def _format_failed_result(result: BVPResult | dict) -> str:
        if isinstance(result, BVPResult):
            result = result.to_dict()
        residual = np.asarray(result.get("boundary_residual", []), dtype=float)
        residual_text = np.array2string(residual, precision=6, separator=", ")
        residual_norm = result.get("boundary_residual_norm", float("inf"))
        boundary_atol = result.get("boundary_atol", "unknown")
        boundary_rtol = result.get("boundary_rtol", "unknown")
        thresholds = np.asarray(result.get("boundary_thresholds", []), dtype=float)
        component_success = np.asarray(
            result.get("boundary_component_success", []), dtype=bool
        )
        failed_components = np.flatnonzero(~component_success).tolist()
        maximum_ratio = result.get("boundary_max_scaled_ratio", float("inf"))
        metadata = result.get("solver_metadata", {})
        run_metadata = result.get("run_metadata") or {}
        diagnostic_keys = (
            "tolerance_mode",
            "legacy_eps",
            "effective_ivp_rtol",
            "effective_ivp_atol",
            "effective_root_tol",
            "effective_least_squares_ftol",
            "effective_least_squares_xtol",
            "effective_least_squares_gtol",
            "effective_continuation_residual_tol",
            "optimizer",
            "fallback_used",
            "root",
            "least_squares",
            "failure_step",
            "failure_reason",
        )
        metadata_summary = {
            key: metadata[key] for key in diagnostic_keys if key in metadata
        }
        return (
            f"method: {result.get('method', 'unknown')}\n"
            f"run_id: {run_metadata.get('run_id', 'unknown')}\n"
            f"request_id: {run_metadata.get('request_id', 'unknown')}\n"
            f"elapsed_seconds: {run_metadata.get('elapsed_seconds', 'unknown')}\n"
            f"status: {result.get('status', 'unknown')}\n"
            f"message: {result.get('message', 'No diagnostic message')}\n"
            f"optimizer_success: {result.get('optimizer_success', False)}\n"
            f"ivp_success: {result.get('ivp_success', False)}\n"
            f"finite_success: {result.get('finite_success', False)}\n"
            f"boundary_success: {result.get('boundary_success', False)}\n"
            f"boundary_residual: {residual_text}\n"
            f"boundary_residual_norm: {residual_norm}\n"
            f"boundary_atol: {boundary_atol}\n"
            f"boundary_rtol: {boundary_rtol}\n"
            f"boundary_thresholds: {thresholds.tolist()}\n"
            f"boundary_max_scaled_ratio: {maximum_ratio}\n"
            f"failed_boundary_components: {failed_components}\n"
            f"solver_metadata: {metadata_summary}"
        )

    @staticmethod
    def _build_full_state(
        request: SolveRequest, p_opt: np.ndarray
    ) -> np.ndarray:
        """Build x(t_start) exclusively from the originating request."""
        return request.build_initial_state(p_opt)

    def _compute_aux_outputs(
        self, request: SolveRequest, result: BVPResult
    ) -> tuple[dict[str, np.ndarray], np.ndarray, tuple[str, ...]]:
        """Compute auxiliary outputs exclusively from the request snapshot.

        Each expression failure is retained as a diagnostic. A valid primary
        solution is never discarded because one auxiliary expression failed.
        """
        problem = request.problem
        has_lunula_outputs = (
            "26.4" in problem.name or "Лунка" in problem.name
        ) and len(problem.var_names) >= 5
        if not problem.auxiliary_expressions and not has_lunula_outputs:
            return {}, np.array([], dtype=float), ()

        errors: list[str] = []
        try:
            parser = SymPyParser(list(problem.odes), list(problem.var_names))
            parser.parse()
            parser.lambdify_all()
            sol = result.sol
            if sol is None:
                raise ValueError("dense IVP solution is unavailable")
            t0, t1 = float(sol.t[0]), float(sol.t[-1])
            td = np.linspace(t0, t1, 500)
            if hasattr(sol, "sol") and callable(sol.sol):
                yd = sol.sol(td)
            else:
                yd = np.array(
                    [
                        np.interp(td, sol.t, sol.y[j])
                        for j in range(len(problem.var_names))
                    ]
                )
        except Exception as exc:
            diagnostic = (
                "auxiliary setup failed for request "
                f"{request.request_id}: {type(exc).__name__}: {exc}"
            )
            LOGGER.warning(diagnostic)
            context = self._run_contexts.get(request.request_id)
            if context is not None:
                emit_solver_event(
                    context,
                    "auxiliary_output_failed",
                    phase="auxiliary",
                    status="failed",
                    details={
                        "name": None,
                        "error_type": type(exc).__name__,
                        "reason": str(exc),
                    },
                    level=logging.WARNING,
                )
            return {}, np.array([], dtype=float), (diagnostic,)

        aux_data: dict[str, np.ndarray] = {}
        if has_lunula_outputs:
            try:
                u1_values = []
                u2_values = []
                for index, time_value in enumerate(td):
                    f_values = parser.f(time_value, yd[:, index])
                    time_scale = yd[4, index]
                    if abs(time_scale) > 1e-12:
                        u1 = f_values[0] / time_scale - yd[1, index]
                        u2 = (
                            f_values[1] / time_scale
                            + 1.5 * yd[0, index]
                            + 0.25 * yd[1, index]
                        )
                    else:
                        u1 = u2 = 0.0
                    u1_values.append(u1)
                    u2_values.append(u2)
                aux_data["u1"] = np.asarray(u1_values, dtype=float)
                aux_data["u2"] = np.asarray(u2_values, dtype=float)
            except Exception as exc:
                diagnostic = f"auxiliary 'u1/u2' failed: {type(exc).__name__}: {exc}"
                errors.append(diagnostic)
                LOGGER.warning("Request %s: %s", request.request_id, diagnostic)
                context = self._run_contexts.get(request.request_id)
                if context is not None:
                    emit_solver_event(
                        context,
                        "auxiliary_output_failed",
                        phase="auxiliary",
                        status="failed",
                        details={
                            "name": "u1/u2",
                            "error_type": type(exc).__name__,
                            "reason": str(exc),
                        },
                        level=logging.WARNING,
                    )

        for name, expression in problem.auxiliary_expressions.items():
            if name in aux_data:
                continue
            try:
                function = parser.lambdify_aux({name: expression})[name]
                values = np.array(
                    [
                        function(time_value, *yd[:, index])
                        for index, time_value in enumerate(td)
                    ],
                    dtype=float,
                ).reshape(-1)
                if values.size != td.size or not np.isfinite(values).all():
                    raise ValueError("values are non-finite or have an invalid shape")
                aux_data[name] = values
            except Exception as exc:
                diagnostic = (
                    f"auxiliary {name!r} failed: {type(exc).__name__}: {exc}"
                )
                errors.append(diagnostic)
                LOGGER.warning("Request %s: %s", request.request_id, diagnostic)
                context = self._run_contexts.get(request.request_id)
                if context is not None:
                    emit_solver_event(
                        context,
                        "auxiliary_output_failed",
                        phase="auxiliary",
                        status="failed",
                        details={
                            "name": name,
                            "error_type": type(exc).__name__,
                            "reason": str(exc),
                        },
                        level=logging.WARNING,
                    )

        return aux_data, td, tuple(errors)

    # ------------------------------------------------------------------
    # Построение графиков
    # ------------------------------------------------------------------

    def on_plot(self):
        t = self.TRANSLATIONS[self.current_lang]
        if not self.solve_records or self.last_record is None:
            QMessageBox.warning(self, "Warning", t["err_no_solution"])
            return

        compatible_records, rejected = partition_plot_records(
            self.solve_records, self.last_record
        )
        if rejected:
            summary = "; ".join(
                f"{record.request.task_name}[{record.request_id[:8]}]: {reason}"
                for record, reason in rejected
            )
            diagnostic = (
                f"Plot isolation kept {len(compatible_records)} compatible "
                f"record(s) and excluded {len(rejected)}: {summary}"
            )
            self.progress_detail.setText(diagnostic)
            LOGGER.warning(diagnostic)
        if hasattr(self, '_plot_widget') and self._plot_widget is not None:
            try:
                self._plot_widget.update_data(
                    list(compatible_records), lang=self.current_lang
                )
                self._plot_widget.show()
                self._plot_widget.raise_()
                self._plot_widget.activateWindow()
                return
            except RuntimeError:
                self._plot_widget = None
        self._plot_widget = IntegratedPlotWidget(
            list(compatible_records), lang=self.current_lang, parent=self
        )
        self._plot_widget.show()
        self._plot_widget.raise_()
        self._plot_widget.activateWindow()

    def _do_plot(self, all_results: List[dict], cfg: dict, var_names: List[str]):
        """
        Построение графика.
        Фазовый портрет: square figure + equal aspect (без tight_layout!).
        多Y叠加: 每个solution × 每个y_var 画一条曲线.
        """
        x_axis = cfg["x_axis"]
        y_axes = cfg.get("y_axes", [cfg.get("y_axis", var_names[0])])
        # 相图模式只取第一个Y
        if x_axis != "t":
            y_axes = [y_axes[0]]
        grid = cfg["grid"]
        equal = cfg["equal_aspect"]
        lw = cfg.get("line_width", 1.5)
        show_markers = cfg.get("show_markers", True)
        visible = set(cfg.get("visible_indices", list(range(len(all_results)))))

        colors = [(31/255,119/255,180/255),(255/255,127/255,14/255),
                  (44/255,160/255,44/255),(214/255,39/255,40/255),
                  (148/255,103/255,189/255),(140/255,86/255,75/255),
                  (227/255,119/255,194/255),(127/255,127/255,127/255),
                  (188/255,189/255,34/255),(23/255,190/255,207/255)]
        linestyles = ["-", "--", ":", "-."]

        is_phase = (x_axis != "t" and y_axes[0] != "t")

        def axis_data(name, t_vals, y_vals):
            if name == "t":
                return t_vals, "t"
            return y_vals[var_names.index(name)], name

        # ---- 1. Собираем данные ----
        # curves: (sol_idx, y_var_name, x_arr, y_arr)
        curves = []
        plot_diagnostics = []
        for i, r in enumerate(all_results):
            if i not in visible:
                continue
            try:
                sol = r["sol"]
                t0, t1 = float(sol.t[0]), float(sol.t[-1])
                td = np.linspace(t0, t1, 500)
                if hasattr(sol, 'sol') and callable(sol.sol):
                    yd = sol.sol(td)
                else:
                    yd = np.array([np.interp(td, sol.t, sol.y[j])
                                   for j in range(len(var_names))])
                xd, xl = axis_data(x_axis, td, yd)
                for y_name in y_axes:
                    try:
                        yd_arr, yl = axis_data(y_name, td, yd)
                        curves.append((i, y_name, np.asarray(xd), np.asarray(yd_arr)))
                    except (ValueError, IndexError, TypeError) as exc:
                        plot_diagnostics.append(
                            f"solution {i + 1}, axis {y_name!r}: "
                            f"{type(exc).__name__}: {exc}"
                        )
            except (KeyError, ValueError, IndexError, TypeError, AttributeError) as exc:
                plot_diagnostics.append(
                    f"solution {i + 1}: {type(exc).__name__}: {exc}"
                )

        if plot_diagnostics:
            diagnostic = " | ".join(plot_diagnostics)
            self.progress_detail.setText(diagnostic)
            LOGGER.warning("Legacy plot diagnostic: %s", diagnostic)

        if not curves:
            QMessageBox.warning(self, "Plot", "Нет данных")
            return

        # Диапазон
        all_x = np.concatenate([c[2] for c in curves])
        all_y = np.concatenate([c[3] for c in curves])
        x_min, x_max = float(all_x.min()), float(all_x.max())
        y_min, y_max = float(all_y.min()), float(all_y.max())
        x_pad = 0.1 * max(x_max - x_min, 1e-12)
        y_pad = 0.1 * max(y_max - y_min, 1e-12)

        # ---- 2. Figure ----
        figsize = (7, 7) if is_phase else (8, 6)
        fig, ax = plt.subplots(figsize=figsize)

        # Подписи
        _, x_lbl = axis_data(x_axis, None, np.zeros(len(var_names)))
        ax.set_xlabel(f"$\\mathit{{{x_lbl}}}$", fontsize=13)

        if is_phase:
            y_lbl = y_axes[0]
            title = f"${y_lbl}$ vs ${x_lbl}$"
            ax.set_ylabel(f"$\\mathit{{{y_lbl}}}$", fontsize=13)
        elif x_axis == "t":
            # 时间图: Y轴标注选中的变量名
            y_label_str = ", ".join(y_axes)
            ax.set_ylabel(f"$\\mathit{{{y_label_str}}}$", fontsize=12)
            if len(y_axes) == 1:
                title = f"${y_axes[0]}(t)$"
            else:
                title = "状态变量 $x_i(t)$"
        else:
            y_lbl = y_axes[0]
            title = f"${x_lbl}(t)$"
            ax.set_ylabel(f"$\\mathit{{{y_lbl}}}$", fontsize=13)
        ax.set_title(title, fontsize=14)

        # ---- 3. Рисуем (多Y叠加) ----
        # 颜色按 y_var 分配, 线型按 solution 分配
        n_y = len(y_axes)
        n_sol = len(set(c[0] for c in curves))
        for sol_idx, y_name, xd, yd in curves:
            y_idx = y_axes.index(y_name) if y_name in y_axes else 0
            c = colors[y_idx % len(colors)]
            ls = linestyles[sol_idx % len(linestyles)] if n_sol > 1 else "-"
            label = f"{y_name}" if n_sol == 1 else f"{y_name} #{sol_idx + 1}"
            ax.plot(xd, yd, ls=ls, color=c, lw=lw, label=label)
            if show_markers and is_phase:
                ax.plot(xd[0], yd[0], "o", color=c, ms=7,
                        mec="black", mew=0.5)
                ax.plot(xd[-1], yd[-1], "s", color=c, ms=7,
                        mec="black", mew=0.5)

        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        if grid:
            ax.grid(True, alpha=0.3, ls="--")

        # 相图交叉检测 (只取每条solution的第一个y_var)
        if is_phase and n_sol >= 2:
            phase_curves = [(c[0], c[2], c[3]) for c in curves if c[1] == y_axes[0]]
            if len(phase_curves) >= 2:
                self._draw_phase_with_crossing(phase_curves, ax, colors)

        # S/F 标记 (相图模式)
        if is_phase:
            sf_offsets = [(10, 8), (-14, -12), (10, -14), (-14, 8)]
            ff_offsets = [(10, -12), (-14, 8), (10, 10), (-14, -14)]
            seen_sol = set()
            for sol_idx, y_name, xd, yd in curves:
                if y_name != y_axes[0] or sol_idx in seen_sol:
                    continue
                seen_sol.add(sol_idx)
                y_idx = y_axes.index(y_name) if y_name in y_axes else 0
                c = colors[y_idx % len(colors)]
                ax.plot(xd[0], yd[0], "o", color=c, ms=7, mec="black", mew=0.5)
                ax.plot(xd[-1], yd[-1], "s", color=c, ms=7, mec="black", mew=0.5)
                idx = len(seen_sol) - 1
                ax.annotate("S", (xd[0], yd[0]), textcoords="offset points",
                           xytext=sf_offsets[idx % 4], fontsize=13,
                           fontweight="bold", color=c)
                ax.annotate("F", (xd[-1], yd[-1]), textcoords="offset points",
                           xytext=ff_offsets[idx % 4], fontsize=13,
                           fontweight="bold", color=c)

        ax.legend(loc="best", fontsize=11, framealpha=0.9)
        ax.tick_params(labelsize=11)

        # ---- 4. Финализация ----
        if is_phase and equal:
            ax.set_aspect("equal")
            x_c, y_c = (x_min + x_max) / 2, (y_min + y_max) / 2
            half = max(x_max - x_min, y_max - y_min) / 2 + max(x_pad, y_pad)
            ax.set_xlim(x_c - half, x_c + half)
            ax.set_ylim(y_c - half, y_c + half)
        else:
            fig.tight_layout()

        try:
            plt.show(block=False)
            plt.pause(0.001)
        except Exception as exc:
            LOGGER.warning("Non-blocking plot display failed: %s", exc)
            plt.show()

    def _draw_phase_with_crossing(self, curves, ax, colors):
        """
        Стиль методички рис. 26.1: пересечения — пунктир, снаружи — сплошная.
        """
        from scipy.spatial import cKDTree

        def find_crossing_segments(xa, ya, xb, yb, tol=0.08):
            pts_b = np.column_stack([xb, yb])
            tree = cKDTree(pts_b)
            dists, _ = tree.query(np.column_stack([xa, ya]), k=1)
            cross = dists < tol
            segs = []
            in_cross, start = False, 0
            for j in range(len(cross)):
                if cross[j] and not in_cross:
                    in_cross = True
                    if start < j:
                        segs.append((start, j, False))
                    start = j
                elif not cross[j] and in_cross:
                    in_cross = False
                    segs.append((start, j, True))
                    start = j
            if start < len(cross):
                segs.append((start, len(cross), in_cross))
            return segs

        # Рисуем каждую кривую с учётом пересечений
        for idx_a, (i_a, xa, ya) in enumerate(curves):
            segs = []
            for idx_b, (i_b, xb, yb) in enumerate(curves):
                if idx_a == idx_b:
                    continue
                segs_b = find_crossing_segments(xa, ya, xb, yb)
                segs.extend(segs_b)
            # Объединяем все пересечения
            if not segs:
                ax.plot(xa, ya, "-", color=colors[i_a % len(colors)], lw=1.5)
                continue
            # Маркируем точки как "в пересечении" если любая пара говорит да
            is_cross = np.zeros(len(xa), dtype=bool)
            for s, e, cross in segs:
                if cross:
                    is_cross[s:e] = True
            # Разбиваем на сегменты
            final_segs = []
            in_c, start = False, 0
            for j in range(len(is_cross)):
                if is_cross[j] and not in_c:
                    in_c = True
                    if start < j:
                        final_segs.append((start, j, False))
                    start = j
                elif not is_cross[j] and in_c:
                    in_c = False
                    final_segs.append((start, j, True))
                    start = j
            if start < len(is_cross):
                final_segs.append((start, len(is_cross), in_c))
            # Рисуем
            c = colors[i_a % len(colors)]
            for s, e, cross in final_segs:
                ls = "--" if cross else "-"
                if e - s > 1:
                    ax.plot(xa[s:e], ya[s:e], ls=ls, color=c, lw=1.5)

    # ------------------------------------------------------------------
    # Экспорт
    # ------------------------------------------------------------------

    def on_export(self):
        t = self.TRANSLATIONS[self.current_lang]
        if self.last_record is None:
            QMessageBox.warning(self, "Warning", t["err_no_solution"])
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export", "result.json", "JSON (*.json);;Text (*.txt)"
        )
        if not path:
            return

        record = self.last_record
        context = (
            metadata_context(record.result.run_metadata)
            if record.result.run_metadata is not None
            else self._run_contexts.get(record.request_id)
        )
        if context is not None:
            emit_solver_event(
                context,
                "export_started",
                phase="export",
                status="running",
                details={"format": "json" if path.lower().endswith(".json") else "text"},
            )
        try:
            canonical = build_canonical_export_record(record)
            export_canonical_record(path, canonical)
        except Exception as exc:
            if context is not None:
                emit_solver_event(
                    context,
                    "export_failed",
                    phase="export",
                    status="failed",
                    details={"error_type": type(exc).__name__, "reason": str(exc)},
                    level=logging.ERROR,
                )
            QMessageBox.critical(self, "Export failed", f"{type(exc).__name__}: {exc}")
            return
        if context is not None:
            emit_solver_event(
                context,
                "export_succeeded",
                phase="export",
                status="completed",
                details={"format": "json" if path.lower().endswith(".json") else "text"},
            )
        self.progress_detail.setText(f"已导出: {path}")

    def on_clear(self):
        t = self.TRANSLATIONS[self.current_lang]
        reply = QMessageBox.question(self, "Confirm", t["confirm_clear"])
        if reply == QMessageBox.Yes:
            self.last_record = None
            self.last_failed_record = None
            self.last_cancelled_outcome = None
            self.solve_records.clear()
            self._task_status.clear()
            self._run_contexts.clear()
            self._refresh_task_table()
            if hasattr(self, "_plot_widget") and self._plot_widget is not None:
                try:
                    self._plot_widget.close()
                except RuntimeError:
                    LOGGER.info("Plot widget was already destroyed during clear")
                self._plot_widget = None
            self.progress_detail.setText("")
            if self.solve_state not in {
                GuiSolveState.RUNNING,
                GuiSolveState.CANCEL_REQUESTED,
            }:
                self._set_gui_state(GuiSolveState.IDLE)
            else:
                self._render_gui_state()

    # ------------------------------------------------------------------
    # Локализация
    # ------------------------------------------------------------------

    def update_language(self, lang: str):
        self.current_lang = lang
        t = self.TRANSLATIONS[lang]
        self.setWindowTitle(t["window_title"])
        self.menu_language.setTitle(t["menu_language"])
        self.menu_theme.setTitle(t["menu_theme"])
        self.action_zh.setText(t["lang_chinese"])
        self.action_ru.setText(t["lang_russian"])
        self.action_bg.setText(t["theme_bg"])
        self.action_frame.setText(t["theme_frame"])

        self.menu_help.setTitle(t["menu_help"])
        self.action_help_doc.setText(t["help_doc"])
        self.action_help_about.setText(t["help_about"])

        # ---- Task library ----
        self.label_task_library.setText(t["task_library"])
        self.task_table.setHorizontalHeaderLabels(
            [t["task_col_num"], t["task_col_name"], t["task_col_status"]]
        )
        self.btn_task_add.setText(t["btn_add"])
        self.btn_task_add.setToolTip(t["btn_add_tip"])
        self.btn_task_delete.setText(t["btn_del"])
        self.btn_task_delete.setToolTip(t["btn_del_tip"])
        self.btn_task_load.setText(t["btn_load"])
        self.btn_task_load.setToolTip(t["btn_load_tip"])
        self.btn_task_save.setText(t["btn_save"])
        self.btn_task_save.setToolTip(t["btn_save_tip"])
        self.btn_task_save_current.setText(t["btn_task_save"])
        self.btn_task_save_current.setToolTip(t["btn_task_save_tip"])
        self.input_task_name.setPlaceholderText(t["task_name_placeholder"])

        # ---- Equation editor ----
        self.eq_editor.group_eq.setTitle(t["tab_equations"])
        self.eq_editor.group_bc.setTitle(t["bc_title"])
        self.eq_editor.group_iv.setTitle(t["iv_title"])
        self.eq_editor.bc_editor.setPlaceholderText(t["bc_placeholder"])
        # Обновляем подписи чекбоксов начальных условий
        for i, chk in enumerate(self.eq_editor.iv_checkboxes):
            chk.setText(t["iv_known"].format(idx=i))
        # Обновляем метку "Размерность системы"
        # (первая метка в layout — обновляем через поиск)
        dim_label = self.eq_editor.layout().itemAt(0).layout().itemAt(0).widget()
        if isinstance(dim_label, QLabel):
            dim_label.setText(t["dim_label"])
        # Обновляем метку "Имена переменных"
        # Находим в layout последний элемент перед stretch
        for i in range(self.eq_editor.layout().count()):
            item = self.eq_editor.layout().itemAt(i)
            if item and item.layout() and item.layout().count() >= 2:
                w0 = item.layout().itemAt(0).widget()
                w1 = item.layout().itemAt(1).widget()
                if isinstance(w0, QLabel) and isinstance(w1, QLineEdit) and w1 == self.eq_editor.input_varnames:
                    w0.setText(t["var_names_label"])

        # ---- Parameters ----
        self.group_params.setTitle(t["tab_params"])
        self.label_T.setText(t["param_T"])
        self.label_guess.setText(t["param_guess"])
        self.label_eps.setText(t["param_eps"])
        self.label_method.setText(t["param_method"])
        self.label_solver.setText(t["param_solver"])
        self.label_steps.setText(t["param_steps"])

        # Обновляем тексты в combo_solver
        current_data = self.combo_solver.currentData()
        self.combo_solver.clear()
        self.combo_solver.addItem(t["solver_cont"], "continuation")
        self.combo_solver.addItem(t["solver_shoot"], "shooting")
        idx = self.combo_solver.findData(current_data)
        if idx >= 0:
            self.combo_solver.setCurrentIndex(idx)

        # ---- Progress ----
        self.group_progress.setTitle(t["progress_title"])

        # ---- Action buttons ----
        self.btn_solve.setText(t["btn_solve"])
        self.btn_cancel.setText(t["btn_cancel"])
        self.btn_plot.setText(t["btn_plot"])
        self.btn_export.setText(t["btn_export"])
        self.btn_clear.setText(t["btn_clear"])

        # (状态栏已移除)

        # ---- 同步进度标签语言和集中状态 ----
        self._render_gui_state()

        # ---- 同步更新绘图窗口语言 ----
        if hasattr(self, '_plot_widget') and self._plot_widget is not None:
            try:
                self._plot_widget.set_lang(lang)
            except RuntimeError:
                self._plot_widget = None

    def closeEvent(self, event) -> None:
        """Cooperatively cancel and join the active worker before destruction."""
        self._closing = True
        worker = self.active_worker
        if worker is not None and worker.isRunning():
            if hasattr(worker, "request_cancel"):
                worker.request_cancel()
            elif self.active_cancellation_token is not None:
                self.active_cancellation_token.cancel()
            if self.solve_state is GuiSolveState.RUNNING:
                self._set_gui_state(
                    GuiSolveState.CANCEL_REQUESTED,
                    "Window close requested cancellation; waiting for worker exit.",
                )
            LOGGER.info(
                "Window close is waiting for request %s",
                self.active_request_id,
            )
            if not worker.wait(self.close_wait_timeout_ms):
                diagnostic = (
                    "Window close wait timed out; the worker remains active and the "
                    "window was not destroyed."
                )
                LOGGER.warning(diagnostic)
                context = self._run_contexts.get(self.active_request_id or "")
                if context is not None:
                    emit_solver_event(
                        context,
                        "window_close_wait_timeout",
                        phase="gui",
                        status="timeout",
                        details={"timeout_ms": self.close_wait_timeout_ms},
                        level=logging.WARNING,
                    )
                self.progress_detail.setText(diagnostic)
                self._closing = False
                event.ignore()
                return

        self.active_worker = None
        self.active_request_id = None
        self.active_cancellation_token = None
        event.accept()


# ---------------------------------------------------------------------------
# 8. Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = BvpSolverApp()
    window.show()
    sys.exit(app.exec_())
