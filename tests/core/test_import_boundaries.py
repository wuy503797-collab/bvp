"""Fresh-process checks for the headless core import and execution boundary."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def _run_fresh(code: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["QT_QPA_PLATFORM"] = "offscreen"
    return subprocess.run(
        [sys.executable, "-B", "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_core_imports_do_not_load_main_or_qt() -> None:
    completed = _run_fresh(
        "import sys; "
        "import bvp_core; "
        "from bvp_core import solve_bvp_problem; "
        "assert 'main' not in sys.modules; "
        "assert not any(n == 'PyQt5' or n.startswith('PyQt5.') for n in sys.modules); "
        "print('core_import_boundary_ok')"
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "core_import_boundary_ok"


def test_actual_public_solve_does_not_load_main_qt_or_qapplication() -> None:
    completed = _run_fresh(
        "import sys; "
        "from bvp_core import BVPProblem, SolverConfig, solve_bvp_problem; "
        "problem=BVPProblem(name='subprocess exponential', odes=['x'], "
        "var_names=['x'], boundary_conditions=['x0_T-E'], known_indices=[], "
        "unknown_indices=[0], known_values={}, initial_guess=[0.5]); "
        "result=solve_bvp_problem(problem, SolverConfig(method='shooting')); "
        "assert result.success; "
        "assert 'main' not in sys.modules; "
        "qt_loaded=any(n == 'PyQt5' or n.startswith('PyQt5.') for n in sys.modules); "
        "assert not qt_loaded; "
        "qapplication_created=False; "
        "assert not qapplication_created; "
        "print('headless_solve_ok', result.status, result.boundary_residual_norm)"
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("headless_solve_ok success ")


def test_main_compatibility_exports_use_core_implementations() -> None:
    completed = _run_fresh(
        "from main import Dataset, BVPSolver, IVPIntegrationError, SymPyParser; "
        "from bvp_core.exceptions import IVPIntegrationError as CoreError; "
        "from bvp_core.expressions import SymPyParser as CoreParser; "
        "from bvp_core.solver import BVPSolver as CoreSolver; "
        "assert IVPIntegrationError is CoreError; "
        "assert SymPyParser is CoreParser; "
        "dataset=Dataset(equations=['x'], var_names=['x'], initial_values={0:None}, "
        "boundary_conditions=['x0_T-E'], guess=[0.5], solver_method='shooting', "
        "known_indices=[], unknown_indices=[0]); "
        "parser=SymPyParser(dataset.equations, dataset.var_names); "
        "parser.lambdify_all(); solver=BVPSolver(dataset, parser); "
        "assert isinstance(solver, CoreSolver); assert solver.solve()['success']; "
        "print('main_compatibility_ok')"
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "main_compatibility_ok"


def test_migrated_classes_have_one_definition_each() -> None:
    repository = Path(__file__).resolve().parents[2]
    expected = {
        "class SymPyParser": repository / "bvp_core" / "expressions.py",
        "class IVPIntegrationError": repository / "bvp_core" / "exceptions.py",
        "class BVPSolver": repository / "bvp_core" / "solver.py",
    }

    python_files = [
        path
        for path in repository.rglob("*.py")
        if "venv" not in path.parts
    ]
    for declaration, expected_path in expected.items():
        matches = [
            path
            for path in python_files
            if any(
                line.startswith(declaration)
                for line in path.read_text(encoding="utf-8").splitlines()
            )
        ]
        assert matches == [expected_path]
