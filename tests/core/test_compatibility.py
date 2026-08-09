"""Compatibility tests between legacy Dataset and public core models."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from bvp_core.adapters import config_from_dataset, dataset_kwargs, problem_from_dataset
from main import Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_legacy_dataset_round_trip_preserves_problem_and_solver_controls() -> None:
    task = json.loads((REPO_ROOT / "task1.json").read_text(encoding="utf-8"))[0]
    dataset = Dataset.from_dict(task)

    problem = problem_from_dataset(dataset)
    config = config_from_dataset(dataset)
    rebuilt = Dataset(**dataset_kwargs(problem, config))

    assert rebuilt.to_dict() == dataset.to_dict()
    assert problem.odes == tuple(dataset.equations)
    assert problem.known_values == {0: 2.0, 1: 0.0}
    assert config.method == "continuation"
    assert config.ivp_rtol == dataset.eps
    assert config.ivp_atol == dataset.eps / 10.0


def test_dataset_validate_delegates_dimension_checks_to_core_model() -> None:
    dataset = Dataset(
        equations=["x1", "-x0"],
        var_names=["x0", "x1"],
        initial_values={0: 0.0, 1: None},
        boundary_conditions=["x0_T", "x1_T"],
        guess=[1.0],
        known_indices=[0],
        unknown_indices=[1],
        solver_method="shooting",
    )

    errors = dataset.validate()
    assert len(errors) == 1
    assert "boundary condition count is 2" in errors[0]
    assert "unknown initial parameter count is 1" in errors[0]
