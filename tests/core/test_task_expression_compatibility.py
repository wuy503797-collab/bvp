"""Compatibility coverage for every persisted task expression."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from bvp_core.expressions import SymPyParser
from main import Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_FILES = (
    "tasks.json",
    "task1.json",
    "tasks2.json",
    "tasks3.json",
    "tasks4.json",
    "task5.json",
    "task_error_test.json",
)


@pytest.mark.parametrize("filename", TASK_FILES)
def test_every_task_file_uses_the_restricted_expression_language(filename: str) -> None:
    raw_tasks = json.loads((REPO_ROOT / filename).read_text(encoding="utf-8"))

    for raw_task in raw_tasks:
        dataset = Dataset.from_dict(raw_task)
        parser = SymPyParser(dataset.equations, dataset.var_names)
        parser.lambdify_all()
        SymPyParser.parse_boundary_conditions(
            dataset.boundary_conditions, dataset.var_names
        )
        parser.lambdify_aux(dataset.aux_outputs)


def test_expression_security_does_not_hide_existing_model_validation_errors() -> None:
    tasks = json.loads((REPO_ROOT / "tasks.json").read_text(encoding="utf-8"))
    overdetermined = Dataset.from_dict(tasks[4])

    errors = overdetermined.validate()

    assert any("boundary" in error.lower() for error in errors)


def test_numerically_bad_fixture_remains_syntactically_safe() -> None:
    raw_task = json.loads(
        (REPO_ROOT / "task_error_test.json").read_text(encoding="utf-8")
    )[0]

    assert Dataset.from_dict(raw_task).validate() == []
