"""Compatibility coverage for every persisted task expression."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from bvp_core.expressions import SymPyParser
from main import Dataset
from task_asset_paths import (
    EXAMPLE_TASK_FILES,
    OVERDETERMINED_FIXTURE_PATH,
    PROJECT_ROOT,
    SINGULAR_IVP_FIXTURE_PATH,
    TASK_FIXTURE_FILES,
)


TASK_FILES = (*EXAMPLE_TASK_FILES, *TASK_FIXTURE_FILES)


@pytest.mark.parametrize(
    "task_path",
    TASK_FILES,
    ids=lambda path: path.relative_to(PROJECT_ROOT).as_posix(),
)
def test_every_task_file_uses_the_restricted_expression_language(
    task_path: Path,
) -> None:
    raw_tasks = json.loads(task_path.read_text(encoding="utf-8"))

    for raw_task in raw_tasks:
        dataset = Dataset.from_dict(raw_task)
        parser = SymPyParser(dataset.equations, dataset.var_names)
        parser.lambdify_all()
        SymPyParser.parse_boundary_conditions(
            dataset.boundary_conditions, dataset.var_names
        )
        parser.lambdify_aux(dataset.aux_outputs)


def test_expression_security_does_not_hide_existing_model_validation_errors() -> None:
    raw_task = json.loads(
        OVERDETERMINED_FIXTURE_PATH.read_text(encoding="utf-8")
    )[0]
    overdetermined = Dataset.from_dict(raw_task)

    errors = overdetermined.validate()

    assert any("boundary" in error.lower() for error in errors)


def test_numerically_bad_fixture_remains_syntactically_safe() -> None:
    raw_task = json.loads(
        SINGULAR_IVP_FIXTURE_PATH.read_text(encoding="utf-8")
    )[0]

    assert Dataset.from_dict(raw_task).validate() == []
