"""Contracts for the classified task JSON asset layout introduced in Phase 13."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from bvp_core.expressions import SymPyParser
from main import Dataset
from task_asset_paths import (
    EXAMPLE_TASKS_DIR,
    LEGACY_ROOT_TASK_FILENAMES,
    PROJECT_ROOT,
    TASK_FIXTURES_DIR,
    TEXTBOOK_PROBLEMS_PATH,
)


EXPECTED_EXAMPLE_FILENAMES = {
    "textbook_problems.json",
    "26_1_two_body.json",
    "26_2_limit_cycle.json",
    "26_3_triple_integrator.json",
    "26_4_lunula.json",
    "26_5_time_optimal.json",
}
EXPECTED_FIXTURE_FILENAMES = {"singular_ivp.json", "overdetermined.json"}
EXPECTED_FILE_FINGERPRINTS = {
    "examples/tasks/textbook_problems.json": (
        "dff4fd71d5e63ea997ca1a169858d5f8339d79643515d0554ecf9f61a23a4c7e"
    ),
    "examples/tasks/26_1_two_body.json": (
        "4107004fd69667c9bcc902b29572867250907e50c48af00e7202f3998bd4827e"
    ),
    "examples/tasks/26_2_limit_cycle.json": (
        "ae1d5b199fb132de83ef72c005349b8640af36fa54f7078ddf6051547d529b93"
    ),
    "examples/tasks/26_3_triple_integrator.json": (
        "94fc069eb2dadd03322423e77f01aa35cc8f928b3d96d327cc8df473343c0a57"
    ),
    "examples/tasks/26_4_lunula.json": (
        "3aa2b2b3c6e034e6348a2f007179f177c6655865cefc67b15972433268b31afb"
    ),
    "examples/tasks/26_5_time_optimal.json": (
        "f9bb125ef04abcc7234b5c84aa02aafa4318ca4786c90a041a6684b9df00350c"
    ),
    "tests/fixtures/tasks/singular_ivp.json": (
        "5c62164459746335e943e0abb8a273356734608bec10dd6658e11ce44370870d"
    ),
    "tests/fixtures/tasks/overdetermined.json": (
        "b3b85531249b7c6bc52e1ce0f55acc44ef0f109915a06b63e012ef4a039837a1"
    ),
}
SINGLE_EXAMPLE_NAMES = {
    "26_1_two_body.json": "26.1",
    "26_2_limit_cycle.json": "26.2",
    "26_3_triple_integrator.json": "26.3",
    "26_4_lunula.json": "26.4",
    "26_5_time_optimal.json": "26.5",
}


def _load(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    return data


def _fingerprint(data: object) -> str:
    canonical = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def test_task_asset_directories_and_exact_file_sets_exist() -> None:
    assert EXAMPLE_TASKS_DIR.is_dir()
    assert TASK_FIXTURES_DIR.is_dir()
    assert {path.name for path in EXAMPLE_TASKS_DIR.glob("*.json")} == (
        EXPECTED_EXAMPLE_FILENAMES
    )
    assert {path.name for path in TASK_FIXTURES_DIR.glob("*.json")} == (
        EXPECTED_FIXTURE_FILENAMES
    )


def test_legacy_root_task_assets_are_absent() -> None:
    assert not {
        name for name in LEGACY_ROOT_TASK_FILENAMES if (PROJECT_ROOT / name).exists()
    }


@pytest.mark.parametrize(
    ("relative_path", "expected_fingerprint"),
    EXPECTED_FILE_FINGERPRINTS.items(),
)
def test_migrated_json_matches_pre_migration_canonical_fingerprint(
    relative_path: str,
    expected_fingerprint: str,
) -> None:
    assert _fingerprint(_load(PROJECT_ROOT / relative_path)) == expected_fingerprint


def test_textbook_library_contains_only_normal_course_examples() -> None:
    tasks = _load(TEXTBOOK_PROBLEMS_PATH)
    assert [task["name"].split()[0] for task in tasks] == [
        "26.1",
        "26.2",
        "26.3",
        "26.4",
    ]
    assert all(len(task["boundary_conditions"]) == len(task["unknown_indices"]) for task in tasks)


@pytest.mark.parametrize(("filename", "number"), SINGLE_EXAMPLE_NAMES.items())
def test_single_example_has_expected_identity_and_valid_semantics(
    filename: str,
    number: str,
) -> None:
    raw_tasks = _load(EXAMPLE_TASKS_DIR / filename)
    assert len(raw_tasks) == 1
    raw_task = raw_tasks[0]
    assert raw_task["name"].startswith(number)

    dataset = Dataset.from_dict(raw_task)
    assert dataset.dim() == len(dataset.var_names) > 0
    assert len(dataset.boundary_conditions) == dataset.num_unknown()
    assert dataset.validate() == []

    parser = SymPyParser(dataset.equations, dataset.var_names)
    parser.lambdify_all()
    SymPyParser.parse_boundary_conditions(
        dataset.boundary_conditions,
        dataset.var_names,
    )
    parser.lambdify_aux(dataset.aux_outputs)


def test_single_26_1_example_equals_first_textbook_entry() -> None:
    standalone = _load(EXAMPLE_TASKS_DIR / "26_1_two_body.json")[0]
    textbook = _load(TEXTBOOK_PROBLEMS_PATH)[0]
    assert standalone == textbook


def test_overdetermined_fixture_is_direct_and_semantically_invalid() -> None:
    raw_tasks = _load(TASK_FIXTURES_DIR / "overdetermined.json")
    assert len(raw_tasks) == 1
    dataset = Dataset.from_dict(raw_tasks[0])
    assert len(dataset.boundary_conditions) == 2
    assert dataset.num_unknown() == 1
    assert any("boundary" in error.lower() for error in dataset.validate())


def test_singular_fixture_remains_structurally_and_expression_valid() -> None:
    raw_tasks = _load(TASK_FIXTURES_DIR / "singular_ivp.json")
    assert len(raw_tasks) == 1
    dataset = Dataset.from_dict(raw_tasks[0])
    assert dataset.validate() == []


def test_readme_points_to_existing_classified_task_assets() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    for relative_path in (
        "examples/tasks/26_1_two_body.json",
        "examples/tasks/textbook_problems.json",
        "examples/tasks/",
        "tests/fixtures/tasks/",
    ):
        assert relative_path in readme
        assert (PROJECT_ROOT / relative_path).exists()


def test_active_production_scripts_and_tests_have_no_legacy_asset_dependency() -> None:
    exempt = {
        PROJECT_ROOT / "scripts" / "check_repository_hygiene.py",
        Path(__file__).resolve(),
    }
    active_files = [PROJECT_ROOT / "main.py"]
    active_files.extend((PROJECT_ROOT / "scripts").glob("*.py"))
    active_files.extend((PROJECT_ROOT / "tests").rglob("*.py"))
    for path in active_files:
        if path in exempt:
            continue
        text = path.read_text(encoding="utf-8")
        assert not LEGACY_ROOT_TASK_FILENAMES.intersection(text.split())
        for legacy_name in LEGACY_ROOT_TASK_FILENAMES:
            assert f'"{legacy_name}"' not in text
            assert f"'{legacy_name}'" not in text
