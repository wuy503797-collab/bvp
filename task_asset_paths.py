"""Repository-relative locations for user task examples and test fixtures."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
EXAMPLE_TASKS_DIR = PROJECT_ROOT / "examples" / "tasks"
TASK_FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "tasks"

LEGACY_ROOT_TASK_FILENAMES = frozenset(
    {
        "tasks.json",
        "task1.json",
        "tasks2.json",
        "tasks3.json",
        "tasks4.json",
        "task5.json",
        "task_error_test.json",
    }
)

TEXTBOOK_PROBLEMS_PATH = EXAMPLE_TASKS_DIR / "textbook_problems.json"
SINGULAR_IVP_FIXTURE_PATH = TASK_FIXTURES_DIR / "singular_ivp.json"
OVERDETERMINED_FIXTURE_PATH = TASK_FIXTURES_DIR / "overdetermined.json"
EXAMPLE_TASK_FILES = (
    TEXTBOOK_PROBLEMS_PATH,
    EXAMPLE_TASKS_DIR / "26_1_two_body.json",
    EXAMPLE_TASKS_DIR / "26_2_limit_cycle.json",
    EXAMPLE_TASKS_DIR / "26_3_triple_integrator.json",
    EXAMPLE_TASKS_DIR / "26_4_lunula.json",
    EXAMPLE_TASKS_DIR / "26_5_time_optimal.json",
)
TASK_FIXTURE_FILES = (
    SINGULAR_IVP_FIXTURE_PATH,
    OVERDETERMINED_FIXTURE_PATH,
)
