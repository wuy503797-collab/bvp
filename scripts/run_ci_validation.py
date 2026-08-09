"""Run deterministic phase-eleven checks through one CI-friendly entry point."""

from __future__ import annotations

from importlib import metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


REPOSITORY = Path(__file__).resolve().parents[1]
DEPENDENCIES = ("numpy", "scipy", "sympy", "matplotlib", "PyQt5", "pytest")


def _run(label: str, arguments: list[str], *, environment: dict[str, str]) -> bool:
    print(f"\n[{label}]", flush=True)
    completed = subprocess.run(
        arguments,
        cwd=REPOSITORY,
        env=environment,
        check=False,
    )
    print(f"{label}_exit_code={completed.returncode}", flush=True)
    return completed.returncode == 0


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("[environment]")
    print(f"python={platform.python_version()}")
    print(f"system={platform.system()}")
    print(f"machine={platform.machine()}")
    for dependency in DEPENDENCIES:
        print(f"{dependency}={_version(dependency)}")

    with tempfile.TemporaryDirectory(prefix="bvp-phase11-ci-") as temporary:
        temporary_path = Path(temporary)
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONUTF8": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPYCACHEPREFIX": str(temporary_path / "pycache"),
                "QT_QPA_PLATFORM": "offscreen",
            }
        )
        python = sys.executable
        checks = {
            "dependency_check_passed": _run(
                "pip_check", [python, "-m", "pip", "check"], environment=environment
            ),
            "warning_check_passed": all(
                (
                    _run(
                        "warning_import_bvp_core",
                        [python, "-W", "error", "-c", "import bvp_core"],
                        environment=environment,
                    ),
                    _run(
                        "warning_import_main",
                        [python, "-W", "error", "-c", "import main"],
                        environment=environment,
                    ),
                )
            ),
            "compile_check_passed": _run(
                "compileall",
                [python, "-m", "compileall", "-q", "bvp_core"],
                environment=environment,
            ),
            "pytest_passed": _run(
                "pytest",
                [
                    python,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    f"--basetemp={temporary_path / 'pytest'}",
                ],
                environment=environment,
            ),
            "baseline_passed": _run(
                "baseline",
                [python, "scripts/run_baseline.py"],
                environment=environment,
            ),
            "numerical_validation_passed": _run(
                "numerical_validation",
                [python, "scripts/run_numerical_validation.py"],
                environment=environment,
            ),
            "expression_security_passed": _run(
                "expression_security",
                [python, "scripts/run_expression_security.py"],
                environment=environment,
            ),
            "tolerance_validation_passed": _run(
                "tolerance_validation",
                [python, "scripts/run_tolerance_validation.py"],
                environment=environment,
            ),
            "observability_validation_passed": _run(
                "observability_validation",
                [python, "scripts/run_observability_validation.py"],
                environment=environment,
            ),
            "performance_validation_passed": _run(
                "performance_validation",
                [python, "scripts/run_performance_validation.py", "--ci"],
                environment=environment,
            ),
            "repository_hygiene_passed": _run(
                "repository_hygiene",
                [python, "scripts/check_repository_hygiene.py"],
                environment=environment,
            ),
        }

    print("\n[summary]")
    for name, passed in checks.items():
        print(f"{name}={'PASS' if passed else 'FAIL'}")
    print(f"tests_passed={'PASS' if checks['pytest_passed'] else 'FAIL'}")
    passed = all(checks.values())
    print(f"phase_eleven_local_status={'PASS' if passed else 'FAIL'}")
    release_status = (
        "PENDING_REMOTE_CI_AND_LICENSE"
        if passed
        else "BLOCKED_LOCAL_VALIDATION"
    )
    print(f"release_readiness_status={release_status}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
