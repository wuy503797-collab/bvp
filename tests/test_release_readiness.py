"""Repository-level contracts for dependencies, CI, and release hygiene."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from scripts.check_repository_hygiene import find_environment_files, scan_secret_lines


REPOSITORY = Path(__file__).resolve().parents[1]


def _requirement_lines(name: str) -> list[str]:
    return [
        line.strip()
        for line in (REPOSITORY / name).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _requirement_name(line: str) -> str:
    for marker in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        line = line.split(marker, 1)[0]
    return line.strip().lower().replace("_", "-")


def test_runtime_and_development_requirements_have_distinct_roles() -> None:
    runtime = _requirement_lines("requirements.txt")
    development = _requirement_lines("requirements-dev.txt")
    runtime_names = {_requirement_name(line) for line in runtime}

    assert runtime_names == {"numpy", "scipy", "sympy", "matplotlib", "pyqt5"}
    assert "pytest" not in runtime_names
    assert "-r requirements.txt" in development
    assert any(_requirement_name(line) == "pytest" for line in development)


def test_reference_requirements_use_exact_valid_syntax() -> None:
    locked = _requirement_lines("requirements-lock.txt")
    assert {_requirement_name(line) for line in locked} == {
        "numpy",
        "scipy",
        "sympy",
        "matplotlib",
        "pyqt5",
        "pytest",
    }
    assert all(line.count("==") == 1 for line in locked)
    assert all(line.split("==", 1)[0] and line.split("==", 1)[1] for line in locked)


def test_documented_install_and_policy_files_exist() -> None:
    readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    for name in (
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-lock.txt",
        "CONTRIBUTING.md",
        "SECURITY.md",
    ):
        assert name in readme
        assert (REPOSITORY / name).is_file()


def test_workflow_covers_supported_platforms_without_hidden_failures() -> None:
    workflow = (REPOSITORY / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "push:" in workflow
    assert "pull_request:" in workflow
    assert "windows-latest" in workflow
    assert "ubuntu-latest" in workflow
    assert 'python-version: "3.11"' in workflow
    assert "scripts/run_ci_validation.py" in workflow
    assert "tests/gui" not in workflow
    assert "continue-on-error" not in workflow
    assert "${{ secrets." not in workflow


def test_ci_validation_declares_every_required_summary_field() -> None:
    source = (REPOSITORY / "scripts/run_ci_validation.py").read_text(encoding="utf-8")
    for field in (
        "dependency_check_passed",
        "warning_check_passed",
        "compile_check_passed",
        "pytest_passed",
        "baseline_passed",
        "numerical_validation_passed",
        "expression_security_passed",
        "tolerance_validation_passed",
        "observability_validation_passed",
        "performance_validation_passed",
        "repository_hygiene_passed",
        "phase_eleven_local_status",
        "release_readiness_status",
        "PENDING_REMOTE_CI_AND_LICENSE",
    ):
        assert field in source
    assert "phase_eleven_ci_validation_status" not in source
    assert 'phase_eleven_status=' not in source


def test_performance_ci_mode_keeps_deterministic_checks() -> None:
    source = (REPOSITORY / "scripts/run_performance_validation.py").read_text(
        encoding="utf-8"
    )
    assert '"--ci"' in source
    assert "numerical_equivalence_passed" in source
    assert "counter_consistency_passed" in source
    assert "performance_regression_check_passed" in source
    assert "30%" not in source


def test_secret_scanner_reports_location_without_secret_value(tmp_path: Path) -> None:
    value = "fictional-value-for-test-only"
    sample = "pass" + "word" + ' = "' + value + '"'
    findings = scan_secret_lines(Path("fixture.txt"), sample)
    (tmp_path / ".env").write_text(sample, encoding="utf-8")

    assert [(finding.path, finding.line) for finding in findings] == [
        ("fixture.txt", 1)
    ]
    assert all(value not in repr(finding) for finding in findings)
    assert find_environment_files(tmp_path) == [".env"]


def test_repository_hygiene_script_passes_without_mutating_files() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/check_repository_hygiene.py"],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "tracked_environment_files=0" in completed.stdout
    assert "tracked_cache_files=0" in completed.stdout
    assert "suspicious_secret_files=0" in completed.stdout
    assert "broken_readme_paths=0" in completed.stdout
    assert "repository_hygiene_status=PASS" in completed.stdout
