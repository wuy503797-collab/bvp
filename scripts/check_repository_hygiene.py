"""Read-only repository hygiene and release-readiness checks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable
from urllib.parse import unquote


REPOSITORY = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
}
ENVIRONMENT_PARTS = {"venv", ".venv"}
CACHE_PARTS = {"__pycache__", ".pytest_cache", "htmlcov"}
TEMPORARY_SUFFIXES = {".bak", ".orig", ".swp", ".swo", ".tmp", "~"}
LARGE_FILE_BYTES = 1_000_000
README_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
LOCAL_WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")
LOCAL_POSIX_PATH = re.compile(r"/(?:Users|home)/[^/\s]+/")
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    re.compile(
        r"(?i)\b(?:password|secret|token)\s*=\s*['\"][^'\"]{8,}['\"]"
    ),
)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int


def _git_paths(*arguments: str) -> list[str]:
    completed = subprocess.run(
        ["git", *arguments, "-z"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    )
    return [
        item.decode("utf-8", errors="surrogateescape")
        for item in completed.stdout.split(b"\0")
        if item
    ]


def _tracked_paths() -> list[str]:
    return _git_paths("ls-files")


def _visible_worktree_paths() -> list[str]:
    return _git_paths("ls-files", "--cached", "--others", "--exclude-standard")


def find_environment_files(repository: Path) -> list[str]:
    """Find .env files even when Git ignore rules intentionally hide them."""
    found: list[str] = []
    for current, directories, files in os.walk(repository):
        directories[:] = [
            name
            for name in directories
            if name not in {".git", "venv", ".venv", "__pycache__", ".pytest_cache"}
            and not name.startswith(".venv-")
        ]
        if ".env" in files:
            found.append((Path(current) / ".env").relative_to(repository).as_posix())
    return sorted(found)


def _is_text_path(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in {
        ".gitignore",
        "LICENSE",
    }


def scan_secret_lines(path: Path, text: str) -> list[Finding]:
    """Return locations only; never retain or print the matching secret text."""
    findings: list[Finding] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if any(pattern.search(line) for pattern in SECRET_PATTERNS):
            findings.append(Finding(path.as_posix(), number))
    return findings


def _secret_findings(paths: Iterable[str]) -> list[Finding]:
    findings: list[Finding] = []
    for relative in paths:
        path = REPOSITORY / relative
        if path.name == ".env":
            findings.append(Finding(relative, 0))
            continue
        if not path.is_file() or not _is_text_path(path):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        findings.extend(scan_secret_lines(Path(relative), text))
    return findings


def _readme_broken_paths() -> list[str]:
    readme = REPOSITORY / "README.md"
    if not readme.exists():
        return ["README.md"]
    broken: list[str] = []
    for raw_target in README_LINK.findall(readme.read_text(encoding="utf-8")):
        target = raw_target.strip().strip("<>").split(maxsplit=1)[0]
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        target = unquote(target.split("#", 1)[0].split("?", 1)[0])
        if target and not (REPOSITORY / target).exists():
            broken.append(target)
    return sorted(set(broken))


def _case_mismatches(paths: Iterable[str]) -> list[str]:
    mismatches: list[str] = []
    for relative in paths:
        current = REPOSITORY
        for part in Path(relative).parts:
            if not current.is_dir():
                break
            names = {entry.name.casefold(): entry.name for entry in current.iterdir()}
            actual = names.get(part.casefold())
            if actual is not None and actual != part:
                mismatches.append(relative)
                break
            current /= part
    return sorted(set(mismatches))


def _local_path_findings(paths: Iterable[str]) -> list[Finding]:
    findings: list[Finding] = []
    for relative in paths:
        normalized = Path(relative).as_posix()
        if not (
            normalized.startswith("scripts/")
            or normalized.startswith(".github/")
            or normalized in {"requirements.txt", "requirements-dev.txt", "requirements-lock.txt"}
        ):
            continue
        path = REPOSITORY / relative
        if not path.is_file() or not _is_text_path(path):
            continue
        for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            if LOCAL_WINDOWS_PATH.search(line) or LOCAL_POSIX_PATH.search(line):
                findings.append(Finding(relative, number))
    return findings


def _print_locations(label: str, findings: Iterable[Finding]) -> None:
    for finding in findings:
        suffix = "" if finding.line == 0 else f":{finding.line}"
        print(f"{label}={finding.path}{suffix}")


def _working_tree_clean() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return not completed.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-clean-working-tree",
        action="store_true",
        help="also fail when Git reports tracked or untracked changes",
    )
    arguments = parser.parse_args(argv)

    tracked = _tracked_paths()
    visible = sorted(set(_visible_worktree_paths()) | set(find_environment_files(REPOSITORY)))
    tracked_environment = [
        path
        for path in tracked
        if any(part in ENVIRONMENT_PARTS or part.startswith(".venv-") for part in Path(path).parts)
    ]
    tracked_cache = [
        path
        for path in tracked
        if any(part in CACHE_PARTS for part in Path(path).parts)
        or Path(path).name == ".coverage"
    ]
    tracked_temporary = [
        path
        for path in tracked
        if Path(path).suffix.lower() in TEMPORARY_SUFFIXES
        or Path(path).name.endswith("~")
        or path.startswith("baseline-output/")
    ]
    suspicious = _secret_findings(visible)
    broken_readme = _readme_broken_paths()
    case_mismatches = _case_mismatches(tracked)
    local_paths = _local_path_findings(visible)
    large_files = [
        path
        for path in tracked
        if (REPOSITORY / path).is_file()
        and (REPOSITORY / path).stat().st_size >= LARGE_FILE_BYTES
    ]
    retained_pdfs = sorted(path.name for path in REPOSITORY.glob("*.pdf"))
    clean = _working_tree_clean()

    print(f"tracked_environment_files={len(tracked_environment)}")
    print(f"tracked_cache_files={len(tracked_cache)}")
    print(f"tracked_temporary_files={len(tracked_temporary)}")
    print(f"suspicious_secret_files={len({item.path for item in suspicious})}")
    print(f"suspicious_secret_locations={len(suspicious)}")
    print(f"broken_readme_paths={len(broken_readme)}")
    print(f"case_mismatched_paths={len(case_mismatches)}")
    print(f"local_absolute_path_locations={len(local_paths)}")
    print(f"large_files={len(large_files)}")
    print(f"retained_pdf_materials={len(retained_pdfs)}")
    print(f"working_tree_clean={clean}")
    print("legacy_file=gui.py classification=obsolete-placeholder pending-separate-cleanup")
    print("legacy_file=solver.py classification=compatibility active-26.1-baseline")
    print("legacy_file=task1.json classification=example active-baseline")
    print("legacy_file=tasks.json classification=example active-library")
    print("legacy_file=tasks2.json classification=example compatibility-fixture")
    print("legacy_file=tasks3.json classification=example compatibility-fixture")
    print("legacy_file=tasks4.json classification=example compatibility-fixture")
    print("legacy_file=task5.json classification=example compatibility-fixture")
    print("legacy_file=task_error_test.json classification=negative-test-data")
    for name in retained_pdfs:
        print(f"retained_material={name} classification=intentional-project-material")
    for path in broken_readme:
        print(f"broken_readme_path={path}")
    for path in case_mismatches:
        print(f"case_mismatched_path={path}")
    _print_locations("suspicious_secret_location", suspicious)
    _print_locations("local_absolute_path_location", local_paths)

    checks = (
        not tracked_environment,
        not tracked_cache,
        not tracked_temporary,
        not suspicious,
        not broken_readme,
        not case_mismatches,
        not local_paths,
        not large_files,
        clean if arguments.require_clean_working_tree else True,
    )
    passed = all(checks)
    print(f"repository_hygiene_passed={passed}")
    print(f"repository_hygiene_status={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
