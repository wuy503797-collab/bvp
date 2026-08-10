"""Check structural and factual consistency of the zh-CN/ru-RU docs.

This checker deliberately does not judge Russian grammar or translation quality.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote


REPOSITORY = Path(__file__).resolve().parents[1]
COVERAGE_PATH = REPOSITORY / "docs" / "i18n" / "coverage.json"
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
NUMBERED_H2 = re.compile(r"^##\s+(\d+)\.\s+", re.MULTILINE)
SHA = re.compile(r"^[0-9a-f]{7,40}$")
FORBIDDEN_PLACEHOLDERS = (
    "TODO" + " TRANSLATE",
    "TBD" + " TRANSLATION",
    "待" + "翻译",
)
REQUIRED_RUSSIAN_TERMS = (
    "краевая задача",
    "задача Коши",
    "метод стрельбы",
    "метод продолжения по параметру",
    "невязка краевых условий",
)


def read_text(relative: str) -> str:
    return (REPOSITORY / relative).read_text(encoding="utf-8")


def local_link_failures(relative: str, text: str) -> list[str]:
    source = REPOSITORY / relative
    failures: list[str] = []
    for raw_target in MARKDOWN_LINK.findall(text):
        target = raw_target.strip().strip("<>").split(maxsplit=1)[0]
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        target = unquote(target.split("#", 1)[0].split("?", 1)[0])
        if target and not (source.parent / target).resolve().exists():
            failures.append(f"{relative} -> {target}")
    return failures


def block_counts(text: str) -> tuple[int, int]:
    fences = sum(1 for line in text.splitlines() if line.strip().startswith("```"))
    formulas = text.count("\\[") + text.count("$$")
    return fences, formulas


def check(condition: bool, label: str, details: list[str]) -> None:
    print(f"{label}={'PASS' if condition else 'FAIL'}")
    if not condition:
        details.append(label)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    failures: list[str] = []
    coverage = json.loads(COVERAGE_PATH.read_text(encoding="utf-8"))
    pairs = coverage["current_documentation"]["pairs"]

    expected_languages = coverage.get("languages") == ["zh-CN", "ru-RU"]
    check(expected_languages, "languages_declared", failures)

    managed_paths = {
        "README.md",
        "README.ru.md",
        "docs/bilingual/README.md",
        "docs/i18n/glossary.zh-ru.md",
        "docs/i18n/style-guide.md",
        "docs/translations/v2.10/README.md",
        "docs/translations/v2.10/report.ru.md",
    }
    pair_paths_exist = True
    heading_pairs_match = True
    block_pairs_match = True
    for pair in pairs:
        zh_path = pair["zh-CN"]
        ru_path = pair["ru-RU"]
        managed_paths.update((zh_path, ru_path))
        if not (REPOSITORY / zh_path).is_file() or not (REPOSITORY / ru_path).is_file():
            pair_paths_exist = False
            continue
        zh_text = read_text(zh_path)
        ru_text = read_text(ru_path)
        zh_headings = NUMBERED_H2.findall(zh_text)
        ru_headings = NUMBERED_H2.findall(ru_text)
        if not zh_headings or zh_headings != ru_headings:
            heading_pairs_match = False
        zh_fences, zh_formulas = block_counts(zh_text)
        ru_fences, ru_formulas = block_counts(ru_text)
        if (
            zh_fences % 2
            or ru_fences % 2
            or zh_fences != ru_fences
            or zh_formulas != ru_formulas
        ):
            block_pairs_match = False

    check(pair_paths_exist and len(pairs) == 7, "bilingual_file_pairs", failures)
    check(heading_pairs_match, "paired_numbered_sections", failures)
    check(block_pairs_match, "paired_formula_code_blocks", failures)

    missing_links: list[str] = []
    placeholders: list[str] = []
    for relative in sorted(managed_paths):
        path = REPOSITORY / relative
        if not path.is_file():
            missing_links.append(relative)
            continue
        text = read_text(relative)
        missing_links.extend(local_link_failures(relative, text))
        for placeholder in FORBIDDEN_PLACEHOLDERS:
            if placeholder.casefold() in text.casefold():
                placeholders.append(f"{relative}: {placeholder}")
    check(not missing_links, "managed_relative_links", failures)
    check(not placeholders, "translation_placeholders_absent", failures)

    readme_zh = read_text("README.md")
    readme_ru = read_text("README.ru.md")
    readme_links_ok = (
        "README.ru.md" in readme_zh
        and "docs/bilingual/README.md" in readme_zh
        and "README.md" in readme_ru
        and "docs/bilingual/README.md" in readme_ru
    )
    check(readme_links_ok, "root_readme_language_links", failures)

    index_text = read_text("docs/bilingual/README.md")
    index_links_ok = all(
        Path(pair[language]).name in index_text
        for pair in pairs
        for language in ("zh-CN", "ru-RU")
    )
    check(index_links_ok, "bilingual_index_links", failures)

    glossary = read_text("docs/i18n/glossary.zh-ru.md")
    check(
        all(term in glossary for term in REQUIRED_RUSSIAN_TERMS),
        "required_russian_terminology",
        failures,
    )

    historical = coverage["historical_translation"]
    metadata = json.loads(
        (REPOSITORY / "docs" / "reports" / "v2.10" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    software_sha = historical["software_snapshot_commit"]
    report_sha = historical["report_document_commit"]
    sha_ok = (
        bool(SHA.fullmatch(software_sha))
        and bool(SHA.fullmatch(report_sha))
        and metadata["git_commit"] == software_sha
        and metadata["report_document_commit"] == report_sha
    )
    check(sha_ok, "historical_sha_binding", failures)

    translation = read_text(historical["ru-RU"])
    historical_facts_ok = (
        software_sha in translation
        and "285" in translation
        and "1413ba43" in translation
        and "31305225622" in translation
        and "31310177928" in translation
        and historical["historical_test_count"] == metadata["test_count"] == 285
    )
    check(historical_facts_ok, "historical_translation_facts", failures)

    version_ok = (
        coverage.get("project_version") == "v2.10"
        and coverage.get("software_version_changed") is False
        and coverage.get("gui_i18n_implemented_in_phase") is False
    )
    check(version_ok, "phase_scope_flags", failures)

    passed = not failures
    if failures:
        print("failed_checks=" + ",".join(failures))
    print(f"bilingual_documentation_passed={passed}")
    print(f"bilingual_documentation_status={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
