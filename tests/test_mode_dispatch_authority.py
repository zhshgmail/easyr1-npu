"""Enforce that mode_dispatch.py is the SINGLE authority for
Mode / Outcome / ArtifactRole / EvidenceType enums + REQUIRED_* tables.

Per TEST_SAFETY_FEEDBACK_DESIGN.md §2.5 (v4):
  Any other file in src/ that declares `class Outcome` / `class Mode` /
  `class ArtifactRole` / `class EvidenceType` is a drift signal —
  fail sanity suite.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

AUTHORITY = REPO / "src" / "scripts" / "safety" / "mode_dispatch.py"


def test_authority_file_exists():
    assert AUTHORITY.exists(), f"canonical authority {AUTHORITY} missing"


def test_no_competing_enum_declarations():
    """grep all .py files under src/scripts/safety/ for class Outcome/Mode/
    ArtifactRole/EvidenceType; only mode_dispatch.py is allowed."""
    pattern = re.compile(
        r"^\s*class\s+(Outcome|Mode|ArtifactRole|EvidenceType)\b",
        re.MULTILINE,
    )
    safety_dir = REPO / "src" / "scripts" / "safety"
    if not safety_dir.exists():
        return  # nothing to scan
    violators = []
    for py in safety_dir.rglob("*.py"):
        if py == AUTHORITY:
            continue
        text = py.read_text()
        if pattern.search(text):
            violators.append(py.relative_to(REPO))
    assert not violators, (
        f"Mode/Outcome/ArtifactRole/EvidenceType redeclared outside "
        f"{AUTHORITY.relative_to(REPO)}: {violators}"
    )


def test_outcome_enum_canonical_only():
    """Scan claim manifest fixtures for prose outcome variants like
    'A-with-note' which the canonical Outcome enum forbids.

    Scope: only `outcome: <value>` lines in claim_manifest fixtures.
    Code files (mode_dispatch.py / docs) may mention forbidden variants
    as cautionary examples — that's the whole point of declaring them
    forbidden.
    """
    forbidden = {"A-with-note", "A-w-note", "a-with-note", "a_with_note"}
    pattern = re.compile(r"^\s*outcome:\s*['\"]?([^'\"\s]+)['\"]?\s*(#.*)?$")

    fixture_dirs = [REPO / "tests" / "fixtures" / "good"]
    for fix_dir in fixture_dirs:
        if not fix_dir.exists():
            continue
        for yf in fix_dir.glob("*.yaml"):
            for lineno, line in enumerate(yf.read_text().splitlines(), 1):
                m = pattern.match(line)
                if m and m.group(1) in forbidden:
                    raise AssertionError(
                        f"{yf.relative_to(REPO)}:{lineno}: good fixture "
                        f"uses forbidden prose outcome variant {m.group(1)!r}"
                    )
