"""DEBT-2 enforcement: any file that mentions specific NPU bind paths
must cite the canonical source (NPU-OPS-009/011/012/013/014) or
link to src/skills/_shared/npu-container-runner/SKILL.md.

Per OL-32 (npu-container-runner is the SOLE authority for bind set).
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Paths that signal "this file is talking about NPU bind set"
BIND_PATH_MARKERS = [
    "/dev/devmm_svm",
    "/dev/hisi_hdc",
    "/dev/davinci_manager",
]

# Markers that prove the file cites the canonical source
CITATION_MARKERS = [
    "NPU-OPS-009",
    "NPU-OPS-011",
    "NPU-OPS-012",
    "NPU-OPS-013",
    "NPU-OPS-014",
    "npu-container-runner",
    "run-npu-container.sh",
    "OL-32",
]

# Files exempt from the rule (the authoritative sources themselves, or
# historical archives / generated outputs)
EXEMPT_PATTERNS = [
    re.compile(r"src/skills/_shared/npu-container-runner/SKILL\.md$"),
    re.compile(r"src/scripts/run-npu-container\.sh$"),
    re.compile(r"src/skills/_shared/integrated-overlay-build/SKILL\.md$"),
    re.compile(r"knowledge/npu-patterns\.md$"),
    re.compile(r"docs/_archive/"),
    re.compile(r"docs/_meta/postmortem/"),
    re.compile(r"docs/_meta/design/"),
    re.compile(r"docs/_meta/handovers/"),
    re.compile(r"tests/test_bind_set_single_source\.py$"),
    re.compile(r"tests/fixtures/"),
    re.compile(r"\.pytest_cache/"),
]


def _is_exempt(rel: str) -> bool:
    return any(p.search(rel) for p in EXEMPT_PATTERNS)


def test_bind_paths_cite_canonical_source():
    """Every non-exempt file mentioning bind paths must cite npu-container
    -runner or NPU-OPS-009/011/012/013/014 (OL-32, DEBT-2)."""
    violators: list[str] = []
    for src in REPO.rglob("*"):
        if not src.is_file():
            continue
        if src.suffix not in {".md", ".sh", ".py", ".yaml", ".yml"}:
            continue
        rel = str(src.relative_to(REPO))
        if _is_exempt(rel):
            continue
        try:
            text = src.read_text(errors="replace")
        except (OSError, UnicodeDecodeError):
            continue
        if not any(m in text for m in BIND_PATH_MARKERS):
            continue
        if not any(c in text for c in CITATION_MARKERS):
            violators.append(rel)

    assert not violators, (
        "files mention NPU bind paths but don't cite "
        "NPU-OPS-009/011/012/013/014 or npu-container-runner "
        "(DEBT-2 / OL-32 violation):\n  - "
        + "\n  - ".join(violators)
    )
