"""Canonical mode_dispatch_table — single authoritative source for
Mode / Outcome / ArtifactRole / EvidenceType enums + REQUIRED_* dicts.

Per TEST_SAFETY_FEEDBACK_DESIGN.md §2.5 (v4):

  The entire repo's day-0 / port outcome semantics flow from THIS file.
  Schema validator, scanner, finalize gates, sanity suite — all derive
  required-field sets from REQUIRED_ARTIFACTS / REQUIRED_EVIDENCE here.

  DO NOT redeclare any of these enums elsewhere. test_mode_dispatch_
  authority.py enforces single-source-of-truth via repo-wide grep.

Author: T31.4 (P0h.0 implementation).
"""
from __future__ import annotations

import enum
from typing import Dict, Set


class Mode(str, enum.Enum):
    """Workflow modes; each implies different required artifact set."""

    FORK_PATCH = "fork_patch"               # we patch upstream NPU project via fork
    ISSUE_ONLY = "issue_only"               # we file issue to upstream; no fork
    INTEGRATED_OVERLAY = "integrated_overlay"  # we build integrated image from N forks


class Outcome(str, enum.Enum):
    """Canonical outcome enum. Prose variants (e.g. 'A-with-note',
    'A-w-note') are FORBIDDEN — sanity suite checks for them."""

    A = "A"                       # works as-is; no NPU integration delta
    A_WITH_NOTE = "A_WITH_NOTE"   # works + additive note
    B = "B"                       # single-commit / env-var workaround
    C_PATCH = "C_PATCH"           # needs forward-compat shim
    C_REPORT = "C_REPORT"         # bug in community upstream; we file issue


class ArtifactRole(str, enum.Enum):
    """Roles for artifacts attached to a manifest. Role-keyed (not
    filename-keyed) to keep gate logic mode-invariant."""

    SHIM_MODULE = "shim_module"           # compat shim Python module
    PR_MATERIAL = "pr_material"           # PR_MATERIAL.md handoff doc
    WORKAROUND_DOC = "workaround_doc"     # issue_only's customer note
    IMAGE_TAG = "image_tag"               # integrated overlay image SHA
    SMOKE_LOG = "smoke_log"               # on-A3 smoke log
    CHECKPOINT_PATH = "checkpoint_path"   # V1.4 GRPO checkpoint


class EvidenceType(str, enum.Enum):
    """Types of evidence items that back claims."""

    SMOKE_LOG = "smoke_log"
    COMMIT_REF = "commit_ref"
    NUMERIC_METRIC = "numeric_metric"
    GITHUB_ISSUE = "github_issue"
    IMAGE_SHA = "image_sha"


# ---------------------------------------------------------------------------
# REQUIRED_* — the single dispatch authority.
# Downstream consumers (schema validator, scanner, gates) read from these.
# ---------------------------------------------------------------------------

REQUIRED_ARTIFACTS: Dict[Mode, Set[ArtifactRole]] = {
    Mode.FORK_PATCH: {
        ArtifactRole.SHIM_MODULE,
        ArtifactRole.PR_MATERIAL,
    },
    Mode.ISSUE_ONLY: {
        ArtifactRole.WORKAROUND_DOC,
    },
    Mode.INTEGRATED_OVERLAY: {
        ArtifactRole.IMAGE_TAG,
        ArtifactRole.SMOKE_LOG,
        ArtifactRole.CHECKPOINT_PATH,
    },
}

REQUIRED_EVIDENCE: Dict[Outcome, Set[EvidenceType]] = {
    Outcome.A: {EvidenceType.SMOKE_LOG, EvidenceType.COMMIT_REF},
    Outcome.A_WITH_NOTE: {EvidenceType.SMOKE_LOG, EvidenceType.COMMIT_REF},
    Outcome.B: {EvidenceType.SMOKE_LOG, EvidenceType.COMMIT_REF},
    Outcome.C_PATCH: {
        EvidenceType.SMOKE_LOG,
        EvidenceType.COMMIT_REF,
        EvidenceType.NUMERIC_METRIC,
    },
    Outcome.C_REPORT: {EvidenceType.GITHUB_ISSUE},
}

REQUIRED_VALIDATION_LEVEL: Dict[Mode, int] = {
    Mode.FORK_PATCH: 3,           # L3 on-A3 import smoke
    Mode.ISSUE_ONLY: 3,
    Mode.INTEGRATED_OVERLAY: 5,   # L5 V1.4 GRPO end-to-end
}

SELF_CHALLENGE_MIN_PASSED = 8     # of 11 patterns in docs/_meta/kb/challenge_patterns/
SCHEMA_VERSION = "1.0"
CLAIM_MANIFEST_VERSION = 1


def required_artifacts(mode: Mode) -> Set[ArtifactRole]:
    return REQUIRED_ARTIFACTS[mode]


def required_evidence(outcome: Outcome) -> Set[EvidenceType]:
    return REQUIRED_EVIDENCE[outcome]


def required_validation_level(mode: Mode) -> int:
    return REQUIRED_VALIDATION_LEVEL[mode]
