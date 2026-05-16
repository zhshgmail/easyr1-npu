"""P0h.0 — schema roundtrip + per-fixture validation tests.

Per TEST_SAFETY_FEEDBACK_DESIGN.md §3.6 (v4):
  - good fixtures → validate_manifest() OK
  - schema_bad fixtures → validate_manifest() rejects with expected error code

Run via: bash scripts/run_sanity_suite.sh
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from validate_claim_manifest import (  # type: ignore  # noqa: E402
    ValidationError,
    validate_manifest,
    validate_manifest_collect,
)

REPO = Path(__file__).resolve().parent.parent
FIX = REPO / "tests" / "fixtures"


# ---------------------------------------------------------------------------
# Good fixtures
# ---------------------------------------------------------------------------

GOOD_FIXTURES = [
    "good_fork_patch_C_PATCH.yaml",
    "good_issue_only_C_REPORT.yaml",
    "good_integrated_overlay_A.yaml",
]


@pytest.mark.parametrize("fix_name", GOOD_FIXTURES)
def test_good_fixture_passes(fix_name):
    """Each good fixture must validate cleanly under repo_root."""
    data = yaml.safe_load((FIX / "good" / fix_name).read_text())
    result = validate_manifest_collect(data, repo_root=REPO)
    assert result.ok, f"good fixture {fix_name} failed validation:\n{result.render()}"


# ---------------------------------------------------------------------------
# Schema-bad fixtures — each must raise with the expected error code
# ---------------------------------------------------------------------------

SCHEMA_BAD_FIXTURES = [
    # (filename, substring-of-error-code-or-description-that-must-appear)
    ("schema_bad_outcome_string_variant.yaml", "outcome-canonical"),
    ("schema_bad_perf_null_with_pass.yaml", "evidence-numeric-missing"),
    ("schema_bad_issue_only_with_shim.yaml", "mode-artifact-mismatch"),
    ("schema_bad_missing_self_challenge.yaml", "self-challenge-required"),
    ("schema_bad_repo_root_anchor_missing.yaml", "repo-root-anchor"),
]


@pytest.mark.parametrize("fix_name,expected_code_substr", SCHEMA_BAD_FIXTURES)
def test_schema_bad_fixture_rejected(fix_name, expected_code_substr):
    data = yaml.safe_load((FIX / "schema_bad" / fix_name).read_text())
    result = validate_manifest_collect(data, repo_root=REPO)
    assert not result.ok, f"schema_bad fixture {fix_name} unexpectedly passed"
    matching = [e for e in result.errors if expected_code_substr in e.code]
    assert matching, (
        f"schema_bad fixture {fix_name} rejected for wrong reason; "
        f"expected code containing {expected_code_substr!r}; "
        f"got errors: {[str(e) for e in result.errors]}"
    )


@pytest.mark.parametrize("fix_name,expected_code_substr", SCHEMA_BAD_FIXTURES)
def test_schema_bad_fixture_raises(fix_name, expected_code_substr):
    """validate_manifest (raising form) raises ValidationError."""
    data = yaml.safe_load((FIX / "schema_bad" / fix_name).read_text())
    with pytest.raises(ValidationError):
        validate_manifest(data, repo_root=REPO)
