"""DEBT-3: install-skills.sh writes a version stamp; sanity test verifies
the stamp file format + presence of required fields.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STAMP = REPO / ".claude" / ".easyr1_skills_version"


def test_stamp_file_exists():
    """install-skills.sh must have been run at least once."""
    assert STAMP.exists(), (
        f".claude/.easyr1_skills_version missing; run "
        f"`bash src/scripts/install-skills.sh --force` once"
    )


def test_stamp_has_required_fields():
    text = STAMP.read_text()
    required = ["_owner: easyr1-npu", "manifest_sha256:", "deployed_count:",
                "deployed_at:", "skills_dir:"]
    for field in required:
        assert field in text, f"stamp missing {field!r}; got: {text[:300]}"


def test_stamp_sha256_format():
    text = STAMP.read_text()
    m = re.search(r"manifest_sha256: ([0-9a-f]{64})$", text, re.MULTILINE)
    assert m, f"manifest_sha256 not a valid 64-hex string; got: {text[:300]}"


def test_install_script_writes_stamp():
    """grep install-skills.sh to confirm DEBT-3 wiring present."""
    sh = REPO / "src" / "scripts" / "install-skills.sh"
    text = sh.read_text()
    assert "_owner: easyr1-npu" in text, (
        "install-skills.sh missing DEBT-3 stamp wiring"
    )
    assert "manifest_sha256" in text, (
        "install-skills.sh missing manifest_sha256 emit"
    )
