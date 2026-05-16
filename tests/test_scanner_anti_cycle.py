"""P0i scanner tests: M2 anti-cycle dynamic check (v4).

Per TEST_SAFETY_FEEDBACK_DESIGN.md v4 §4.2:

  Test the scanner against:
    - good fixtures: PASS
    - gate_bad_self_citing_verifier: REJECT with code=anti-cycle
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scan_outcome_claims import scan  # type: ignore  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FIX = REPO / "tests" / "fixtures"


def test_good_fixture_scanner_passes_fork_patch():
    """fork_patch good fixture must scan-pass (no verifier_scripts that
    reference manifest paths)."""
    manifest = FIX / "good" / "good_fork_patch_C_PATCH.yaml"
    result = scan(manifest, REPO)
    assert result.ok, f"good fixture failed scan:\n{result.render()}"


def test_scanner_rejects_self_citing_verifier():
    """gate_bad_self_citing_verifier has a verifier whose body reads
    the manifest's smoke_log path. M2 anti-cycle must reject."""
    manifest = FIX / "gate_bad" / "gate_bad_self_citing_verifier" / "claim_manifest.yaml"
    result = scan(manifest, REPO)
    assert not result.ok, "scanner unexpectedly accepted self-citing verifier"
    anti_cycle_errors = [e for e in result.errors if e.code == "anti-cycle"]
    assert anti_cycle_errors, (
        f"expected anti-cycle error; got: {[str(e) for e in result.errors]}"
    )


def test_scanner_rejects_missing_verifier_script():
    """If a manifest declares a verifier_script.path that doesn't exist,
    scanner must reject (verifier-missing)."""
    # Use the good_fork_patch but synthesize a bad verifier path inline.
    # For test simplicity, point to a known-missing path.
    import yaml
    import tempfile

    data = yaml.safe_load(
        (FIX / "good" / "good_fork_patch_C_PATCH.yaml").read_text()
    )
    data["verifier_scripts"] = [
        {
            "role": "smoke_replay",
            "path": "tests/fixtures/_does_not_exist/missing_verifier.sh",
            "purpose": "test",
        }
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        tmp = Path(f.name)
    try:
        result = scan(tmp, REPO)
        assert not result.ok, "scanner unexpectedly accepted missing verifier"
        missing_errors = [
            e for e in result.errors if e.code == "verifier-missing"
        ]
        assert missing_errors, (
            f"expected verifier-missing error; got: "
            f"{[str(e) for e in result.errors]}"
        )
    finally:
        tmp.unlink(missing_ok=True)
