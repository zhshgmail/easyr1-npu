"""P0k tests: snapshot_current + diff_snapshot round-trip + regression detection."""
from __future__ import annotations

from pathlib import Path

import yaml

from diff_snapshot import diff  # type: ignore  # noqa: E402
from snapshot_current import snapshot as snapshot_current  # type: ignore  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def test_snapshot_yaml_roundtrips():
    """snapshot_current output is valid YAML and contains baselines key."""
    yml = snapshot_current(REPO)
    data = yaml.safe_load(yml)
    assert isinstance(data, dict)
    assert "as_of_date" in data
    assert "baselines" in data
    assert isinstance(data["baselines"], list)
    assert len(data["baselines"]) >= 5  # at least our 5 fork rows


def test_diff_self_pass():
    """Snapshot vs itself yields no regressions."""
    yml = snapshot_current(REPO)
    data = yaml.safe_load(yml)
    regressions = diff(data, data)
    assert regressions == [], f"unexpected regressions: {regressions}"


def test_diff_detects_skill_removal():
    """If a skill disappears, diff reports REGRESSION."""
    yml = snapshot_current(REPO)
    baseline = yaml.safe_load(yml)
    current = {
        "baselines": [
            b for b in baseline["baselines"]
            if b.get("skill") != "/vllm-ascend-day0"
        ]
    }
    regressions = diff(baseline, current)
    assert any("vllm-ascend-day0" in r for r in regressions), (
        f"expected regression for missing vllm-ascend-day0; got {regressions}"
    )


def test_diff_detects_negative_marker_gain():
    """If a skill's status gains BLOCKED/FAIL, diff reports REGRESSION."""
    yml = snapshot_current(REPO)
    baseline = yaml.safe_load(yml)
    # Mutate one entry's status to simulate regression.
    current = yaml.safe_load(yml)
    for b in current["baselines"]:
        if b.get("skill") == "/torch-npu-day0":
            b["status_summary"] = "FAIL — broke on torch 2.13"
            break
    regressions = diff(baseline, current)
    assert any("torch-npu-day0" in r and ("FAIL" in r or "negative" in r)
               for r in regressions), (
        f"expected FAIL-marker regression; got {regressions}"
    )


def test_diff_image_sha_change():
    """If image_sha256 changes, diff reports REGRESSION."""
    yml = snapshot_current(REPO)
    baseline = yaml.safe_load(yml)
    current = yaml.safe_load(yml)
    for b in current["baselines"]:
        if b.get("skill") == "/integrated-overlay-build":
            b["image_sha256"] = "deadbeef99999"
            break
    regressions = diff(baseline, current)
    assert any("image_sha256" in r for r in regressions), (
        f"expected image_sha256 regression; got {regressions}"
    )
