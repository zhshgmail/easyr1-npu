"""Test that the local repo satisfies the invariants the helper script
guards against (DEBT-1 / NPU-OPS-014 / NPU-OPS-015 territory).

If sanity suite runs on a tree that violates these, the helper script's
stale-repo guard would warn — let's surface it earlier in CI/dev.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_repo_is_git_tracked():
    """Sanity suite must run inside a real git worktree (not a hand-copied
    v0 snapshot). NPU-OPS-014."""
    r = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--git-dir"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, (
        f"REPO={REPO} is not a git working tree (NPU-OPS-014): "
        f"{r.stderr.strip()}"
    )


def test_src_scripts_layout_present():
    """The current layout assumes `src/scripts/run-npu-container.sh`
    (not the v0 layout's `scripts/run-npu-container.sh`)."""
    target = REPO / "src" / "scripts" / "run-npu-container.sh"
    assert target.exists(), (
        f"src/scripts/run-npu-container.sh missing — looks like v0 stale "
        f"layout (NPU-OPS-014). Expected at {target}"
    )


def test_run_npu_container_helper_has_stale_guard():
    """run-npu-container.sh should warn if invoked from a non-git tree.
    Verify the guard code is present (grep-level check)."""
    sh = REPO / "src" / "scripts" / "run-npu-container.sh"
    text = sh.read_text()
    assert "NPU-OPS-014" in text, (
        "run-npu-container.sh missing NPU-OPS-014 stale-repo guard "
        "(DEBT-1 fix should add it)"
    )
    assert "rev-parse --git-dir" in text, (
        "expected `git rev-parse --git-dir` check in stale-repo guard"
    )
