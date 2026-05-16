"""Shared pytest fixtures + sys.path setup for sanity suite."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SAFETY_DIR = REPO_ROOT / "src" / "scripts" / "safety"

if str(SAFETY_DIR) not in sys.path:
    sys.path.insert(0, str(SAFETY_DIR))
