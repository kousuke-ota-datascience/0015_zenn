from __future__ import annotations

import shutil
import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "py" / "myproj" / "src"))


def pytest_sessionfinish(session: object, exitstatus: int) -> None:
    """Keep repository-level smoke checks independent from generated caches."""

    for pycache in EXPERIMENT_DIR.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)
