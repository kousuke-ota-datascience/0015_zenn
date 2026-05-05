from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[5]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT
