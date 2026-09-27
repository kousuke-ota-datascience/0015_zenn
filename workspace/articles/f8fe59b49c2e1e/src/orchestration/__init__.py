"""Executable Workflow 00 orchestration.

Exports are resolved lazily so `python -m src.orchestration.entry_pipeline`
does not import the module before runpy executes it.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "MAX_REVIEW_CYCLES_PER_RUN",
    "EntryClassification",
    "PipelineResult",
    "classify_entry_state",
    "run_entry_pipeline",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from . import entry_pipeline

    return getattr(entry_pipeline, name)
