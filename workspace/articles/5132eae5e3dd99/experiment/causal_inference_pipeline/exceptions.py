"""Custom exceptions raised by the causal inference pipeline."""

from __future__ import annotations


class PipelineConfigError(ValueError):
    """Raised when pipeline configuration is missing or inconsistent."""

