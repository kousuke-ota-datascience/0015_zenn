"""Base strategy interface for pipeline modes."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..context import RunContext


class AnalysisModeStrategy(ABC):
    """Abstract base class for executable pipeline modes."""

    mode: str

    @abstractmethod
    def run(self, context: RunContext) -> None:
        """Runs the analysis mode.

        Args:
            context: Resolved run context containing configuration and paths.

        Raises:
            ValueError: If the mode configuration is invalid.
        """
        raise NotImplementedError

