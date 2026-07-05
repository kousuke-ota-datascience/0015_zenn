"""Structured output writing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class OutputWriter:
    """Writes pipeline outputs to a structured output directory.

    Args:
        output_dir: Root output directory.
        write_csv: Whether CSV outputs should be written.
        write_markdown: Whether Markdown reports should be written.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        write_csv: bool = True,
        write_markdown: bool = True,
    ) -> None:
        """Initialize the writer.

        Args:
            output_dir: Root output directory.
            write_csv: Whether CSV files should be written.
            write_markdown: Whether Markdown files should be written.
        """
        self.output_dir = output_dir
        self.write_csv = write_csv
        self.write_markdown = write_markdown

    def write_csv_table(self, relative_path: str, frame: pd.DataFrame) -> None:
        """Write a CSV table when CSV output is enabled.

        Args:
            relative_path: Path relative to ``output_dir``.
            frame: Data frame to write.
        """
        if not self.write_csv:
            return
        path = self.output_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    def write_markdown_text(self, relative_path: str, text: str) -> None:
        """Write a Markdown text file when Markdown output is enabled.

        Args:
            relative_path: Path relative to ``output_dir``.
            text: Markdown content.
        """
        if not self.write_markdown:
            return
        path = self.output_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

