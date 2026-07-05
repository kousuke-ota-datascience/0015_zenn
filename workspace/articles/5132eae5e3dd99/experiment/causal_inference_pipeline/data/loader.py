"""Source table loading through the project dataset registry."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from myproj.io.config_resolver import load_dataset_definition
from myproj.io.file_io import FileConfigRegistry, FileIOUtils
from myproj.logger.custom_logger import CustomLogger

from ..features.config_schema import TableSpec
from .validation import validate_required_columns


LOGGER_NAME = "causal_inference_completejourney"


class DataLoader:
    """Loads raw input tables required by the causal inference pipeline.

    Args:
        project_root: Repository root used to resolve the dataset registry.
        dataset_yaml: Dataset registry YAML path.
        table_specs: Feature-config table specifications keyed by logical
            table name.

    Attributes:
        project_root: Repository root.
        dataset_yaml: Dataset registry YAML path.
        table_specs: Table specifications keyed by logical table name.
    """

    def __init__(
        self,
        *,
        project_root: Path,
        dataset_yaml: Path,
        table_specs: Mapping[str, TableSpec],
    ) -> None:
        """Initialize the loader.

        Args:
            project_root: Repository root used to resolve registry paths.
            dataset_yaml: Dataset registry YAML path.
            table_specs: Table specifications keyed by logical table name.
        """
        self.project_root = project_root
        self.dataset_yaml = dataset_yaml
        self.table_specs = table_specs

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Loads all configured raw tables.

        Returns:
            Mapping from logical table names to data frames.

        Raises:
            FileNotFoundError: If a configured input file does not exist.
            ValueError: If a loaded table misses required columns.
        """
        logger = CustomLogger(
            LOGGER_NAME,
            Path.cwd() / f"{LOGGER_NAME}.log",
        ).get_logger()
        dataset_definition = load_dataset_definition(self.dataset_yaml, self.project_root)
        registry = FileConfigRegistry.from_mapping(dataset_definition)
        file_io = FileIOUtils(logger)

        tables: dict[str, pd.DataFrame] = {}
        for logical_name, spec in self.table_specs.items():
            frame = file_io.read_file(registry.read_config(spec.name), use_dask=False)
            validate_required_columns(frame, spec.required_columns, logical_name)
            tables[logical_name] = frame
        return tables


class CompleteJourneyDataLoader(DataLoader):
    """Backward-compatible loader name for Complete Journey experiments."""

