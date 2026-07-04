from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from myproj.io.config_resolver import load_dataset_definition
from myproj.io.file_io import FileConfigRegistry, FileIOUtils
from myproj.logger.custom_logger import CustomLogger


LOGGER_NAME = "causal_discovery_pipeline"
DEFAULT_TABLES = (
    "campaign_descriptions",
    "campaigns",
    "demographics",
    "transactions",
)


class CompleteJourneyDataLoader:
    """Load Complete Journey tables from the configured dataset registry.

    Args:
        project_root: Repository root used to resolve dataset placeholders.
        dataset_yaml: Dataset registry YAML path.
    """

    def __init__(self, *, project_root: Path, dataset_yaml: Path) -> None:
        """Initialize the loader.

        Args:
            project_root: Repository root used by dataset config resolution.
            dataset_yaml: YAML file describing source table locations.
        """
        self.project_root = project_root
        self.dataset_yaml = dataset_yaml

    def load_tables(
        self,
        table_entries: Iterable[str] = DEFAULT_TABLES,
    ) -> dict[str, pd.DataFrame]:
        """Load selected logical tables into pandas data frames.

        Args:
            table_entries: Dataset registry entry names to load.

        Returns:
            Mapping from registry entry name to loaded data frame.
        """
        logger = CustomLogger(
            LOGGER_NAME,
            Path.cwd() / f"{LOGGER_NAME}.log",
        ).get_logger()
        dataset_definition = load_dataset_definition(self.dataset_yaml, self.project_root)
        registry = FileConfigRegistry.from_mapping(dataset_definition)
        file_io = FileIOUtils(logger)

        return {
            entry: file_io.read_file(registry.read_config(entry), use_dask=False)
            for entry in table_entries
        }
