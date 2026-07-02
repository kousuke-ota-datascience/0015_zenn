from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.models import ColumnSpec, DatasetSchema


class CatalogError(Exception):
    """Base class for catalog-related errors."""


class DatasetNotFoundError(CatalogError):
    """Raised when a dataset ID is not registered in the catalog."""


class CatalogValidationError(CatalogError):
    """Raised when a requested dimension, metric, filter, or sort is invalid."""


class DimensionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    column: str
    type: str


class MetricDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    expression: str
    type: str


class DatasetDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    source_type: Literal["csv", "parquet"]
    source: str
    dimensions: dict[str, DimensionDefinition] = Field(default_factory=dict)
    metrics: dict[str, MetricDefinition] = Field(default_factory=dict)


class CatalogDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    datasets: dict[str, DatasetDefinition]


class DatasetCatalog:
    def __init__(self, document: CatalogDocument) -> None:
        self._document = document

    @classmethod
    def load(cls, path: Path) -> "DatasetCatalog":
        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
        if raw is None:
            raise CatalogValidationError(f"Catalog is empty: {path}")
        return cls(CatalogDocument.model_validate(raw))

    def list_datasets(self) -> list[DatasetSchema]:
        return [
            self._build_schema(dataset_id, dataset)
            for dataset_id, dataset in self._document.datasets.items()
        ]

    def has_dataset(self, dataset_id: str) -> bool:
        return dataset_id in self._document.datasets

    def get_dataset(self, dataset_id: str) -> DatasetDefinition:
        try:
            return self._document.datasets[dataset_id]
        except KeyError as exc:
            raise DatasetNotFoundError(f"Unknown dataset: {dataset_id}") from exc

    def get_schema(self, dataset_id: str) -> DatasetSchema:
        dataset = self.get_dataset(dataset_id)
        return self._build_schema(dataset_id, dataset)

    def validate_dimensions(
        self,
        dataset_id: str,
        dimension_ids: list[str],
    ) -> dict[str, DimensionDefinition]:
        dataset = self.get_dataset(dataset_id)
        unknown = [item for item in dimension_ids if item not in dataset.dimensions]
        if unknown:
            raise CatalogValidationError(
                f"Unknown dimension(s) for dataset '{dataset_id}': {', '.join(unknown)}"
            )
        return {item: dataset.dimensions[item] for item in dimension_ids}

    def validate_metrics(
        self,
        dataset_id: str,
        metric_ids: list[str],
    ) -> dict[str, MetricDefinition]:
        dataset = self.get_dataset(dataset_id)
        unknown = [item for item in metric_ids if item not in dataset.metrics]
        if unknown:
            raise CatalogValidationError(
                f"Unknown metric(s) for dataset '{dataset_id}': {', '.join(unknown)}"
            )
        return {item: dataset.metrics[item] for item in metric_ids}

    def columns_for_request(
        self,
        dataset_id: str,
        dimension_ids: list[str],
        metric_ids: list[str],
    ) -> list[ColumnSpec]:
        dataset = self.get_dataset(dataset_id)
        return [
            ColumnSpec(
                id=dimension_id,
                label=dataset.dimensions[dimension_id].label,
                type=dataset.dimensions[dimension_id].type,
            )
            for dimension_id in dimension_ids
        ] + [
            ColumnSpec(
                id=metric_id,
                label=dataset.metrics[metric_id].label,
                type=dataset.metrics[metric_id].type,
            )
            for metric_id in metric_ids
        ]

    @staticmethod
    def _build_schema(dataset_id: str, dataset: DatasetDefinition) -> DatasetSchema:
        return DatasetSchema(
            id=dataset_id,
            label=dataset.label,
            dimensions=[
                ColumnSpec(id=item_id, label=item.label, type=item.type)
                for item_id, item in dataset.dimensions.items()
            ],
            metrics=[
                ColumnSpec(id=item_id, label=item.label, type=item.type)
                for item_id, item in dataset.metrics.items()
            ],
        )
