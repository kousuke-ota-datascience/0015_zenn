from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


FilterOperator = Literal["=", "!=", ">", ">=", "<", "<=", "in"]
SortDirection = Literal["asc", "desc"]


class FilterSpec(BaseModel):
    field: str
    operator: FilterOperator
    value: Any

    @model_validator(mode="after")
    def validate_value_shape(self) -> "FilterSpec":
        if self.operator == "in":
            if not isinstance(self.value, list) or len(self.value) == 0:
                raise ValueError("'in' filter requires a non-empty list value")
            return self

        if isinstance(self.value, (dict, list)):
            raise ValueError(f"'{self.operator}' filter requires a scalar value")
        return self


class SortSpec(BaseModel):
    field: str
    direction: SortDirection = "asc"


class QueryRequest(BaseModel):
    dataset: str
    dimensions: list[str] = Field(default_factory=list, max_length=3)
    metrics: list[str] = Field(min_length=1, max_length=5)
    filters: list[FilterSpec] = Field(default_factory=list)
    sort: list[SortSpec] = Field(default_factory=list)
    limit: int = Field(default=100, ge=1, le=1000)


class ColumnSpec(BaseModel):
    id: str
    label: str
    type: str


class DatasetSchema(BaseModel):
    id: str
    label: str
    dimensions: list[ColumnSpec]
    metrics: list[ColumnSpec]


class QueryResponse(BaseModel):
    columns: list[ColumnSpec]
    rows: list[dict[str, Any]]
