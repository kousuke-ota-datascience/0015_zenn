from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


ALLOWED_ALGORITHMS = ("pc", "ges", "lingam", "notears")
ALLOWED_TRANSFORMS = (
    "identity",
    "ordered_category_midpoint",
    "numeric_category",
    "equals",
    "is_missing_or_unknown",
    "campaign_membership",
    "log1p",
    "signed_log1p",
)
ALLOWED_AGGREGATIONS = ("sum", "nunique", "mean", "count")
DEFAULT_PC_INDEP_TESTS = ("fisherz", "kci", "chisq", "gsq")
DEFAULT_DISCRETE_PC_INDEP_TESTS = ("chisq", "gsq")
DEFAULT_ALPHA_GRID = (0.001, 0.005, 0.01, 0.05)


def _as_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    """Validate that a YAML value is a mapping.

    Args:
        value: Parsed YAML value.
        field_name: Human-readable field name for error messages.

    Returns:
        ``value`` typed as a mapping.

    Raises:
        ValueError: If ``value`` is not a mapping.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _as_bool(value: Any, field_name: str) -> bool:
    """Validate that a YAML value is a boolean.

    Args:
        value: Parsed YAML value.
        field_name: Human-readable field name for error messages.

    Returns:
        ``value`` typed as ``bool``.

    Raises:
        ValueError: If ``value`` is not a boolean.
    """
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _as_tuple(value: Any, field_name: str) -> tuple[Any, ...]:
    """Normalize a YAML list-like value to a tuple.

    Args:
        value: Parsed YAML value.
        field_name: Human-readable field name for error messages.

    Returns:
        Tuple representation. ``None`` becomes an empty tuple.

    Raises:
        ValueError: If ``value`` is neither ``None`` nor list-like.
    """
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise ValueError(f"{field_name} must be a list")
    return tuple(value)


def _clean_dict(value: dict[str, Any]) -> dict[str, Any]:
    """Drop keys whose values are ``None`` from a dictionary.

    Args:
        value: Dictionary to filter.

    Returns:
        New dictionary without ``None`` values.
    """
    return {key: child for key, child in value.items() if child is not None}


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset registry settings from ``analysis.yaml``.

    Attributes:
        yaml_path: Dataset registry YAML path.
    """

    yaml_path: Path = Path("shared/py/myproj/conf/dataset/completejourney/10_interim.yaml")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "DatasetConfig":
        """Build dataset configuration from YAML data.

        Args:
            value: Mapping under the ``dataset`` key.

        Returns:
            Parsed dataset configuration.
        """
        data = _as_mapping(value or {}, "dataset")
        return cls(yaml_path=Path(data.get("yaml_path", cls.yaml_path)))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the dataset configuration to YAML-compatible data.

        Returns:
            Dictionary representation.
        """
        return {"yaml_path": str(self.yaml_path)}


@dataclass(frozen=True)
class RunConfig:
    """Runtime settings from ``analysis.yaml``.

    Attributes:
        campaign_id: Campaign identifier to analyze.
        pre_weeks: Number of pre-treatment weeks.
        output_dir: Directory where outputs are written.
        random_seed: Random seed for stochastic diagnostics.
    """

    campaign_id: str = "18"
    pre_weeks: int = 8
    output_dir: Path = Path("articles/5132eae5e3dd99/artifacts/causal_discovery")
    random_seed: int = 20260630

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "RunConfig":
        """Build runtime configuration from YAML data.

        Args:
            value: Mapping under the ``run`` key.

        Returns:
            Parsed runtime configuration.

        Raises:
            ValueError: If ``pre_weeks`` is not positive.
        """
        data = _as_mapping(value or {}, "run")
        pre_weeks = int(data.get("pre_weeks", cls.pre_weeks))
        if pre_weeks < 1:
            raise ValueError("run.pre_weeks must be positive")
        return cls(
            campaign_id=str(data.get("campaign_id", cls.campaign_id)),
            pre_weeks=pre_weeks,
            output_dir=Path(data.get("output_dir", cls.output_dir)),
            random_seed=int(data.get("random_seed", cls.random_seed)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize runtime configuration to YAML-compatible data.

        Returns:
            Dictionary representation.
        """
        return {
            "campaign_id": self.campaign_id,
            "pre_weeks": self.pre_weeks,
            "output_dir": str(self.output_dir),
            "random_seed": self.random_seed,
        }


@dataclass(frozen=True)
class PreprocessingConfig:
    """Preprocessing settings from ``analysis.yaml``.

    Attributes:
        collinearity_threshold: Maximum allowed absolute pairwise correlation.
    """

    collinearity_threshold: float = 0.995

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "PreprocessingConfig":
        """Build preprocessing configuration from YAML data.

        Args:
            value: Mapping under the ``preprocessing`` key.

        Returns:
            Parsed preprocessing configuration.

        Raises:
            ValueError: If ``collinearity_threshold`` is outside ``(0, 1]``.
        """
        data = _as_mapping(value or {}, "preprocessing")
        threshold = float(data.get("collinearity_threshold", cls.collinearity_threshold))
        if not 0 < threshold <= 1:
            raise ValueError("preprocessing.collinearity_threshold must be in (0, 1]")
        return cls(collinearity_threshold=threshold)

    def to_dict(self) -> dict[str, Any]:
        """Serialize preprocessing configuration to YAML-compatible data.

        Returns:
            Dictionary representation.
        """
        return {"collinearity_threshold": self.collinearity_threshold}


@dataclass(frozen=True)
class PCConfig:
    """PC algorithm settings from ``analysis.yaml``.

    Attributes:
        alpha: Main PC significance level.
        indep_test: Conditional-independence test name.
        allowed_indep_tests: Test names accepted by the CLI and schema.
        discrete_indep_tests: Test names that require discretization.
        discrete_bins: Number of quantile bins for discrete tests.
        alpha_grid: Alpha values for sensitivity analysis.
    """

    alpha: float = 0.01
    indep_test: str = "fisherz"
    allowed_indep_tests: tuple[str, ...] = DEFAULT_PC_INDEP_TESTS
    discrete_indep_tests: tuple[str, ...] = DEFAULT_DISCRETE_PC_INDEP_TESTS
    discrete_bins: int = 4
    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "PCConfig":
        """Build PC configuration from YAML data.

        Args:
            value: Mapping under ``discovery.pc``.

        Returns:
            Parsed PC configuration.

        Raises:
            ValueError: If tests, bins, or alpha grid are invalid.
        """
        data = _as_mapping(value or {}, "discovery.pc")
        allowed = tuple(str(item) for item in data.get("allowed_indep_tests", cls.allowed_indep_tests))
        discrete = tuple(str(item) for item in data.get("discrete_indep_tests", cls.discrete_indep_tests))
        indep_test = str(data.get("indep_test", cls.indep_test))
        discrete_bins = int(data.get("discrete_bins", cls.discrete_bins))
        alpha_grid = tuple(float(item) for item in data.get("alpha_grid", cls.alpha_grid))

        if indep_test not in allowed:
            raise ValueError("discovery.pc.indep_test must be in allowed_indep_tests")
        if not set(discrete).issubset(set(allowed)):
            raise ValueError("discovery.pc.discrete_indep_tests must be a subset of allowed_indep_tests")
        if discrete_bins < 2:
            raise ValueError("discovery.pc.discrete_bins must be at least 2")
        if not alpha_grid:
            raise ValueError("discovery.pc.alpha_grid must not be empty")

        return cls(
            alpha=float(data.get("alpha", cls.alpha)),
            indep_test=indep_test,
            allowed_indep_tests=allowed,
            discrete_indep_tests=discrete,
            discrete_bins=discrete_bins,
            alpha_grid=alpha_grid,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize PC configuration to YAML-compatible data.

        Returns:
            Dictionary representation.
        """
        return {
            "alpha": self.alpha,
            "indep_test": self.indep_test,
            "allowed_indep_tests": list(self.allowed_indep_tests),
            "discrete_indep_tests": list(self.discrete_indep_tests),
            "discrete_bins": self.discrete_bins,
            "alpha_grid": list(self.alpha_grid),
        }


@dataclass(frozen=True)
class NotearsConfig:
    """NOTEARS settings from ``analysis.yaml``.

    Attributes:
        threshold: Absolute edge-weight threshold for reporting.
    """

    threshold: float = 0.3

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "NotearsConfig":
        """Build NOTEARS configuration from YAML data.

        Args:
            value: Mapping under ``discovery.notears``.

        Returns:
            Parsed NOTEARS configuration.
        """
        data = _as_mapping(value or {}, "discovery.notears")
        return cls(threshold=float(data.get("threshold", cls.threshold)))

    def to_dict(self) -> dict[str, Any]:
        """Serialize NOTEARS configuration to YAML-compatible data.

        Returns:
            Dictionary representation.
        """
        return {"threshold": self.threshold}


@dataclass(frozen=True)
class DiscoveryConfig:
    """Causal discovery algorithm settings from ``analysis.yaml``.

    Attributes:
        algorithms: Algorithm names to run.
        use_background_knowledge: Whether to apply configured temporal tiers.
        pc: PC-specific settings.
        notears: NOTEARS-specific settings.
    """

    algorithms: tuple[str, ...] = ALLOWED_ALGORITHMS
    use_background_knowledge: bool = True
    pc: PCConfig = field(default_factory=PCConfig)
    notears: NotearsConfig = field(default_factory=NotearsConfig)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "DiscoveryConfig":
        """Build discovery configuration from YAML data.

        Args:
            value: Mapping under the ``discovery`` key.

        Returns:
            Parsed discovery configuration.

        Raises:
            ValueError: If no algorithm is selected or an algorithm is unknown.
        """
        data = _as_mapping(value or {}, "discovery")
        algorithms = tuple(str(item) for item in data.get("algorithms", cls.algorithms))
        unknown = set(algorithms).difference(ALLOWED_ALGORITHMS)
        if unknown:
            raise ValueError(f"discovery.algorithms contains unsupported values: {sorted(unknown)}")
        if not algorithms:
            raise ValueError("discovery.algorithms must not be empty")
        return cls(
            algorithms=algorithms,
            use_background_knowledge=_as_bool(
                data.get("use_background_knowledge", cls.use_background_knowledge),
                "discovery.use_background_knowledge",
            ),
            pc=PCConfig.from_mapping(data.get("pc")),
            notears=NotearsConfig.from_mapping(data.get("notears")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize discovery configuration to YAML-compatible data.

        Returns:
            Dictionary representation.
        """
        return {
            "algorithms": list(self.algorithms),
            "use_background_knowledge": self.use_background_knowledge,
            "pc": self.pc.to_dict(),
            "notears": self.notears.to_dict(),
        }


@dataclass(frozen=True)
class BootstrapConfig:
    """Bootstrap diagnostic settings from ``analysis.yaml``.

    Attributes:
        samples: Number of bootstrap samples.
        sample_fraction: Fraction of rows sampled per bootstrap.
    """

    samples: int = 100
    sample_fraction: float = 1.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "BootstrapConfig":
        """Build bootstrap configuration from YAML data.

        Args:
            value: Mapping under ``diagnostics.bootstrap``.

        Returns:
            Parsed bootstrap configuration.

        Raises:
            ValueError: If sample count or fraction is invalid.
        """
        data = _as_mapping(value or {}, "diagnostics.bootstrap")
        samples = int(data.get("samples", cls.samples))
        sample_fraction = float(data.get("sample_fraction", cls.sample_fraction))
        if samples < 0:
            raise ValueError("diagnostics.bootstrap.samples must be non-negative")
        if not 0 < sample_fraction <= 1:
            raise ValueError("diagnostics.bootstrap.sample_fraction must be in (0, 1]")
        return cls(samples=samples, sample_fraction=sample_fraction)

    def to_dict(self) -> dict[str, Any]:
        """Serialize bootstrap configuration to YAML-compatible data.

        Returns:
            Dictionary representation.
        """
        return {"samples": self.samples, "sample_fraction": self.sample_fraction}


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Diagnostic settings from ``analysis.yaml``.

    Attributes:
        bootstrap: Bootstrap edge-stability settings.
    """

    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "DiagnosticsConfig":
        """Build diagnostics configuration from YAML data.

        Args:
            value: Mapping under the ``diagnostics`` key.

        Returns:
            Parsed diagnostics configuration.
        """
        data = _as_mapping(value or {}, "diagnostics")
        return cls(bootstrap=BootstrapConfig.from_mapping(data.get("bootstrap")))

    def to_dict(self) -> dict[str, Any]:
        """Serialize diagnostics configuration to YAML-compatible data.

        Returns:
            Dictionary representation.
        """
        return {"bootstrap": self.bootstrap.to_dict()}


@dataclass(frozen=True)
class ReportingConfig:
    """Output switches from ``analysis.yaml``.

    Attributes:
        write_raw_input: Whether to write raw discovery input.
        write_processed_input: Whether to write transformed discovery input.
        write_standardized_input: Whether to write standardized input.
        write_variable_metadata: Whether to write variable metadata.
        write_variable_diagnostics: Whether to write variable diagnostics.
        write_algorithm_summary: Whether to write algorithm summary.
        write_graph_markdown: Whether to write graph Markdown reports.
        write_alpha_sensitivity: Whether to write PC alpha sensitivity outputs.
        write_bootstrap_stability: Whether to write bootstrap stability outputs.
    """

    write_raw_input: bool = True
    write_processed_input: bool = True
    write_standardized_input: bool = True
    write_variable_metadata: bool = True
    write_variable_diagnostics: bool = True
    write_algorithm_summary: bool = True
    write_graph_markdown: bool = True
    write_alpha_sensitivity: bool = True
    write_bootstrap_stability: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ReportingConfig":
        """Build reporting configuration from YAML data.

        Args:
            value: Mapping under the ``reporting`` key.

        Returns:
            Parsed reporting configuration.
        """
        data = _as_mapping(value or {}, "reporting")
        kwargs = {
            field_name: _as_bool(data.get(field_name, getattr(cls, field_name)), f"reporting.{field_name}")
            for field_name in cls.__dataclass_fields__
        }
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize reporting configuration to YAML-compatible data.

        Returns:
            Dictionary representation.
        """
        return {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}


@dataclass(frozen=True)
class AnalysisConfig:
    """Top-level analysis configuration.

    Attributes:
        dataset: Dataset registry settings.
        run: Runtime settings.
        preprocessing: Preprocessing settings.
        discovery: Discovery algorithm settings.
        diagnostics: Diagnostic settings.
        reporting: Output switches.
    """

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    run: RunConfig = field(default_factory=RunConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AnalysisConfig":
        """Build top-level analysis configuration from YAML data.

        Args:
            value: Parsed ``analysis.yaml`` mapping.

        Returns:
            Validated analysis configuration.
        """
        data = _as_mapping(value, "analysis config")
        return cls(
            dataset=DatasetConfig.from_mapping(data.get("dataset")),
            run=RunConfig.from_mapping(data.get("run")),
            preprocessing=PreprocessingConfig.from_mapping(data.get("preprocessing")),
            discovery=DiscoveryConfig.from_mapping(data.get("discovery")),
            diagnostics=DiagnosticsConfig.from_mapping(data.get("diagnostics")),
            reporting=ReportingConfig.from_mapping(data.get("reporting")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize analysis configuration to YAML-compatible data.

        Returns:
            Dictionary representation.
        """
        return {
            "dataset": self.dataset.to_dict(),
            "run": self.run.to_dict(),
            "preprocessing": self.preprocessing.to_dict(),
            "discovery": self.discovery.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
            "reporting": self.reporting.to_dict(),
        }


@dataclass(frozen=True)
class CategoryMidpoint:
    """One ordered category and its numeric midpoint.

    Attributes:
        label: Category label as it appears in the source data.
        midpoint: Numeric midpoint used for ordinal approximation.
    """

    label: str
    midpoint: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CategoryMidpoint":
        """Build a category midpoint from YAML data.

        Args:
            value: Mapping with ``label`` and ``midpoint`` keys.

        Returns:
            Parsed category midpoint.
        """
        data = _as_mapping(value, "category")
        return cls(label=str(data["label"]), midpoint=float(data["midpoint"]))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the category midpoint to YAML-compatible data.

        Returns:
            Dictionary representation.
        """
        return {"label": self.label, "midpoint": self.midpoint}


@dataclass(frozen=True)
class CategoricalMapping:
    """Mapping from ordered categorical values to numeric midpoints.

    Attributes:
        type: Mapping type, currently ``ordered_midpoint``.
        categories: Ordered categories and midpoint values.
        unknown_values: Source values treated as unknown.
        unit: Optional unit label for documentation.
    """

    type: str
    categories: tuple[CategoryMidpoint, ...]
    unknown_values: tuple[Any, ...] = ()
    unit: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CategoricalMapping":
        """Build a categorical mapping from YAML data.

        Args:
            value: Mapping under a ``categorical_mappings`` entry.

        Returns:
            Parsed categorical mapping.
        """
        data = _as_mapping(value, "categorical_mapping")
        return cls(
            type=str(data.get("type", "ordered_midpoint")),
            categories=tuple(CategoryMidpoint.from_mapping(item) for item in data.get("categories", [])),
            unknown_values=_as_tuple(data.get("unknown_values"), "categorical_mapping.unknown_values"),
            unit=str(data["unit"]) if data.get("unit") is not None else None,
        )

    def midpoint_map(self) -> dict[str, float]:
        """Return category label to midpoint mapping.

        Returns:
            Dictionary mapping source labels to numeric midpoints.
        """
        return {category.label: category.midpoint for category in self.categories}

    def median_midpoint(self) -> float:
        """Return the median of configured midpoint values.

        Returns:
            Median midpoint used as an imputation value.

        Raises:
            ValueError: If no categories are configured.
        """
        if not self.categories:
            raise ValueError("ordered midpoint mapping must contain categories")
        values = sorted(category.midpoint for category in self.categories)
        middle = len(values) // 2
        if len(values) % 2:
            return values[middle]
        return (values[middle - 1] + values[middle]) / 2

    def to_dict(self) -> dict[str, Any]:
        """Serialize the categorical mapping to YAML-compatible data.

        Returns:
            Dictionary representation.
        """
        return _clean_dict(
            {
                "type": self.type,
                "unit": self.unit,
                "unknown_values": list(self.unknown_values),
                "categories": [category.to_dict() for category in self.categories],
            }
        )


@dataclass(frozen=True)
class TableSpec:
    """Source table and column names used by feature engineering.

    Attributes:
        name: Dataset registry entry name.
        household_key: Household identifier column.
        campaign_id: Campaign identifier column.
        week: Transaction week column.
        start_day: Campaign start-day column.
        end_day: Campaign end-day column.
        transaction_timestamp: Transaction timestamp column.
    """

    name: str
    household_key: str | None = None
    campaign_id: str | None = None
    week: str | None = None
    start_day: str | None = None
    end_day: str | None = None
    transaction_timestamp: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TableSpec":
        """Build a table specification from YAML data.

        Args:
            value: Mapping under a ``tables`` entry.

        Returns:
            Parsed table specification.
        """
        data = _as_mapping(value, "table spec")
        return cls(
            name=str(data["name"]),
            household_key=str(data["household_key"]) if data.get("household_key") is not None else None,
            campaign_id=str(data["campaign_id"]) if data.get("campaign_id") is not None else None,
            week=str(data["week"]) if data.get("week") is not None else None,
            start_day=str(data["start_day"]) if data.get("start_day") is not None else None,
            end_day=str(data["end_day"]) if data.get("end_day") is not None else None,
            transaction_timestamp=(
                str(data["transaction_timestamp"]) if data.get("transaction_timestamp") is not None else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the table specification to YAML-compatible data.

        Returns:
            Dictionary representation.
        """
        return _clean_dict(
            {
                "name": self.name,
                "household_key": self.household_key,
                "campaign_id": self.campaign_id,
                "week": self.week,
                "start_day": self.start_day,
                "end_day": self.end_day,
                "transaction_timestamp": self.transaction_timestamp,
            }
        )


@dataclass(frozen=True)
class FeatureSpec:
    """Specification for one generated discovery feature.

    Attributes:
        name: Output feature name.
        source_table: Logical source table name.
        source_column: Source column name.
        transform: Allowlisted transform name.
        role: Analytical role, such as baseline, treatment, or outcome.
        data_type: Variable data type used for metadata.
        background_tier: Temporal tier for background knowledge.
        used_in_discovery: Whether the feature is included in discovery input.
        fisherz_caution: Whether Fisher-z assumptions need caution.
        mapping: Optional categorical mapping name.
        unknown_values: Values treated as unknown.
        value: Reference value for equality transforms.
        window: Transaction window name.
        aggregation: Household-level aggregation name.
        fill_value: Value used after household-level reindexing.
    """

    name: str
    source_table: str
    source_column: str
    transform: str = "identity"
    role: str | None = None
    data_type: str | None = None
    background_tier: str | None = None
    used_in_discovery: bool = True
    fisherz_caution: bool = False
    mapping: str | None = None
    unknown_values: tuple[Any, ...] = ()
    value: Any = None
    window: str | None = None
    aggregation: str | None = None
    fill_value: Any = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FeatureSpec":
        """Build a feature specification from YAML data.

        Args:
            value: One feature mapping from ``features.yaml``.

        Returns:
            Parsed feature specification.

        Raises:
            ValueError: If transform, aggregation, or required metadata is
                invalid.
        """
        data = _as_mapping(value, "feature spec")
        transform = str(data.get("transform", "identity"))
        if transform not in ALLOWED_TRANSFORMS:
            raise ValueError(f"feature transform is not supported: {transform}")
        aggregation = str(data["aggregation"]) if data.get("aggregation") is not None else None
        if aggregation is not None and aggregation not in ALLOWED_AGGREGATIONS:
            raise ValueError(f"feature aggregation is not supported: {aggregation}")
        spec = cls(
            name=str(data["name"]),
            source_table=str(data["source_table"]),
            source_column=str(data["source_column"]),
            transform=transform,
            role=str(data["role"]) if data.get("role") is not None else None,
            data_type=str(data["data_type"]) if data.get("data_type") is not None else None,
            background_tier=str(data["background_tier"]) if data.get("background_tier") is not None else None,
            used_in_discovery=bool(data.get("used_in_discovery", True)),
            fisherz_caution=bool(data.get("fisherz_caution", False)),
            mapping=str(data["mapping"]) if data.get("mapping") is not None else None,
            unknown_values=_as_tuple(data.get("unknown_values"), "feature.unknown_values"),
            value=data.get("value"),
            window=str(data["window"]) if data.get("window") is not None else None,
            aggregation=aggregation,
            fill_value=data.get("fill_value"),
        )
        if spec.used_in_discovery and (not spec.role or not spec.data_type):
            raise ValueError(f"used feature must define role and data_type: {spec.name}")
        return spec

    def to_dict(self) -> dict[str, Any]:
        """Serialize the feature specification to YAML-compatible data.

        Returns:
            Dictionary representation.
        """
        return _clean_dict(
            {
                "name": self.name,
                "source_table": self.source_table,
                "source_column": self.source_column,
                "window": self.window,
                "aggregation": self.aggregation,
                "fill_value": self.fill_value,
                "transform": self.transform,
                "mapping": self.mapping,
                "unknown_values": list(self.unknown_values) if self.unknown_values else None,
                "value": self.value,
                "role": self.role,
                "data_type": self.data_type,
                "background_tier": self.background_tier,
                "used_in_discovery": self.used_in_discovery,
                "fisherz_caution": self.fisherz_caution,
            }
        )


@dataclass(frozen=True)
class BackgroundKnowledgeConfig:
    """Temporal background knowledge settings from ``features.yaml``.

    Attributes:
        tier_order: Ordered causal tiers from earlier to later.
        forbid_backward_edges: Whether backward edges should be disallowed by
            the downstream background-knowledge object.
        tiers: Optional explicit mapping from tier name to feature names.
    """

    tier_order: tuple[str, ...]
    forbid_backward_edges: bool = True
    tiers: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "BackgroundKnowledgeConfig":
        """Build background knowledge configuration from YAML data.

        Args:
            value: Mapping under ``background_knowledge``.

        Returns:
            Parsed background knowledge configuration.
        """
        data = _as_mapping(value or {}, "background_knowledge")
        tiers = {
            str(tier): tuple(str(name) for name in names)
            for tier, names in _as_mapping(data.get("tiers", {}), "background_knowledge.tiers").items()
        }
        return cls(
            tier_order=tuple(str(item) for item in data.get("tier_order", ("baseline", "pre", "treatment", "outcome"))),
            forbid_backward_edges=bool(data.get("forbid_backward_edges", True)),
            tiers=tiers,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize background knowledge settings to YAML-compatible data.

        Returns:
            Dictionary representation.
        """
        data: dict[str, Any] = {
            "tier_order": list(self.tier_order),
            "forbid_backward_edges": self.forbid_backward_edges,
        }
        if self.tiers:
            data["tiers"] = {tier: list(names) for tier, names in self.tiers.items()}
        return data


@dataclass(frozen=True)
class FeatureConfig:
    """Top-level feature-generation configuration.

    Attributes:
        metadata: Global feature metadata such as entity and time columns.
        tables: Logical source table specifications.
        campaign_window: Campaign window calculation settings.
        categorical_mappings: Named categorical-to-numeric mappings.
        features: Feature specifications grouped by role or period.
        background_knowledge: Temporal tier configuration.
    """

    metadata: dict[str, Any]
    tables: dict[str, TableSpec]
    campaign_window: dict[str, Any]
    categorical_mappings: dict[str, CategoricalMapping]
    features: dict[str, tuple[FeatureSpec, ...]]
    background_knowledge: BackgroundKnowledgeConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FeatureConfig":
        """Build feature configuration from YAML data.

        Args:
            value: Parsed ``features.yaml`` mapping.

        Returns:
            Validated feature configuration.

        Raises:
            ValueError: If feature references, transforms, or tiers are invalid.
        """
        data = _as_mapping(value, "feature config")
        tables = {
            str(name): TableSpec.from_mapping(table)
            for name, table in _as_mapping(data.get("tables", {}), "tables").items()
        }
        mappings = {
            str(name): CategoricalMapping.from_mapping(mapping)
            for name, mapping in _as_mapping(data.get("categorical_mappings", {}), "categorical_mappings").items()
        }
        features = {
            str(group): tuple(FeatureSpec.from_mapping(item) for item in specs)
            for group, specs in _as_mapping(data.get("features", {}), "features").items()
        }
        config = cls(
            metadata=dict(_as_mapping(data.get("metadata", {}), "metadata")),
            tables=tables,
            campaign_window=dict(_as_mapping(data.get("campaign_window", {}), "campaign_window")),
            categorical_mappings=mappings,
            features=features,
            background_knowledge=BackgroundKnowledgeConfig.from_mapping(data.get("background_knowledge")),
        )
        config.validate()
        return config

    def all_features(self) -> tuple[FeatureSpec, ...]:
        """Return all configured feature specifications.

        Returns:
            Tuple of feature specifications in YAML group order.
        """
        return tuple(spec for specs in self.features.values() for spec in specs)

    def used_feature_specs(self) -> tuple[FeatureSpec, ...]:
        """Return features marked for discovery use.

        Returns:
            Tuple of feature specifications where ``used_in_discovery`` is true.
        """
        return tuple(spec for spec in self.all_features() if spec.used_in_discovery)

    def feature_by_name(self) -> dict[str, FeatureSpec]:
        """Index feature specifications by output name.

        Returns:
            Mapping from feature name to specification.
        """
        return {spec.name: spec for spec in self.all_features()}

    def features_for_source_table(self, source_table: str) -> tuple[FeatureSpec, ...]:
        """Select discovery features generated from one logical source table.

        Args:
            source_table: Logical table name from ``features.yaml``.

        Returns:
            Tuple of used feature specifications for that table.
        """
        return tuple(spec for spec in self.used_feature_specs() if spec.source_table == source_table)

    def background_tiers_for_nodes(self, node_names: list[str]) -> list[set[str]]:
        """Build tier assignments for retained discovery nodes.

        Args:
            node_names: Variables retained in the discovery input.

        Returns:
            List of node-name sets ordered by configured tier order.
        """
        node_set = set(node_names)
        feature_by_name = self.feature_by_name()
        tier_map: dict[str, set[str]] = {tier: set() for tier in self.background_knowledge.tier_order}

        for spec in feature_by_name.values():
            if spec.name in node_set and spec.background_tier in tier_map:
                tier_map[spec.background_tier].add(spec.name)

        for tier, names in self.background_knowledge.tiers.items():
            if tier not in tier_map:
                continue
            tier_map[tier].update(name for name in names if name in node_set)

        return [tier_map[tier] for tier in self.background_knowledge.tier_order]

    def validate(self) -> None:
        """Validate cross-field references in the feature configuration.

        Raises:
            ValueError: If feature names are duplicated, references point to
                unknown tables/mappings/tiers, or transaction features are
                incomplete.
        """
        names = [spec.name for spec in self.all_features()]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"features[*].name must be unique: {duplicates}")

        known_tables = set(self.tables)
        known_mappings = set(self.categorical_mappings)
        tier_order = set(self.background_knowledge.tier_order)
        feature_names = set(names)

        for spec in self.all_features():
            if spec.source_table not in known_tables:
                raise ValueError(f"feature source_table is unknown: {spec.name} -> {spec.source_table}")
            if spec.mapping is not None and spec.mapping not in known_mappings:
                raise ValueError(f"feature mapping is unknown: {spec.name} -> {spec.mapping}")
            if spec.background_tier is not None and spec.background_tier not in tier_order:
                raise ValueError(f"feature background_tier is unknown: {spec.name} -> {spec.background_tier}")
            if spec.source_table == "transactions" and (spec.window is None or spec.aggregation is None):
                raise ValueError(f"transaction feature must define window and aggregation: {spec.name}")

        for tier, tier_features in self.background_knowledge.tiers.items():
            if tier not in tier_order:
                raise ValueError(f"background_knowledge.tiers contains unknown tier: {tier}")
            unknown_features = set(tier_features).difference(feature_names)
            if unknown_features:
                raise ValueError(
                    f"background_knowledge.tiers.{tier} contains unknown features: {sorted(unknown_features)}"
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize feature configuration to YAML-compatible data.

        Returns:
            Dictionary representation.
        """
        return {
            "metadata": self.metadata,
            "tables": {name: table.to_dict() for name, table in self.tables.items()},
            "campaign_window": self.campaign_window,
            "categorical_mappings": {
                name: mapping.to_dict() for name, mapping in self.categorical_mappings.items()
            },
            "features": {
                group: [spec.to_dict() for spec in specs]
                for group, specs in self.features.items()
            },
            "background_knowledge": self.background_knowledge.to_dict(),
        }
