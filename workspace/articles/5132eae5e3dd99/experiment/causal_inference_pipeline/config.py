"""Pipeline-level YAML configuration loading and CLI override handling."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from .constants import (
    DEFAULT_DATASET_YAML,
    DEFAULT_DISCOVERY_DIR,
    DEFAULT_FEATURE_CONFIG_PATH,
    DEFAULT_OUTPUT_DIR,
    SUPPORTED_ADJUSTMENT_STRATEGIES,
    SUPPORTED_ALGORITHMS,
    SUPPORTED_EFFECT_METHODS,
    SUPPORTED_ESTIMANDS,
    SUPPORTED_MODES,
    SUPPORTED_ROBUST_SE,
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Validate that a parsed YAML value is a mapping.

    Args:
        value: Parsed YAML value.
        name: Human-readable field name used in error messages.

    Returns:
        Mapping view of ``value``.

    Raises:
        ValueError: If ``value`` is not a mapping.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _tuple(value: Any, name: str) -> tuple[Any, ...]:
    """Normalize a YAML sequence to a tuple.

    Args:
        value: Parsed YAML value.
        name: Human-readable field name used in error messages.

    Returns:
        Tuple representation. ``None`` becomes an empty tuple.

    Raises:
        ValueError: If ``value`` is not a list or tuple.
    """
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise ValueError(f"{name} must be a list")
    return tuple(value)


def _validate_choice(value: str, choices: tuple[str, ...], name: str) -> str:
    """Validate that a string belongs to a fixed set of choices.

    Args:
        value: Candidate value.
        choices: Allowed values.
        name: Human-readable field name used in error messages.

    Returns:
        The validated value.

    Raises:
        ValueError: If ``value`` is not in ``choices``.
    """
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}: {value}")
    return value


@dataclass(frozen=True)
class DataConfig:
    """Data and path settings for a pipeline run.

    Attributes:
        dataset_yaml: Project-relative or absolute dataset registry YAML path.
        campaign_id: Campaign identifier used to define treatment.
        pre_weeks: Number of pre-treatment weeks.
        collinearity_threshold: Absolute correlation threshold used when
            pruning standardized features.
        discovery_dir: Directory containing causal discovery outputs.
        output_dir: Directory where inference outputs are written.
    """

    dataset_yaml: Path = DEFAULT_DATASET_YAML
    campaign_id: str = "18"
    pre_weeks: int = 8
    collinearity_threshold: float = 0.995
    discovery_dir: Path = DEFAULT_DISCOVERY_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "DataConfig":
        """Build data configuration from YAML.

        Args:
            value: Mapping under the ``data`` key.

        Returns:
            Parsed data configuration.

        Raises:
            ValueError: If numeric values are outside valid ranges.
        """
        data = _mapping(value or {}, "data")
        pre_weeks = int(data.get("pre_weeks", cls.pre_weeks))
        if pre_weeks < 1:
            raise ValueError("data.pre_weeks must be positive")
        threshold = float(data.get("collinearity_threshold", cls.collinearity_threshold))
        if not 0 < threshold <= 1:
            raise ValueError("data.collinearity_threshold must be in (0, 1]")
        return cls(
            dataset_yaml=Path(data.get("dataset_yaml", cls.dataset_yaml)),
            campaign_id=str(data.get("campaign_id", cls.campaign_id)),
            pre_weeks=pre_weeks,
            collinearity_threshold=threshold,
            discovery_dir=Path(data.get("discovery_dir", cls.discovery_dir)),
            output_dir=Path(data.get("output_dir", cls.output_dir)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this configuration to YAML-compatible data.

        Returns:
            Dictionary representation.
        """
        return {
            "dataset_yaml": str(self.dataset_yaml),
            "campaign_id": self.campaign_id,
            "pre_weeks": self.pre_weeks,
            "collinearity_threshold": self.collinearity_threshold,
            "discovery_dir": str(self.discovery_dir),
            "output_dir": str(self.output_dir),
        }


@dataclass(frozen=True)
class EdgeWeightConfig:
    """Configuration for discovered-edge coefficient estimation.

    Attributes:
        algorithms: Causal discovery algorithms whose ``edges.csv`` files are
            read.
        robust_se: Standard error type for edge regressions.
        min_samples: Minimum complete-case sample size for each edge model.
    """

    algorithms: tuple[str, ...] = SUPPORTED_ALGORITHMS
    robust_se: str = "none"
    min_samples: int = 30

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "EdgeWeightConfig":
        """Build edge-weight configuration from YAML.

        Args:
            value: Mapping under the ``edge_weight`` key.

        Returns:
            Parsed edge-weight configuration.

        Raises:
            ValueError: If an algorithm or robust SE type is unsupported.
        """
        data = _mapping(value or {}, "edge_weight")
        algorithms = tuple(str(item) for item in _tuple(data.get("algorithms", cls.algorithms), "edge_weight.algorithms"))
        for algorithm in algorithms:
            _validate_choice(algorithm, SUPPORTED_ALGORITHMS, "edge_weight.algorithms")
        robust_se = _validate_choice(
            str(data.get("robust_se", cls.robust_se)),
            SUPPORTED_ROBUST_SE,
            "edge_weight.robust_se",
        )
        min_samples = int(data.get("min_samples", cls.min_samples))
        if min_samples < 1:
            raise ValueError("edge_weight.min_samples must be positive")
        return cls(algorithms=algorithms, robust_se=robust_se, min_samples=min_samples)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this configuration to YAML-compatible data.

        Returns:
            Dictionary representation.
        """
        return {
            "algorithms": list(self.algorithms),
            "robust_se": self.robust_se,
            "min_samples": self.min_samples,
        }


@dataclass(frozen=True)
class TreatmentEffectConfig:
    """Configuration for explicit treatment effect estimation.

    Attributes:
        treatment: Binary treatment column.
        outcome: Outcome column.
        estimand: Target estimand, either ``ATE`` or ``ATT``.
        adjustment_strategy: Covariate selection strategy.
        covariates: Manual covariates used when strategy is ``manual``.
        effect_methods: Estimators to run.
        robust_se: Robust standard error type for OLS.
        propensity_clip: Lower and upper propensity clipping bounds.
        cross_fitting_folds: Number of AIPW cross-fitting folds. Current
            implementation preserves the legacy no-sample-splitting behavior
            when this value is ``0``.
    """

    treatment: str = "treated"
    outcome: str = "outcome_sales_value"
    estimand: str = "ATE"
    adjustment_strategy: str = "pre_treatment_covariates"
    covariates: tuple[str, ...] | None = None
    effect_methods: tuple[str, ...] = ("diff_in_means", "ols", "ipw")
    robust_se: str = "HC3"
    propensity_clip: tuple[float, float] = (0.01, 0.99)
    cross_fitting_folds: int = 0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "TreatmentEffectConfig":
        """Build treatment-effect configuration from YAML.

        Args:
            value: Mapping under the ``treatment_effect`` key.

        Returns:
            Parsed treatment-effect configuration.

        Raises:
            ValueError: If choices or propensity clip bounds are invalid.
        """
        data = _mapping(value or {}, "treatment_effect")
        estimand = _validate_choice(
            str(data.get("estimand", cls.estimand)),
            SUPPORTED_ESTIMANDS,
            "treatment_effect.estimand",
        )
        strategy = _validate_choice(
            str(data.get("adjustment_strategy", cls.adjustment_strategy)),
            SUPPORTED_ADJUSTMENT_STRATEGIES,
            "treatment_effect.adjustment_strategy",
        )
        methods = tuple(
            str(item)
            for item in _tuple(
                data.get("effect_methods", cls.effect_methods),
                "treatment_effect.effect_methods",
            )
        )
        for method in methods:
            _validate_choice(method, SUPPORTED_EFFECT_METHODS, "treatment_effect.effect_methods")
        robust_se = _validate_choice(
            str(data.get("robust_se", cls.robust_se)),
            SUPPORTED_ROBUST_SE,
            "treatment_effect.robust_se",
        )
        clip_data = data.get("propensity_clip", {})
        if isinstance(clip_data, Mapping):
            lower = float(clip_data.get("lower", cls.propensity_clip[0]))
            upper = float(clip_data.get("upper", cls.propensity_clip[1]))
        elif isinstance(clip_data, list | tuple) and len(clip_data) == 2:
            lower = float(clip_data[0])
            upper = float(clip_data[1])
        else:
            raise ValueError("treatment_effect.propensity_clip must be a mapping or pair")
        if not 0 < lower < upper < 1:
            raise ValueError("treatment_effect.propensity_clip must satisfy 0 < lower < upper < 1")
        covariates = data.get("covariates", cls.covariates)
        parsed_covariates = (
            None
            if covariates is None
            else tuple(str(item) for item in _tuple(covariates, "treatment_effect.covariates"))
        )
        folds = int(data.get("cross_fitting_folds", cls.cross_fitting_folds))
        if folds < 0:
            raise ValueError("treatment_effect.cross_fitting_folds must be non-negative")
        return cls(
            treatment=str(data.get("treatment", cls.treatment)),
            outcome=str(data.get("outcome", cls.outcome)),
            estimand=estimand,
            adjustment_strategy=strategy,
            covariates=parsed_covariates,
            effect_methods=methods,
            robust_se=robust_se,
            propensity_clip=(lower, upper),
            cross_fitting_folds=folds,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this configuration to YAML-compatible data.

        Returns:
            Dictionary representation.
        """
        return {
            "treatment": self.treatment,
            "outcome": self.outcome,
            "estimand": self.estimand,
            "adjustment_strategy": self.adjustment_strategy,
            "covariates": None if self.covariates is None else list(self.covariates),
            "effect_methods": list(self.effect_methods),
            "robust_se": self.robust_se,
            "propensity_clip": {
                "lower": self.propensity_clip[0],
                "upper": self.propensity_clip[1],
            },
            "cross_fitting_folds": self.cross_fitting_folds,
        }


@dataclass(frozen=True)
class ReportConfig:
    """Output and report-writing settings.

    Attributes:
        language: Report language marker.
        write_markdown: Whether Markdown reports should be written.
        write_csv: Whether CSV tables should be written.
        write_config_snapshot: Whether resolved YAML snapshots should be
            written.
    """

    language: str = "ja"
    write_markdown: bool = True
    write_csv: bool = True
    write_config_snapshot: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ReportConfig":
        """Build report configuration from YAML.

        Args:
            value: Mapping under the ``report`` key.

        Returns:
            Parsed report configuration.
        """
        data = _mapping(value or {}, "report")
        return cls(
            language=str(data.get("language", cls.language)),
            write_markdown=bool(data.get("write_markdown", cls.write_markdown)),
            write_csv=bool(data.get("write_csv", cls.write_csv)),
            write_config_snapshot=bool(
                data.get("write_config_snapshot", cls.write_config_snapshot)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this configuration to YAML-compatible data.

        Returns:
            Dictionary representation.
        """
        return {
            "language": self.language,
            "write_markdown": self.write_markdown,
            "write_csv": self.write_csv,
            "write_config_snapshot": self.write_config_snapshot,
        }


@dataclass(frozen=True)
class PipelineConfig:
    """Resolved pipeline configuration.

    Attributes:
        mode: Analysis mode to run.
        data: Data and path settings.
        feature_config_path: Feature construction YAML path.
        edge_weight: Edge-weight mode settings.
        treatment_effect: Treatment-effect mode settings.
        report: Output report settings.
    """

    mode: str = "edge_weight"
    data: DataConfig = DataConfig()
    feature_config_path: Path = DEFAULT_FEATURE_CONFIG_PATH
    edge_weight: EdgeWeightConfig = EdgeWeightConfig()
    treatment_effect: TreatmentEffectConfig = TreatmentEffectConfig()
    report: ReportConfig = ReportConfig()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "PipelineConfig":
        """Build a full pipeline configuration from YAML data.

        Args:
            value: Parsed YAML mapping.

        Returns:
            Validated pipeline configuration.

        Raises:
            ValueError: If the selected mode is unsupported.
        """
        data = _mapping(value or {}, "pipeline")
        mode = _validate_choice(str(data.get("mode", cls.mode)), SUPPORTED_MODES, "mode")
        return cls(
            mode=mode,
            data=DataConfig.from_mapping(data.get("data")),
            feature_config_path=Path(
                data.get("feature_config_path", cls.feature_config_path)
            ),
            edge_weight=EdgeWeightConfig.from_mapping(data.get("edge_weight")),
            treatment_effect=TreatmentEffectConfig.from_mapping(data.get("treatment_effect")),
            report=ReportConfig.from_mapping(data.get("report")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this configuration to YAML-compatible data.

        Returns:
            Dictionary representation.
        """
        return {
            "mode": self.mode,
            "data": self.data.to_dict(),
            "feature_config_path": str(self.feature_config_path),
            "edge_weight": self.edge_weight.to_dict(),
            "treatment_effect": self.treatment_effect.to_dict(),
            "report": self.report.to_dict(),
        }


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk.

    Args:
        path: YAML file path.

    Returns:
        Parsed mapping. Empty YAML files produce an empty dictionary.

    Raises:
        ValueError: If the YAML root is not a mapping.
    """
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return dict(data)


def load_pipeline_config(path: Path | None) -> PipelineConfig:
    """Load a pipeline config, falling back to built-in defaults.

    Args:
        path: Optional YAML path. If ``None`` or missing, built-in defaults are
            used.

    Returns:
        Parsed pipeline configuration.
    """
    if path is None or not path.exists():
        return PipelineConfig()
    return PipelineConfig.from_mapping(load_yaml(path))


def resolve_project_path(path: Path, project_root: Path) -> Path:
    """Resolve a path relative to the project root.

    Args:
        path: Absolute or project-relative path.
        project_root: Repository root.

    Returns:
        Absolute path if ``path`` was relative.
    """
    return path if path.is_absolute() else project_root / path


def merge_cli_overrides(config: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    """Apply explicit CLI arguments over a YAML configuration.

    Args:
        config: Base configuration from YAML or built-in defaults.
        args: Parsed CLI namespace. Only non-``None`` values are applied.

    Returns:
        New validated configuration.
    """
    data = config.data
    edge_weight = config.edge_weight
    treatment_effect = config.treatment_effect
    mode = config.mode
    feature_config_path = config.feature_config_path

    if args.mode is not None:
        mode = args.mode
    if args.dataset_yaml is not None:
        data = replace(data, dataset_yaml=args.dataset_yaml)
    if args.campaign_id is not None:
        data = replace(data, campaign_id=str(args.campaign_id))
    if args.pre_weeks is not None:
        data = replace(data, pre_weeks=int(args.pre_weeks))
    if args.collinearity_threshold is not None:
        data = replace(data, collinearity_threshold=float(args.collinearity_threshold))
    if args.discovery_dir is not None:
        data = replace(data, discovery_dir=args.discovery_dir)
    if args.output_dir is not None:
        data = replace(data, output_dir=args.output_dir)
    if args.feature_config is not None:
        feature_config_path = args.feature_config
    if args.algorithms is not None:
        edge_weight = replace(edge_weight, algorithms=tuple(args.algorithms))
    if args.edge_robust_se is not None:
        edge_weight = replace(edge_weight, robust_se=args.edge_robust_se)
    if args.min_samples is not None:
        edge_weight = replace(edge_weight, min_samples=int(args.min_samples))
    if args.treatment is not None:
        treatment_effect = replace(treatment_effect, treatment=args.treatment)
    if args.outcome is not None:
        treatment_effect = replace(treatment_effect, outcome=args.outcome)
    if args.estimand is not None:
        treatment_effect = replace(treatment_effect, estimand=args.estimand)
    if args.adjustment_strategy is not None:
        treatment_effect = replace(
            treatment_effect,
            adjustment_strategy=args.adjustment_strategy,
        )
    if args.covariates is not None:
        treatment_effect = replace(treatment_effect, covariates=tuple(args.covariates))
    if args.effect_methods is not None:
        treatment_effect = replace(
            treatment_effect,
            effect_methods=tuple(args.effect_methods),
        )
    if args.robust_se is not None:
        treatment_effect = replace(treatment_effect, robust_se=args.robust_se)

    return PipelineConfig.from_mapping(
        {
            "mode": mode,
            "data": data.to_dict(),
            "feature_config_path": str(feature_config_path),
            "edge_weight": edge_weight.to_dict(),
            "treatment_effect": treatment_effect.to_dict(),
            "report": config.report.to_dict(),
        }
    )


def write_resolved_configs(
    *,
    output_dir: Path,
    pipeline_config: PipelineConfig,
    feature_config_data: Mapping[str, Any],
) -> None:
    """Write resolved config snapshots for reproducibility.

    Args:
        output_dir: Directory receiving snapshot files.
        pipeline_config: Resolved pipeline configuration.
        feature_config_data: Resolved feature configuration as plain data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(pipeline_config.to_dict(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (output_dir / "resolved_feature_config.yaml").write_text(
        yaml.safe_dump(dict(feature_config_data), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

