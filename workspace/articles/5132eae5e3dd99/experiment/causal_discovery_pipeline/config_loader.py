from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config_schema import (
    AnalysisConfig,
    DatasetConfig,
    DiagnosticsConfig,
    DiscoveryConfig,
    FeatureConfig,
    PCConfig,
    PreprocessingConfig,
    RunConfig,
)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary.

    Args:
        path: YAML file path.

    Returns:
        Parsed YAML mapping. Empty files are returned as an empty dictionary.

    Raises:
        ValueError: If the YAML root is not a mapping.
    """
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return dict(data)


def load_analysis_config(path: Path) -> AnalysisConfig:
    """Load and validate an analysis configuration file.

    Args:
        path: Path to ``analysis.yaml``.

    Returns:
        Validated analysis configuration.
    """
    return AnalysisConfig.from_mapping(load_yaml(path))


def load_feature_config(path: Path) -> FeatureConfig:
    """Load and validate a feature configuration file.

    Args:
        path: Path to ``features.yaml``.

    Returns:
        Validated feature configuration.
    """
    return FeatureConfig.from_mapping(load_yaml(path))


def resolve_project_path(path: Path, project_root: Path) -> Path:
    """Resolve a possibly project-relative path.

    Args:
        path: Absolute or project-relative path.
        project_root: Repository root used for relative paths.

    Returns:
        Absolute path when ``path`` is relative; otherwise ``path`` unchanged.
    """
    return path if path.is_absolute() else project_root / path


def merge_cli_overrides(
    analysis_config: AnalysisConfig,
    args: argparse.Namespace,
) -> AnalysisConfig:
    """Apply command-line overrides to an analysis configuration.

    Args:
        analysis_config: Base configuration loaded from YAML.
        args: Parsed command-line namespace. ``None`` values are ignored.

    Returns:
        New validated configuration after applying overrides.

    Raises:
        ValueError: If the resulting configuration violates schema rules.
    """
    dataset = analysis_config.dataset
    run = analysis_config.run
    preprocessing = analysis_config.preprocessing
    discovery = analysis_config.discovery
    diagnostics = analysis_config.diagnostics
    pc = discovery.pc
    notears = discovery.notears
    bootstrap = diagnostics.bootstrap

    if args.dataset_yaml is not None:
        dataset = DatasetConfig(yaml_path=args.dataset_yaml)

    if args.campaign_id is not None:
        run = replace(run, campaign_id=str(args.campaign_id))
    if args.pre_weeks is not None:
        run = replace(run, pre_weeks=int(args.pre_weeks))
    if args.output_dir is not None:
        run = replace(run, output_dir=args.output_dir)
    if args.random_seed is not None:
        run = replace(run, random_seed=int(args.random_seed))

    if args.collinearity_threshold is not None:
        preprocessing = PreprocessingConfig(
            collinearity_threshold=float(args.collinearity_threshold)
        )

    if args.algorithms is not None:
        discovery = replace(discovery, algorithms=tuple(args.algorithms))
    if args.no_background_knowledge is not None:
        discovery = replace(
            discovery,
            use_background_knowledge=not bool(args.no_background_knowledge),
        )

    if args.alpha is not None:
        pc = replace(pc, alpha=float(args.alpha))
    if args.pc_indep_test is not None:
        pc = replace(pc, indep_test=str(args.pc_indep_test))
    if args.alpha_grid is not None:
        pc = replace(pc, alpha_grid=tuple(float(value) for value in args.alpha_grid))
    if args.pc_discrete_bins is not None:
        pc = replace(pc, discrete_bins=int(args.pc_discrete_bins))
    if args.notears_threshold is not None:
        notears = replace(notears, threshold=float(args.notears_threshold))

    if args.bootstrap_samples is not None:
        bootstrap = replace(bootstrap, samples=int(args.bootstrap_samples))
    if args.bootstrap_sample_fraction is not None:
        bootstrap = replace(
            bootstrap,
            sample_fraction=float(args.bootstrap_sample_fraction),
        )

    discovery = DiscoveryConfig(
        algorithms=discovery.algorithms,
        use_background_knowledge=discovery.use_background_knowledge,
        pc=PCConfig(
            alpha=pc.alpha,
            indep_test=pc.indep_test,
            allowed_indep_tests=pc.allowed_indep_tests,
            discrete_indep_tests=pc.discrete_indep_tests,
            discrete_bins=pc.discrete_bins,
            alpha_grid=pc.alpha_grid,
        ),
        notears=notears,
    )
    diagnostics = DiagnosticsConfig(bootstrap=bootstrap)

    merged = AnalysisConfig(
        dataset=dataset,
        run=run,
        preprocessing=preprocessing,
        discovery=discovery,
        diagnostics=diagnostics,
        reporting=analysis_config.reporting,
    )
    return AnalysisConfig.from_mapping(merged.to_dict())


def write_resolved_config(
    *,
    analysis_config: AnalysisConfig,
    feature_config: FeatureConfig,
    output_dir: Path,
) -> None:
    """Write resolved YAML configurations for reproducibility.

    Args:
        analysis_config: Final analysis configuration after CLI overrides.
        feature_config: Validated feature configuration.
        output_dir: Directory where resolved config files are written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_analysis_config.yaml").write_text(
        yaml.safe_dump(
            analysis_config.to_dict(),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "resolved_features_config.yaml").write_text(
        yaml.safe_dump(
            feature_config.to_dict(),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
