"""Command-line interface for the causal inference pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from myproj.io.config_resolver import find_project_root

from .config import (
    load_pipeline_config,
    merge_cli_overrides,
    resolve_project_path,
    write_resolved_configs,
)
from .constants import (
    DEFAULT_CONFIG_PATH,
    SUPPORTED_ADJUSTMENT_STRATEGIES,
    SUPPORTED_ALGORITHMS,
    SUPPORTED_EFFECT_METHODS,
    SUPPORTED_ESTIMANDS,
    SUPPORTED_MODES,
    SUPPORTED_ROBUST_SE,
)
from .context import RunContext
from .data.loader import CompleteJourneyDataLoader
from .features.builder import FeatureBuilder
from .features.config_schema import load_feature_config
from .modes.registry import MODE_STRATEGY_BY_NAME


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the inference workflow.

    Returns:
        Configured argument parser. Non-``None`` parsed values override the
        YAML configuration.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Estimate causal-discovery edge weights or explicit treatment effects "
            "for completejourney."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Pipeline YAML. Defaults to {DEFAULT_CONFIG_PATH} when present.",
    )
    parser.add_argument("--feature-config", type=Path, default=None)
    parser.add_argument("--mode", choices=SUPPORTED_MODES, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--dataset-yaml", type=Path, default=None)
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--pre-weeks", type=int, default=None)
    parser.add_argument("--collinearity-threshold", type=float, default=None)
    parser.add_argument("--discovery-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        choices=SUPPORTED_ALGORITHMS,
    )
    parser.add_argument(
        "--edge-robust-se",
        choices=SUPPORTED_ROBUST_SE,
        default=None,
        help="Robust SE type for edge-weight regressions.",
    )
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--treatment", default=None)
    parser.add_argument("--outcome", default=None)
    parser.add_argument("--estimand", choices=SUPPORTED_ESTIMANDS, default=None)
    parser.add_argument(
        "--adjustment-strategy",
        choices=SUPPORTED_ADJUSTMENT_STRATEGIES,
        default=None,
    )
    parser.add_argument("--covariates", nargs="*", default=None)
    parser.add_argument(
        "--effect-methods",
        nargs="+",
        choices=SUPPORTED_EFFECT_METHODS,
        default=None,
    )
    parser.add_argument(
        "--robust-se",
        choices=SUPPORTED_ROBUST_SE,
        default=None,
        help="Robust SE type for treatment-effect OLS.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional command-line arguments. If ``None``, arguments are read
            from ``sys.argv``.

    Returns:
        Parsed command-line namespace.
    """
    return build_parser().parse_args(argv)


def resolve_config_path(path: Path | None, project_root: Path) -> Path | None:
    """Resolve the selected or default pipeline config path.

    Args:
        path: User-supplied config path.
        project_root: Repository root.

    Returns:
        Existing resolved path, or ``None`` when no config file is available.
    """
    candidate = path if path is not None else DEFAULT_CONFIG_PATH
    resolved = candidate if candidate.is_absolute() else project_root / candidate
    return resolved if resolved.exists() else None


def main(argv: Sequence[str] | None = None) -> None:
    """Runs the causal inference pipeline from command-line arguments.

    Args:
        argv: Optional command-line arguments. If ``None``, arguments are read
            from ``sys.argv``.

    Raises:
        ValueError: If the resolved configuration is invalid.
        FileNotFoundError: If required input files do not exist.
    """
    args = parse_args(argv)
    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else find_project_root(Path.cwd())
    )
    config_path = resolve_config_path(args.config, project_root)
    config = merge_cli_overrides(load_pipeline_config(config_path), args)

    dataset_yaml = resolve_project_path(config.data.dataset_yaml, project_root)
    discovery_dir = resolve_project_path(config.data.discovery_dir, project_root)
    output_dir = resolve_project_path(config.data.output_dir, project_root)
    feature_config_path = resolve_project_path(config.feature_config_path, project_root)
    feature_config = load_feature_config(feature_config_path)

    data_loader = CompleteJourneyDataLoader(
        project_root=project_root,
        dataset_yaml=dataset_yaml,
        table_specs=feature_config.tables,
    )
    tables = data_loader.load_all()

    preprocessing_result = FeatureBuilder(feature_config).build(
        tables=tables,
        campaign_id=config.data.campaign_id,
        pre_weeks=config.data.pre_weeks,
        collinearity_threshold=config.data.collinearity_threshold,
    )
    resolved_config = config.__class__.from_mapping(
        {
            **config.to_dict(),
            "data": {
                **config.data.to_dict(),
                "dataset_yaml": str(dataset_yaml),
                "discovery_dir": str(discovery_dir),
                "output_dir": str(output_dir),
            },
            "feature_config_path": str(feature_config_path),
        }
    )

    if config.report.write_config_snapshot:
        write_resolved_configs(
            output_dir=output_dir,
            pipeline_config=resolved_config,
            feature_config_data=feature_config.to_dict(),
        )

    context = RunContext(
        config=resolved_config,
        feature_config=feature_config,
        project_root=project_root,
        dataset_yaml=dataset_yaml,
        discovery_dir=discovery_dir,
        output_dir=output_dir,
        preprocessing_result=preprocessing_result,
    )
    strategy = MODE_STRATEGY_BY_NAME[config.mode]()
    strategy.run(context)

