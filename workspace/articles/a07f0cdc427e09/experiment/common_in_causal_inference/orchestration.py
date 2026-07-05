"""因果探索から因果推論までを連続実行する薄い orchestration CLI。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """統合 entrypoint 用の CLI parser を作る。

    Returns:
        因果探索と因果推論の主要な共通上書きを受け取る parser。
    """
    parser = argparse.ArgumentParser(
        description="Run causal discovery first, then causal inference for Complete Journey.",
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--dataset-yaml", type=Path, default=None)
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--pre-weeks", type=int, default=None)
    parser.add_argument("--collinearity-threshold", type=float, default=None)

    parser.add_argument("--discovery-analysis-config", type=Path, default=None)
    parser.add_argument("--discovery-feature-config", type=Path, default=None)
    parser.add_argument("--discovery-output-dir", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--pc-indep-test", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=None)

    parser.add_argument("--inference-config", type=Path, default=None)
    parser.add_argument("--inference-feature-config", type=Path, default=None)
    parser.add_argument("--inference-output-dir", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=("edge_weight", "treatment_effect"),
        default=None,
    )
    parser.add_argument("--treatment", default=None)
    parser.add_argument("--outcome", default=None)
    parser.add_argument("--estimand", choices=("ATE", "ATT"), default=None)
    parser.add_argument(
        "--effect-methods",
        nargs="+",
        choices=("diff_in_means", "ols", "ipw", "aipw"),
        default=None,
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """統合 entrypoint の引数を解釈する。"""
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """因果探索を実行した後、同じ実験設定を使って因果推論を実行する。

    Args:
        argv: テストや notebook から渡す任意の引数列。``None`` の場合は
            ``sys.argv`` が使われる。

    Notes:
        この関数は orchestration だけを担当する。探索・推論の実装は既存の
        各 pipeline CLI に委譲し、設定の解釈や出力生成の責務を重複させない。
    """
    args = parse_args(argv)

    from causal_discovery_pipeline.cli import main as discovery_main
    from causal_inference_pipeline.cli import main as inference_main

    discovery_args = _build_discovery_args(args)
    inference_args = _build_inference_args(args)

    print("stage: causal_discovery")
    discovery_main(discovery_args)
    print("stage: causal_inference")
    inference_main(inference_args)


def _build_discovery_args(args: argparse.Namespace) -> list[str]:
    """探索 CLI に渡す引数列を組み立てる。"""
    output: list[str] = []
    _append_path(output, "--project-root", args.project_root)
    _append_path(output, "--analysis-config", args.discovery_analysis_config)
    _append_path(output, "--feature-config", args.discovery_feature_config)
    _append_path(output, "--dataset-yaml", args.dataset_yaml)
    _append_value(output, "--campaign-id", args.campaign_id)
    _append_value(output, "--pre-weeks", args.pre_weeks)
    _append_value(output, "--collinearity-threshold", args.collinearity_threshold)
    _append_path(output, "--output-dir", args.discovery_output_dir)
    _append_value(output, "--alpha", args.alpha)
    _append_value(output, "--pc-indep-test", args.pc_indep_test)
    _append_value(output, "--bootstrap-samples", args.bootstrap_samples)
    return output


def _build_inference_args(args: argparse.Namespace) -> list[str]:
    """推論 CLI に渡す引数列を組み立てる。"""
    output: list[str] = []
    _append_path(output, "--project-root", args.project_root)
    _append_path(output, "--config", args.inference_config)
    _append_path(output, "--feature-config", args.inference_feature_config)
    _append_path(output, "--dataset-yaml", args.dataset_yaml)
    _append_value(output, "--campaign-id", args.campaign_id)
    _append_value(output, "--pre-weeks", args.pre_weeks)
    _append_value(output, "--collinearity-threshold", args.collinearity_threshold)
    _append_path(output, "--discovery-dir", args.discovery_output_dir)
    _append_path(output, "--output-dir", args.inference_output_dir)
    _append_value(output, "--mode", args.mode)
    _append_value(output, "--treatment", args.treatment)
    _append_value(output, "--outcome", args.outcome)
    _append_value(output, "--estimand", args.estimand)
    if args.effect_methods is not None:
        output.append("--effect-methods")
        output.extend(str(value) for value in args.effect_methods)
    return output


def _append_value(output: list[str], flag: str, value: object | None) -> None:
    """値が ``None`` でないときだけ CLI 引数に追加する。"""
    if value is None:
        return
    output.extend([flag, str(value)])


def _append_path(output: list[str], flag: str, value: Path | None) -> None:
    """Path 値が ``None`` でないときだけ CLI 引数に追加する。"""
    if value is None:
        return
    output.extend([flag, str(value)])
