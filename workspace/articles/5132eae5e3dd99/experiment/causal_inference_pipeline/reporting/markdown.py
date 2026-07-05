"""Markdown report renderers."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from ..config import PipelineConfig
from .tables import dataframe_to_markdown


def render_edge_weight_report(
    results: pd.DataFrame,
    skipped_edges: pd.DataFrame,
    dropped_columns: pd.DataFrame,
    config: PipelineConfig,
) -> str:
    """Renders a Markdown report for edge weight estimation.

    Args:
        results: Edge coefficient estimates.
        skipped_edges: Skipped edge records with reasons.
        dropped_columns: Dropped preprocessing columns.
        config: Resolved pipeline configuration.

    Returns:
        Markdown report text.
    """
    return "\n".join(
        [
            "# Edge Weight Estimation Report",
            "",
            "## Interpretation",
            "",
            "This report estimates conditional linear coefficients for directed edges "
            "discovered in the causal discovery step.",
            "",
            "For each edge:",
            "",
            "`target ~ source + other_parents_of_target`",
            "",
            "The coefficient should not automatically be interpreted as an ATE, ATT, "
            "or total causal effect.",
            "",
            "It may be interpreted as a direct structural coefficient only under strong "
            "assumptions:",
            "",
            "- the discovered graph is correct",
            "- no unobserved confounding",
            "- no bad-control adjustment",
            "- linear additive structural form",
            "- no relevant measurement error",
            "- graph-selection uncertainty is ignored",
            "",
            "## Data Summary",
            "",
            f"- discovery_dir: `{config.data.discovery_dir}`",
            f"- output_dir: `{config.data.output_dir}`",
            f"- estimated_edges: `{len(results)}`",
            f"- skipped_edges: `{len(skipped_edges)}`",
            f"- dropped_columns: `{len(dropped_columns)}`",
            "",
            "## Results",
            "",
            dataframe_to_markdown(results),
            "",
            "## Skipped Edges",
            "",
            dataframe_to_markdown(skipped_edges),
            "",
            "## Dropped Columns",
            "",
            dataframe_to_markdown(dropped_columns),
            "",
            "## Encoding Notes",
            "",
            "- age and income are encoded as ordinal/midpoint numeric variables.",
            "- Sensitivity to alternative encoding, such as one-hot encoding, is not yet evaluated.",
            "- `homeowner` and `married` are retained for backward compatibility with existing discovery outputs; explicit `_yes` and `_unknown` columns are also available in the unstandardized frame.",
        ]
    )


def render_treatment_effect_report(
    results: pd.DataFrame,
    diagnostics: Mapping[str, pd.DataFrame],
    config: PipelineConfig,
    *,
    adjustment_set: list[str],
) -> str:
    """Renders a Markdown report for treatment effect estimation.

    Args:
        results: Treatment effect results.
        diagnostics: Diagnostic tables keyed by diagnostic name.
        config: Resolved pipeline configuration.
        adjustment_set: Selected covariates.

    Returns:
        Markdown report text.
    """
    te = config.treatment_effect
    graph_warning = (
        "- `graph_parents` was used. This is experimental and may adjust for "
        "mediators or bad controls depending on graph errors and temporal ordering."
        if te.adjustment_strategy == "graph_parents"
        else ""
    )
    lower, upper = te.propensity_clip
    return "\n".join(
        [
            "# Treatment Effect Estimation Report",
            "",
            "## Treatment Definition",
            "",
            f"- treatment: `{te.treatment}`",
            "",
            "## Outcome Definition",
            "",
            f"- outcome: `{te.outcome}`",
            "",
            "## Estimand",
            "",
            f"- estimand: `{te.estimand}`",
            "",
            "Under the specified adjustment assumptions, the estimated effect is reported below.",
            "",
            "## Adjustment Set",
            "",
            f"- strategy: `{te.adjustment_strategy}`",
            f"- covariates: `{', '.join(adjustment_set) if adjustment_set else '(none)'}`",
            graph_warning,
            "",
            "## Design Diagnostics",
            "",
            dataframe_to_markdown(diagnostics["design"]),
            "",
            "## Covariate Balance",
            "",
            dataframe_to_markdown(diagnostics["balance"]),
            "",
            "## Propensity Overlap",
            "",
            dataframe_to_markdown(diagnostics["propensity_overlap"]),
            "",
            "## Outcome Distribution",
            "",
            dataframe_to_markdown(diagnostics["outcome_distribution"]),
            "",
            "## Effect Estimates",
            "",
            dataframe_to_markdown(results),
            "",
            "## Robustness Notes",
            "",
            "- Difference in means is unadjusted and is a baseline association.",
            "- OLS uses the requested heteroskedasticity-consistent SE when specified.",
            f"- IPW and AIPW use a logistic propensity model fitted on the selected covariates and clip propensity scores to [{lower}, {upper}].",
            "- Confidence intervals and p-values use a standard normal approximation.",
            "- AIPW is implemented without sample splitting; nuisance-model overfit uncertainty is not separately handled.",
            "",
            "## Limitations",
            "",
            "- unobserved confounding is not removed by this script",
            "- causal discovery results may be wrong",
            "- graph selection uncertainty is not reflected in standard errors",
            "- adjusting for post-treatment variables changes the estimand away from a total effect",
            "- poor overlap can make IPW and AIPW unstable",
            "- age and income are encoded as ordinal/midpoint numeric variables; one-hot encoding sensitivity is not yet evaluated",
        ]
    )

