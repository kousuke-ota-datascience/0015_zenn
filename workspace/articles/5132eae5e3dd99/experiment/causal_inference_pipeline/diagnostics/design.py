"""Treatment assignment design diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .balance import compute_balance_table
from .outcome import summarize_outcome_distribution
from .overlap import summarize_propensity_overlap


def summarize_design(frame: pd.DataFrame, treatment: str) -> pd.DataFrame:
    """Summarizes treatment assignment counts.

    Args:
        frame: Analysis frame.
        treatment: Binary treatment column.

    Returns:
        One-row data frame containing total sample size, treated count, control
        count, and treated rate.

    Raises:
        ValueError: If the treatment column is missing.
    """
    if treatment not in frame.columns:
        raise ValueError(f"treatment column is missing: {treatment}")
    treatment_values = frame[treatment].astype(float)
    n = int(treatment_values.notna().sum())
    n_treated = int((treatment_values == 1.0).sum())
    n_control = int((treatment_values == 0.0).sum())
    treated_rate = n_treated / n if n > 0 else np.nan
    return pd.DataFrame(
        [
            {
                "n": n,
                "n_treated": n_treated,
                "n_control": n_control,
                "treated_rate": treated_rate,
            }
        ]
    )


class DesignDiagnostics:
    """Convenience wrapper for treatment-effect diagnostic tables.

    Args:
        frame: Analysis frame.
        treatment: Binary treatment column.
        outcome: Outcome column.
        covariates: Adjustment covariates.
    """

    def __init__(
        self,
        *,
        frame: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: Sequence[str],
    ) -> None:
        """Initialize diagnostics.

        Args:
            frame: Analysis frame.
            treatment: Binary treatment column.
            outcome: Outcome column.
            covariates: Adjustment covariates.
        """
        self.frame = frame
        self.treatment = treatment
        self.outcome = outcome
        self.covariates = list(covariates)

    def treatment_counts(self) -> pd.DataFrame:
        """Return treatment assignment counts.

        Returns:
            One-row treatment-count table.
        """
        return summarize_design(self.frame, self.treatment)

    def balance_table(self) -> pd.DataFrame:
        """Return unweighted covariate balance diagnostics.

        Returns:
            Balance table for configured covariates.
        """
        return compute_balance_table(self.frame, self.treatment, self.covariates)

    def outcome_distribution(self) -> pd.DataFrame:
        """Return outcome distribution diagnostics.

        Returns:
            One-row outcome distribution table.
        """
        return summarize_outcome_distribution(self.frame, self.outcome)

    def propensity_overlap(self, propensity_score: np.ndarray) -> pd.DataFrame:
        """Return propensity score overlap diagnostics.

        Args:
            propensity_score: Estimated propensity scores.

        Returns:
            One-row propensity overlap table.
        """
        return summarize_propensity_overlap(propensity_score, clip=(0.01, 0.99))

