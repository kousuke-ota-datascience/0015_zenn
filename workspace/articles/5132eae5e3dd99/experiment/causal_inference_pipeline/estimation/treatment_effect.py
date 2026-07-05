"""Explicit treatment effect estimators."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..constants import SUPPORTED_EFFECT_METHODS, SUPPORTED_ESTIMANDS, SUPPORTED_ROBUST_SE
from .inference import effective_sample_size, records_to_frame, weighted_mean, weighted_variance
from .linear_model import confidence_interval, fit_linear_regression, normal_p_value, numeric_matrix
from .propensity import fit_logistic_propensity


@dataclass(frozen=True)
class TreatmentEffectResult:
    """Treatment effect estimation result.

    Attributes:
        method: Estimation method name.
        estimand: Target estimand such as ``ATE`` or ``ATT``.
        treatment: Treatment column name.
        outcome: Outcome column name.
        adjustment_set: Covariates used for adjustment.
        n: Number of observations used for estimation.
        n_treated: Number of treated observations.
        n_control: Number of control observations.
        effect: Estimated treatment effect.
        std_error: Approximate standard error.
        ci_low: Lower confidence interval bound.
        ci_high: Upper confidence interval bound.
        p_value: Approximate p-value.
        notes: Method-specific notes and limitations.
    """

    method: str
    estimand: str
    treatment: str
    outcome: str
    adjustment_set: list[str]
    n: int
    n_treated: int
    n_control: int
    effect: float
    std_error: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    notes: str = ""


def validate_treatment_effect_inputs(
    frame: pd.DataFrame,
    treatment: str,
    outcome: str,
) -> None:
    """Validate treatment-effect input columns.

    Args:
        frame: Analysis frame.
        treatment: Treatment column name.
        outcome: Outcome column name.

    Raises:
        ValueError: If treatment/outcome columns are missing or the treatment
            is not binary.
    """
    if treatment not in frame.columns:
        raise ValueError(f"treatment column is missing from frame: {treatment}")
    if outcome not in frame.columns:
        raise ValueError(f"outcome column is missing from frame: {outcome}")
    values = set(pd.to_numeric(frame[treatment].dropna(), errors="coerce").unique())
    if values.difference({0.0, 1.0}):
        raise ValueError(
            "Treatment must be binary for current estimators. "
            f"Observed values: {sorted(values)}"
        )


class TreatmentEffectEstimator:
    """Estimates treatment effects under a specified adjustment strategy.

    The estimators identify ATE or ATT only under the usual adjustment
    assumptions: consistency, positivity/overlap, no unobserved confounding
    conditional on the selected covariates, and correct enough nuisance models
    for model-based estimators. These assumptions are not verified by code.

    Args:
        frame: Household-level analysis frame.
        treatment: Binary treatment column.
        outcome: Outcome column.
        covariates: Adjustment covariates.
        estimand: Target estimand. Supported values are ``ATE`` and ``ATT``.
        robust_se: Robust standard error type for OLS-based estimators.
        propensity_clip: Lower and upper bounds used to clip propensity scores.
        cross_fitting_folds: Number of folds for cross-fitting. ``0``
            preserves the legacy no-sample-splitting behavior.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: list[str],
        estimand: str = "ATE",
        robust_se: str = "HC3",
        propensity_clip: tuple[float, float] = (0.01, 0.99),
        cross_fitting_folds: int = 0,
    ) -> None:
        """Initialize the estimator.

        Args:
            frame: Household-level analysis frame.
            treatment: Binary treatment column.
            outcome: Outcome column.
            covariates: Adjustment covariates.
            estimand: Target estimand.
            robust_se: Robust standard error type.
            propensity_clip: Lower and upper propensity clipping bounds.
            cross_fitting_folds: Number of folds for cross-fitting.

        Raises:
            ValueError: If estimand, robust SE, clip bounds, or inputs are
                invalid.
        """
        if estimand not in SUPPORTED_ESTIMANDS:
            raise ValueError(f"estimand must be ATE or ATT: {estimand}")
        if robust_se not in SUPPORTED_ROBUST_SE:
            raise ValueError(f"robust_se must be one of {SUPPORTED_ROBUST_SE}: {robust_se}")
        if not 0 < propensity_clip[0] < propensity_clip[1] < 1:
            raise ValueError("propensity_clip must satisfy 0 < lower < upper < 1")
        validate_treatment_effect_inputs(frame, treatment, outcome)
        self.frame = frame
        self.treatment = treatment
        self.outcome = outcome
        self.covariates = covariates
        self.estimand = estimand
        self.robust_se = robust_se
        self.propensity_clip = propensity_clip
        self.cross_fitting_folds = cross_fitting_folds
        self.last_propensity_score: np.ndarray | None = None
        self.last_propensity_notes = ""

    def complete_case_data(self, include_covariates: bool) -> pd.DataFrame:
        """Return complete-case data for a requested estimator.

        Args:
            include_covariates: Whether covariates must be complete.

        Returns:
            Complete-case data frame.
        """
        columns = [self.treatment, self.outcome]
        if include_covariates:
            columns.extend(self.covariates)
        return self.frame.loc[:, columns].dropna()

    def result(
        self,
        *,
        method: str,
        data: pd.DataFrame,
        effect: float,
        std_error: float | None,
        notes: str,
    ) -> TreatmentEffectResult:
        """Create a standardized treatment-effect result object.

        Args:
            method: Estimation method name.
            data: Data used by the estimator.
            effect: Point estimate.
            std_error: Approximate standard error.
            notes: Method-specific notes.

        Returns:
            Treatment-effect result with confidence interval and p-value.
        """
        t = data[self.treatment].astype(float)
        ci_low, ci_high = confidence_interval(effect, std_error)
        p_value = (
            normal_p_value(effect / std_error)
            if std_error is not None and std_error > 0
            else None
        )
        return TreatmentEffectResult(
            method=method,
            estimand=self.estimand,
            treatment=self.treatment,
            outcome=self.outcome,
            adjustment_set=list(self.covariates),
            n=int(len(data)),
            n_treated=int((t == 1.0).sum()),
            n_control=int((t == 0.0).sum()),
            effect=float(effect),
            std_error=float(std_error) if std_error is not None else None,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            notes=notes,
        )

    def estimate(self, methods: list[str]) -> pd.DataFrame:
        """Estimate treatment effects with requested methods.

        Args:
            methods: Method names to run.

        Returns:
            Treatment-effect result table.

        Raises:
            ValueError: If an unknown method is requested.
        """
        records = []
        runners = {
            "diff_in_means": self.diff_in_means,
            "ols": self.ols,
            "ipw": self.ipw,
            "aipw": self.aipw,
        }
        for method in methods:
            if method not in SUPPORTED_EFFECT_METHODS:
                raise ValueError(f"unknown effect method: {method}")
            records.append(runners[method]().__dict__)
        columns = list(TreatmentEffectResult.__dataclass_fields__.keys())
        return records_to_frame(records, columns)

    def diff_in_means(self) -> TreatmentEffectResult:
        """Estimate the unadjusted treated-control mean difference.

        Returns:
            Treatment-effect result. The numerical contrast is identical for
            ATE and ATT, but it is not adjusted for confounding.
        """
        data = self.complete_case_data(include_covariates=False)
        y = data[self.outcome].to_numpy(dtype=float)
        t = data[self.treatment].to_numpy(dtype=float)
        y_treated = y[t == 1.0]
        y_control = y[t == 0.0]
        if len(y_treated) == 0 or len(y_control) == 0:
            raise ValueError("diff_in_means requires both treated and control observations")
        effect = float(y_treated.mean() - y_control.mean())
        variance_treated = float(np.var(y_treated, ddof=1)) if len(y_treated) > 1 else 0.0
        variance_control = float(np.var(y_control, ddof=1)) if len(y_control) > 1 else 0.0
        standard_error = math.sqrt(
            variance_treated / len(y_treated) + variance_control / len(y_control)
        )
        return self.result(
            method="diff_in_means",
            data=data,
            effect=effect,
            std_error=standard_error,
            notes="unadjusted_difference; numerical contrast is identical for ATE and ATT",
        )

    def ols(self) -> TreatmentEffectResult:
        """Estimate a regression-adjusted treatment coefficient.

        Returns:
            Treatment-effect result using the treatment coefficient from
            ``outcome ~ treatment + covariates``.
        """
        data = self.complete_case_data(include_covariates=True)
        y = data[self.outcome].to_numpy(dtype=float)
        regressors = [self.treatment, *self.covariates]
        x = numeric_matrix(data, regressors)
        fit = fit_linear_regression(y, x, robust_se=self.robust_se)
        effect = float(fit.coefficients[1])
        standard_error = float(fit.standard_errors[1])
        notes = [
            f"robust_se={self.robust_se}",
            f"rank={fit.rank}/{x.shape[1] + 1}",
            f"condition_number={fit.condition_number:.6g}",
            "normal_approximation_for_ci_and_p_value",
        ]
        if fit.rank < x.shape[1] + 1:
            notes.append("rank_deficient_design_pinv_used")
        return self.result(
            method="ols",
            data=data,
            effect=effect,
            std_error=standard_error,
            notes="; ".join(notes),
        )

    def propensity_data(self) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """Fit propensity scores and return complete-case arrays.

        Returns:
            Tuple of data frame, outcome array, treatment array, and clipped
            propensity scores.
        """
        data = self.complete_case_data(include_covariates=True)
        y = data[self.outcome].to_numpy(dtype=float)
        t = data[self.treatment].to_numpy(dtype=float)
        x = numeric_matrix(data, self.covariates)
        propensity_raw, notes = fit_logistic_propensity(t, x)
        propensity_score = np.clip(propensity_raw, *self.propensity_clip)
        self.last_propensity_score = propensity_raw
        self.last_propensity_notes = notes
        return data, y, t, propensity_score

    def ipw(self) -> TreatmentEffectResult:
        """Estimate an inverse-probability-weighted effect.

        Returns:
            Treatment-effect result using a logistic propensity model and
            configured propensity clipping.
        """
        data, y, t, propensity_score = self.propensity_data()
        if self.estimand == "ATE":
            weights_treated = t / propensity_score
            weights_control = (1.0 - t) / (1.0 - propensity_score)
        else:
            weights_treated = t
            weights_control = (1.0 - t) * propensity_score / (1.0 - propensity_score)

        mean_treated = weighted_mean(y, weights_treated)
        mean_control = weighted_mean(y, weights_control)
        effect = float(mean_treated - mean_control)
        ess_treated = effective_sample_size(weights_treated)
        ess_control = effective_sample_size(weights_control)
        variance_treated = weighted_variance(y, weights_treated, mean_treated)
        variance_control = weighted_variance(y, weights_control, mean_control)
        standard_error = math.sqrt(
            variance_treated / ess_treated + variance_control / ess_control
        )
        raw = self.last_propensity_score
        lower, upper = self.propensity_clip
        overlap_warning = ""
        if raw is not None and ((raw < lower).any() or (raw > upper).any()):
            overlap_warning = f"; overlap_warning=propensity_outside_[{lower},{upper}]"
        notes = (
            f"propensity_logit; clip=[{lower},{upper}]; "
            f"ess_treated={ess_treated:.6g}; ess_control={ess_control:.6g}"
            f"{overlap_warning}"
        )
        if self.last_propensity_notes:
            notes = f"{notes}; {self.last_propensity_notes}"
        return self.result(
            method="ipw",
            data=data,
            effect=effect,
            std_error=standard_error,
            notes=notes,
        )

    def aipw(self) -> TreatmentEffectResult:
        """Estimate an augmented inverse-probability-weighted effect.

        Returns:
            Treatment-effect result. Cross-fitting is not yet implemented; a
            nonzero ``cross_fitting_folds`` value is recorded in notes only.
        """
        data, y, t, propensity_score = self.propensity_data()
        x = numeric_matrix(data, self.covariates)
        treated_mask = t == 1.0
        control_mask = t == 0.0
        if treated_mask.sum() <= x.shape[1] + 1 or control_mask.sum() <= x.shape[1] + 1:
            return self.result(
                method="aipw",
                data=data,
                effect=np.nan,
                std_error=None,
                notes="skipped_insufficient_sample_size_for_outcome_models",
            )

        treated_fit = fit_linear_regression(y[treated_mask], x[treated_mask])
        control_fit = fit_linear_regression(y[control_mask], x[control_mask])
        design = np.column_stack([np.ones(len(data)), x])
        mu_treated = design @ treated_fit.coefficients
        mu_control = design @ control_fit.coefficients

        if self.estimand == "ATE":
            score = (
                mu_treated
                - mu_control
                + t / propensity_score * (y - mu_treated)
                - (1.0 - t) / (1.0 - propensity_score) * (y - mu_control)
            )
        else:
            treated_rate = float(t.mean())
            score = (
                t / treated_rate * (y - mu_control)
                - (1.0 - t)
                * propensity_score
                / (1.0 - propensity_score)
                / treated_rate
                * (y - mu_control)
            )
        effect = float(score.mean())
        standard_error = float(np.std(score, ddof=1) / math.sqrt(len(score)))
        lower, upper = self.propensity_clip
        notes = (
            f"aipw_linear_outcome_models; propensity_logit; clip=[{lower},{upper}]; "
            "no_sample_splitting"
        )
        if self.cross_fitting_folds:
            notes = f"{notes}; cross_fitting_requested_but_not_implemented={self.cross_fitting_folds}"
        if self.last_propensity_notes:
            notes = f"{notes}; {self.last_propensity_notes}"
        return self.result(
            method="aipw",
            data=data,
            effect=effect,
            std_error=standard_error,
            notes=notes,
        )

