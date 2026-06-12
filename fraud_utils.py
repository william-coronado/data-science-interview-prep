"""Reusable utilities for the **Northwind Pay** fraud / credit-risk case study.

This module is written as production-style code rather than notebook scratch:

* every public function and class carries type hints and a docstring,
* inputs are validated up front, and
* failure modes raise explicit, descriptive exceptions instead of returning
  silently-wrong results.

It is imported by the case-study notebook *and* exercised by
``test_fraud_utils.py``, so the exact logic the narrative demonstrates is the
logic under test. Keeping it in one importable module is what lets ``pytest``
run from the command line the way a reviewer would expect in a real repo.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Constants describing the synthetic domain
# --------------------------------------------------------------------------- #
MERCHANT_CATEGORIES: list[str] = [
    "groceries", "fuel", "restaurant", "retail",
    "electronics", "travel", "gambling", "crypto",
]
# Log-odds contribution of each merchant category to fraud risk. Everyday
# spend is protective; gambling / crypto are risk-heavy. These drive the
# synthetic label so downstream models have genuine signal to recover.
_CATEGORY_RISK: dict[str, float] = {
    "groceries": -1.2, "fuel": -0.8, "restaurant": -0.6, "retail": -0.2,
    "electronics": 0.4, "travel": 0.6, "gambling": 1.3, "crypto": 1.6,
}
DEVICE_TYPES: list[str] = ["pos", "web", "mobile"]
REGIONS: list[str] = ["North", "South", "East", "West"]
SIGNUP_CHANNELS: list[str] = ["branch", "web", "partner_app"]
GENDERS: list[str] = ["female", "male"]


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured stdout logger that is safe to call repeatedly.

    Notebooks re-run cells, so naively adding a handler each call produces
    duplicated log lines. This attaches a handler only once per logger name.

    Args:
        name: Logger name (typically the module or pipeline name).
        level: Logging level, e.g. ``logging.INFO``.

    Returns:
        A configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# Synthetic data generation
# --------------------------------------------------------------------------- #
def _solve_intercept(linear_term: np.ndarray, target_rate: float) -> float:
    """Find the intercept that makes a logistic model hit ``target_rate``.

    Bisection on the mean predicted probability. Keeps fraud prevalence stable
    regardless of how the other coefficients shift the linear predictor.

    Args:
        linear_term: The linear predictor excluding the intercept.
        target_rate: Desired mean event probability, in (0, 1).

    Returns:
        The intercept (float) such that ``sigmoid(intercept + linear_term)``
        has mean approximately ``target_rate``.
    """
    if not 0.0 < target_rate < 1.0:
        raise ValueError("target_rate must lie in (0, 1).")
    lo, hi = -15.0, 5.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        mean_p = float((1.0 / (1.0 + np.exp(-(mid + linear_term)))).mean())
        if mean_p > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def generate_transactions(
    n_rows: int = 40_000,
    seed: int = 42,
    fraud_base_rate: float = 0.015,
) -> pd.DataFrame:
    """Generate a synthetic card-transaction dataset for Northwind Pay.

    The fraud label is drawn from a logistic model of the features, so the
    relationships are real but noisy. A deliberately *leaky* column
    (``chargeback_reported``) and *non-random* missingness in
    ``account_tenure_days`` are injected on purpose for later teaching points.

    Args:
        n_rows: Number of transactions to generate (must be positive).
        seed: Seed for the NumPy random generator (reproducibility).
        fraud_base_rate: Target overall fraud prevalence, in (0, 1).

    Returns:
        A DataFrame sorted by ``timestamp`` with one row per transaction.

    Raises:
        ValueError: If ``n_rows`` is not positive or ``fraud_base_rate`` is
            not in (0, 1).
    """
    if n_rows <= 0:
        raise ValueError("n_rows must be a positive integer.")
    if not 0.0 < fraud_base_rate < 1.0:
        raise ValueError("fraud_base_rate must lie in (0, 1).")

    rng = np.random.default_rng(seed)

    customer_id = rng.integers(10_000, 10_000 + max(2, n_rows // 8), size=n_rows)

    start = np.datetime64("2025-01-01T00:00:00")
    days = rng.integers(0, 180, size=n_rows)
    hour = rng.integers(0, 24, size=n_rows)
    minute = rng.integers(0, 60, size=n_rows)
    timestamp = (
        start
        + days.astype("timedelta64[D]")
        + hour.astype("timedelta64[h]")
        + minute.astype("timedelta64[m]")
    )

    amount = np.round(rng.lognormal(mean=3.2, sigma=1.1, size=n_rows), 2)
    num_txns_24h = rng.poisson(lam=4.0, size=n_rows)
    merchant_category = rng.choice(
        MERCHANT_CATEGORIES, size=n_rows,
        p=[0.22, 0.12, 0.16, 0.18, 0.10, 0.09, 0.07, 0.06],
    )
    device_type = rng.choice(DEVICE_TYPES, size=n_rows, p=[0.40, 0.30, 0.30])
    is_foreign = rng.binomial(1, 0.12, size=n_rows)
    account_tenure_days = np.round(
        rng.gamma(shape=2.0, scale=400.0, size=n_rows)
    ).astype(float)
    customer_age = np.clip(rng.normal(42, 14, size=n_rows), 18, 90).round().astype(int)
    customer_region = rng.choice(REGIONS, size=n_rows, p=[0.30, 0.25, 0.25, 0.20])
    customer_gender = rng.choice(GENDERS, size=n_rows, p=[0.49, 0.51])
    signup_channel = rng.choice(SIGNUP_CHANNELS, size=n_rows, p=[0.30, 0.50, 0.20])

    # --- Build the fraud signal (log-odds) ------------------------------- #
    cat_risk = np.array([_CATEGORY_RISK[c] for c in merchant_category])
    log_amount = np.log1p(amount)
    linear_term = (
        0.85 * cat_risk
        + 1.10 * is_foreign
        + 0.55 * (device_type == "web").astype(float)
        + 0.40 * (log_amount - log_amount.mean())
        + 0.30 * (num_txns_24h - num_txns_24h.mean()) / max(num_txns_24h.std(), 1e-9)
        - 0.45 * (account_tenure_days - account_tenure_days.mean())
        / max(account_tenure_days.std(), 1e-9)
        + 0.35 * ((hour >= 1) & (hour <= 5)).astype(float)
        + 0.18 * (customer_gender == "male").astype(float)  # subtle proxy disparity
    )
    intercept = _solve_intercept(linear_term, fraud_base_rate)
    fraud_proba = 1.0 / (1.0 + np.exp(-(intercept + linear_term)))
    is_fraud = rng.binomial(1, fraud_proba).astype(int)

    # --- Deliberately leaky feature: only known *after* the fact --------- #
    chargeback_reported = np.where(
        is_fraud == 1,
        rng.binomial(1, 0.92, size=n_rows),
        rng.binomial(1, 0.01, size=n_rows),
    ).astype(int)

    # --- Non-random missingness: partner_app signups under-report tenure - #
    miss_prob = np.where(signup_channel == "partner_app", 0.25, 0.04)
    account_tenure_days[rng.random(n_rows) < miss_prob] = np.nan

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "timestamp": timestamp,
            "amount": amount,
            "num_txns_24h": num_txns_24h,
            "merchant_category": merchant_category,
            "device_type": device_type,
            "is_foreign": is_foreign,
            "account_tenure_days": account_tenure_days,
            "customer_age": customer_age,
            "customer_region": customer_region,
            "customer_gender": customer_gender,
            "signup_channel": signup_channel,
            "chargeback_reported": chargeback_reported,
            "is_fraud": is_fraud,
        }
    )
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    df.insert(0, "transaction_id", np.arange(1, len(df) + 1))
    return df


def generate_ab_experiment(n_per_group: int = 6_000, seed: int = 7) -> pd.DataFrame:
    """Generate a step-up-authentication A/B experiment dataset.

    Treatment customers see an extra authentication step. It is designed to cut
    fraud but adds checkout friction, so we measure *both* a fraud outcome and a
    conversion outcome plus a skewed continuous friction metric.

    Args:
        n_per_group: Number of customers in each of control / treatment.
        seed: Seed for the NumPy random generator.

    Returns:
        A long DataFrame with one row per customer.

    Raises:
        ValueError: If ``n_per_group`` is not positive.
    """
    if n_per_group <= 0:
        raise ValueError("n_per_group must be a positive integer.")
    rng = np.random.default_rng(seed)
    n = n_per_group

    group = np.array(["control"] * n + ["treatment"] * n)
    converted = rng.binomial(1, np.where(group == "control", 0.780, 0.752))
    is_fraud = rng.binomial(1, np.where(group == "control", 0.0210, 0.0130))
    base_seconds = rng.lognormal(mean=2.6, sigma=0.5, size=2 * n)
    checkout_seconds = np.round(
        np.where(group == "treatment", base_seconds * 1.15, base_seconds), 2
    )

    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, 2 * n + 1),
            "group": group,
            "converted": converted,
            "is_fraud": is_fraud,
            "checkout_seconds": checkout_seconds,
        }
    )
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    *,
    allow_empty: bool = False,
) -> None:
    """Validate a DataFrame's type, emptiness and required columns.

    Args:
        df: Object expected to be a non-empty DataFrame.
        required_columns: Columns that must be present.
        allow_empty: If ``False`` (default), an empty DataFrame is rejected.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
        ValueError: If ``df`` is empty and ``allow_empty`` is ``False``.
        KeyError: If any required column is missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if not allow_empty and df.empty:
        raise ValueError("DataFrame is empty; expected at least one row.")
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


# --------------------------------------------------------------------------- #
# Datetime feature engineering
# --------------------------------------------------------------------------- #
def add_datetime_features(
    df: pd.DataFrame, timestamp_col: str = "timestamp"
) -> pd.DataFrame:
    """Return a copy of ``df`` enriched with datetime-derived features.

    Adds hour, day-of-week, weekend / night flags and a cyclical (sin/cos)
    encoding of the hour so that 23:00 and 00:00 are treated as adjacent.

    Args:
        df: Input DataFrame containing a timestamp column.
        timestamp_col: Name of the timestamp column.

    Returns:
        A new DataFrame with added ``txn_*`` columns.

    Raises:
        KeyError: If ``timestamp_col`` is absent.
    """
    validate_dataframe(df, [timestamp_col])
    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col])
    out["txn_hour"] = ts.dt.hour
    out["txn_dayofweek"] = ts.dt.dayofweek
    out["txn_is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    out["txn_is_night"] = ts.dt.hour.isin(range(0, 6)).astype(int)
    out["txn_hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24.0)
    out["txn_hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24.0)
    return out


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    *,
    numeric_impute: str = "median",
    scale: bool = True,
) -> ColumnTransformer:
    """Compose a scikit-learn ``ColumnTransformer`` for mixed-type data.

    Numeric columns are imputed (and optionally scaled); categoricals are
    imputed with the mode and one-hot encoded with unseen categories ignored at
    inference time.

    Args:
        numeric_features: Numeric column names.
        categorical_features: Categorical column names.
        numeric_impute: Strategy for :class:`SimpleImputer` on numerics.
        scale: Whether to standard-scale numerics (skip for tree models).

    Returns:
        An unfitted :class:`ColumnTransformer`.
    """
    numeric_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy=numeric_impute))
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipe, list(numeric_features)),
            ("cat", categorical_pipe, list(categorical_features)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_classifier(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute the headline metrics for a binary classifier.

    Args:
        y_true: Ground-truth labels (0/1).
        y_proba: Predicted probability of the positive class.
        threshold: Decision threshold for the hard-label metrics.

    Returns:
        Dict with ``roc_auc``, ``pr_auc``, ``precision``, ``recall``, ``f1``
        and the ``threshold`` used.

    Raises:
        ValueError: If inputs differ in length or are empty.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError("y_true and y_proba must have the same length.")
    if y_true.size == 0:
        raise ValueError("Cannot evaluate on empty arrays.")
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }


# --------------------------------------------------------------------------- #
# Frequentist test helpers
# --------------------------------------------------------------------------- #
def two_proportion_ztest(
    success_a: int, n_a: int, success_b: int, n_b: int
) -> tuple[float, float]:
    """Two-sided two-proportion z-test (pooled variance).

    Args:
        success_a: Successes in group A.
        n_a: Sample size of group A.
        success_b: Successes in group B.
        n_b: Sample size of group B.

    Returns:
        ``(z_statistic, two_sided_p_value)``.

    Raises:
        ValueError: If a sample size is non-positive or the pooled standard
            error is zero.
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError("Sample sizes must be positive.")
    if not (0 <= success_a <= n_a):
        raise ValueError("success_a must lie between 0 and n_a.")
    if not (0 <= success_b <= n_b):
        raise ValueError("success_b must lie between 0 and n_b.")
    p_a, p_b = success_a / n_a, success_b / n_b
    p_pool = (success_a + success_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_a + 1.0 / n_b))
    if se == 0:
        raise ValueError("Pooled standard error is zero; test is undefined.")
    z = (p_a - p_b) / se
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(z), float(p_value)


def required_sample_size_two_proportions(
    p_control: float,
    mde: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> int:
    """Sample size **per group** to detect a change in a proportion.

    Args:
        p_control: Baseline (control) proportion, in (0, 1).
        mde: Minimum detectable effect (absolute change in proportion).
        alpha: Significance level.
        power: Desired statistical power (1 - beta).
        two_sided: Whether the test is two-sided.

    Returns:
        Required sample size per group (rounded up).

    Raises:
        ValueError: If ``p_control`` or ``p_control + mde`` is not in (0, 1),
            or ``mde`` is zero.
    """
    if not 0.0 < p_control < 1.0:
        raise ValueError("p_control must lie in (0, 1).")
    if mde == 0:
        raise ValueError("mde must be non-zero.")
    p_treat = p_control + mde
    if not 0.0 < p_treat < 1.0:
        raise ValueError("p_control + mde must lie in (0, 1).")

    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0) if two_sided else stats.norm.ppf(1.0 - alpha)
    z_beta = stats.norm.ppf(power)
    p_bar = (p_control + p_treat) / 2.0
    numerator = (
        z_alpha * np.sqrt(2.0 * p_bar * (1.0 - p_bar))
        + z_beta * np.sqrt(p_control * (1.0 - p_control) + p_treat * (1.0 - p_treat))
    ) ** 2
    return int(np.ceil(numerator / (mde ** 2)))


# --------------------------------------------------------------------------- #
# Fairness metrics
# --------------------------------------------------------------------------- #
def _positive_rate(y_pred: np.ndarray, mask: np.ndarray) -> float:
    """Positive-prediction rate within a boolean ``mask`` (NaN if empty)."""
    subset = y_pred[mask]
    return float(subset.mean()) if subset.size else float("nan")


def _validate_fairness_inputs(
    y_pred: np.ndarray, sensitive: np.ndarray, privileged: str, unprivileged: str
) -> tuple[np.ndarray, np.ndarray]:
    """Validate fairness helper inputs and return 1D arrays."""
    if privileged == unprivileged:
        raise ValueError("privileged and unprivileged groups must differ.")
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive)
    if y_pred.ndim != 1 or sensitive.ndim != 1:
        raise ValueError("y_pred and sensitive must be 1D arrays.")
    if y_pred.shape[0] != sensitive.shape[0]:
        raise ValueError("y_pred and sensitive must have the same length.")
    if y_pred.size == 0:
        raise ValueError("y_pred and sensitive cannot be empty.")
    if not np.any(sensitive == privileged):
        raise ValueError("No rows found for privileged group.")
    if not np.any(sensitive == unprivileged):
        raise ValueError("No rows found for unprivileged group.")
    return y_pred, sensitive


def demographic_parity_difference(
    y_pred: np.ndarray, sensitive: np.ndarray, privileged: str, unprivileged: str
) -> float:
    """Difference in positive-prediction rate (unprivileged - privileged).

    A value near zero indicates demographic parity. For a fraud flag, a large
    positive value means the unprivileged group is flagged disproportionately.
    """
    y_pred, sensitive = _validate_fairness_inputs(
        y_pred, sensitive, privileged, unprivileged
    )
    return _positive_rate(y_pred, sensitive == unprivileged) - _positive_rate(
        y_pred, sensitive == privileged
    )


def disparate_impact_ratio(
    y_pred: np.ndarray, sensitive: np.ndarray, privileged: str, unprivileged: str
) -> float:
    """Ratio of positive rates (unprivileged / privileged).

    The "four-fifths rule" treats values below 0.8 (or above 1.25) as evidence
    of adverse impact.

    Raises:
        ZeroDivisionError: If the privileged-group positive rate is zero.
    """
    y_pred, sensitive = _validate_fairness_inputs(
        y_pred, sensitive, privileged, unprivileged
    )
    rate_priv = _positive_rate(y_pred, sensitive == privileged)
    rate_unpriv = _positive_rate(y_pred, sensitive == unprivileged)
    if rate_priv == 0:
        raise ZeroDivisionError("Privileged-group positive rate is zero; ratio undefined.")
    return rate_unpriv / rate_priv


def equalized_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    privileged: str,
    unprivileged: str,
) -> dict[str, float]:
    """Group differences in true-positive and false-positive rates.

    Equalised odds asks for similar TPR *and* FPR across groups. Returns the
    unprivileged-minus-privileged gap for each.
    """
    y_true = np.asarray(y_true)
    y_pred, sensitive = _validate_fairness_inputs(
        y_pred, sensitive, privileged, unprivileged
    )
    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1D array.")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    def _rates(mask: np.ndarray) -> tuple[float, float]:
        yt, yp = y_true[mask], y_pred[mask]
        tpr = float(yp[yt == 1].mean()) if (yt == 1).any() else float("nan")
        fpr = float(yp[yt == 0].mean()) if (yt == 0).any() else float("nan")
        return tpr, fpr

    tpr_u, fpr_u = _rates(sensitive == unprivileged)
    tpr_p, fpr_p = _rates(sensitive == privileged)
    return {"tpr_difference": tpr_u - tpr_p, "fpr_difference": fpr_u - fpr_p}


# --------------------------------------------------------------------------- #
# End-to-end pipeline class
# --------------------------------------------------------------------------- #
class FraudModelPipeline(ClassifierMixin, BaseEstimator):
    """Validation -> preprocessing -> estimator, behind one fit/predict API.

    Wrapping the scikit-learn pipeline in a small class lets us enforce input
    validation, attach logging, and carry a tunable decision ``threshold``
    (crucial for imbalanced fraud problems) without leaking preprocessing
    details to callers.

    Args:
        estimator: A scikit-learn-compatible classifier exposing
            ``predict_proba``.
        numeric_features: Numeric column names to use.
        categorical_features: Categorical column names to use.
        threshold: Decision threshold applied in :meth:`predict`.
        scale: Whether to scale numerics (turn off for tree models).
        logger: Optional logger for lifecycle messages.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        numeric_features: Sequence[str],
        categorical_features: Sequence[str],
        threshold: float = 0.5,
        scale: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        self.estimator = estimator
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.threshold = threshold
        self.scale = scale
        self.logger = logger

    @property
    def required_columns(self) -> list[str]:
        """Columns the pipeline expects in any input frame."""
        return list(self.numeric_features) + list(self.categorical_features)

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "FraudModelPipeline":
        """Validate inputs, build the pipeline and fit it."""
        validate_dataframe(X, self.required_columns)
        self.pipeline_ = Pipeline(
            [
                (
                    "preprocess",
                    build_preprocessor(
                        self.numeric_features,
                        self.categorical_features,
                        scale=self.scale,
                    ),
                ),
                ("model", clone(self.estimator)),
            ]
        )
        self._log(
            f"Fitting {type(self.estimator).__name__} on {len(X):,} rows "
            f"and {len(self.required_columns)} features."
        )
        self.pipeline_.fit(X[self.required_columns], y)
        self.classes_ = np.unique(y)
        self._log("Fit complete.")
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "pipeline_"):
            raise RuntimeError("Pipeline is not fitted yet; call fit() first.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities for ``X``."""
        self._check_fitted()
        validate_dataframe(X, self.required_columns)
        return self.pipeline_.predict_proba(X[self.required_columns])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return hard 0/1 predictions using the configured ``threshold``."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)
