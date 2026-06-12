"""Unit tests for :mod:`fraud_utils`.

Run from the command line exactly as a reviewer would::

    pytest -q test_fraud_utils.py

These are proper ``test_`` functions (not notebook asserts): they cover the
happy path, the edge cases, and the explicit error contracts of the reusable
data-transformation and metric functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import fraud_utils as fu


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def transactions() -> pd.DataFrame:
    return fu.generate_transactions(n_rows=5_000, seed=123)


# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
def test_generate_transactions_shape_and_columns(transactions: pd.DataFrame) -> None:
    assert len(transactions) == 5_000
    expected = {
        "transaction_id", "customer_id", "timestamp", "amount", "is_fraud",
        "merchant_category", "customer_gender", "chargeback_reported",
    }
    assert expected.issubset(transactions.columns)


def test_generate_transactions_is_reproducible() -> None:
    a = fu.generate_transactions(n_rows=1_000, seed=99)
    b = fu.generate_transactions(n_rows=1_000, seed=99)
    pd.testing.assert_frame_equal(a, b)


def test_generate_transactions_prevalence_near_target() -> None:
    df = fu.generate_transactions(n_rows=20_000, seed=1, fraud_base_rate=0.02)
    assert 0.012 <= df["is_fraud"].mean() <= 0.030


def test_generate_transactions_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        fu.generate_transactions(n_rows=0)
    with pytest.raises(ValueError):
        fu.generate_transactions(fraud_base_rate=1.5)


def test_generate_transactions_has_injected_missingness(transactions: pd.DataFrame) -> None:
    assert transactions["account_tenure_days"].isna().any()


# --------------------------------------------------------------------------- #
# validate_dataframe
# --------------------------------------------------------------------------- #
def test_validate_dataframe_accepts_valid_frame() -> None:
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert fu.validate_dataframe(df, ["a", "b"]) is None


def test_validate_dataframe_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError):
        fu.validate_dataframe([1, 2, 3], ["a"])  # type: ignore[arg-type]


def test_validate_dataframe_rejects_empty() -> None:
    with pytest.raises(ValueError):
        fu.validate_dataframe(pd.DataFrame({"a": []}), ["a"])


def test_validate_dataframe_allows_empty_when_flagged() -> None:
    assert fu.validate_dataframe(pd.DataFrame({"a": []}), ["a"], allow_empty=True) is None


def test_validate_dataframe_reports_missing_columns() -> None:
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(KeyError):
        fu.validate_dataframe(df, ["a", "b"])


# --------------------------------------------------------------------------- #
# add_datetime_features
# --------------------------------------------------------------------------- #
def test_add_datetime_features_adds_expected_columns() -> None:
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-04 23:30", "2025-01-06 02:00"])})
    out = fu.add_datetime_features(df)
    for col in ["txn_hour", "txn_dayofweek", "txn_is_weekend", "txn_is_night",
                "txn_hour_sin", "txn_hour_cos"]:
        assert col in out.columns
    # 2025-01-04 is a Saturday -> weekend; 02:00 -> night
    assert out.loc[0, "txn_is_weekend"] == 1
    assert out.loc[1, "txn_is_night"] == 1


def test_add_datetime_features_does_not_mutate_input() -> None:
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-01 10:00"])})
    _ = fu.add_datetime_features(df)
    assert list(df.columns) == ["timestamp"]


def test_add_datetime_features_missing_column_raises() -> None:
    with pytest.raises(KeyError):
        fu.add_datetime_features(pd.DataFrame({"x": [1]}))


# --------------------------------------------------------------------------- #
# evaluate_classifier
# --------------------------------------------------------------------------- #
def test_evaluate_classifier_perfect_separation() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.01, 0.10, 0.90, 0.99])
    metrics = fu.evaluate_classifier(y_true, y_proba, threshold=0.5)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)


def test_evaluate_classifier_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        fu.evaluate_classifier(np.array([0, 1]), np.array([0.5]))


def test_evaluate_classifier_empty_raises() -> None:
    with pytest.raises(ValueError):
        fu.evaluate_classifier(np.array([]), np.array([]))


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #
def test_two_proportion_ztest_detects_clear_difference() -> None:
    z, p = fu.two_proportion_ztest(success_a=210, n_a=1000, success_b=130, n_b=1000)
    assert z > 0
    assert p < 0.001


def test_two_proportion_ztest_no_difference() -> None:
    z, p = fu.two_proportion_ztest(success_a=100, n_a=1000, success_b=100, n_b=1000)
    assert z == pytest.approx(0.0)
    assert p == pytest.approx(1.0)


def test_two_proportion_ztest_bad_sample_size() -> None:
    with pytest.raises(ValueError):
        fu.two_proportion_ztest(1, 0, 1, 10)


def test_two_proportion_ztest_invalid_success_counts() -> None:
    with pytest.raises(ValueError):
        fu.two_proportion_ztest(success_a=11, n_a=10, success_b=1, n_b=10)
    with pytest.raises(ValueError):
        fu.two_proportion_ztest(success_a=-1, n_a=10, success_b=1, n_b=10)
    with pytest.raises(ValueError):
        fu.two_proportion_ztest(success_a=1, n_a=10, success_b=11, n_b=10)
    with pytest.raises(ValueError):
        fu.two_proportion_ztest(success_a=1, n_a=10, success_b=-1, n_b=10)


def test_two_proportion_ztest_accepts_inclusive_bounds() -> None:
    # success_a/success_b of 0 or n are valid; pair with a non-degenerate
    # success_b/success_a so p_pool isn't 0 or 1 (which raises separately
    # for a zero standard error).
    fu.two_proportion_ztest(success_a=0, n_a=10, success_b=5, n_b=10)
    fu.two_proportion_ztest(success_a=10, n_a=10, success_b=5, n_b=10)


def test_required_sample_size_is_positive_and_monotonic() -> None:
    big_effect = fu.required_sample_size_two_proportions(0.20, mde=0.05)
    small_effect = fu.required_sample_size_two_proportions(0.20, mde=0.01)
    assert big_effect > 0
    # Detecting a smaller effect needs a larger sample.
    assert small_effect > big_effect


def test_required_sample_size_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        fu.required_sample_size_two_proportions(0.0, mde=0.05)


# --------------------------------------------------------------------------- #
# Fairness metrics
# --------------------------------------------------------------------------- #
def test_disparate_impact_ratio_known_value() -> None:
    # unprivileged flagged 2/4 = 0.5; privileged flagged 4/4 = 1.0 -> DI = 0.5
    y_pred = np.array([1, 1, 0, 0, 1, 1, 1, 1])
    sensitive = np.array(["u", "u", "u", "u", "p", "p", "p", "p"])
    di = fu.disparate_impact_ratio(y_pred, sensitive, privileged="p", unprivileged="u")
    assert di == pytest.approx(0.5)


def test_disparate_impact_ratio_zero_privileged_raises() -> None:
    y_pred = np.array([1, 1, 0, 0])
    sensitive = np.array(["u", "u", "p", "p"])
    with pytest.raises(ZeroDivisionError):
        fu.disparate_impact_ratio(y_pred, sensitive, privileged="p", unprivileged="u")


def test_demographic_parity_difference_sign() -> None:
    y_pred = np.array([1, 1, 1, 0, 0, 0])
    sensitive = np.array(["u", "u", "u", "p", "p", "p"])
    diff = fu.demographic_parity_difference(y_pred, sensitive, privileged="p", unprivileged="u")
    assert diff == pytest.approx(1.0)


def test_demographic_parity_difference_missing_group_raises() -> None:
    y_pred = np.array([1, 0, 1])
    sensitive = np.array(["p", "p", "p"])
    with pytest.raises(ValueError):
        fu.demographic_parity_difference(y_pred, sensitive, privileged="p", unprivileged="u")


def test_demographic_parity_difference_missing_privileged_group_raises() -> None:
    y_pred = np.array([1, 0, 1])
    sensitive = np.array(["u", "u", "u"])
    with pytest.raises(ValueError):
        fu.demographic_parity_difference(y_pred, sensitive, privileged="p", unprivileged="u")


def test_demographic_parity_difference_empty_inputs_raise() -> None:
    y_pred = np.array([])
    sensitive = np.array([])
    with pytest.raises(ValueError):
        fu.demographic_parity_difference(y_pred, sensitive, privileged="p", unprivileged="u")


def test_demographic_parity_difference_same_group_raises() -> None:
    y_pred = np.array([1, 0, 1, 0])
    sensitive = np.array(["p", "p", "u", "u"])
    with pytest.raises(ValueError):
        fu.demographic_parity_difference(y_pred, sensitive, privileged="p", unprivileged="p")


def test_disparate_impact_ratio_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        fu.disparate_impact_ratio(
            y_pred=np.array([1, 0]),
            sensitive=np.array(["p"]),
            privileged="p",
            unprivileged="u",
        )


def test_equalized_odds_difference_keys() -> None:
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 1, 0, 0])
    sensitive = np.array(["u", "u", "u", "p", "p", "p"])
    out = fu.equalized_odds_difference(y_true, y_pred, sensitive, "p", "u")
    assert set(out) == {"tpr_difference", "fpr_difference"}


def test_equalized_odds_difference_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        fu.equalized_odds_difference(
            y_true=np.array([1, 0]),
            y_pred=np.array([1]),
            sensitive=np.array(["u"]),
            privileged="p",
            unprivileged="u",
        )


# --------------------------------------------------------------------------- #
# FraudModelPipeline
# --------------------------------------------------------------------------- #
def test_pipeline_fit_predict_roundtrip(transactions: pd.DataFrame) -> None:
    numeric = ["amount", "num_txns_24h", "account_tenure_days", "customer_age"]
    categorical = ["merchant_category", "device_type", "customer_region"]
    pipe = fu.FraudModelPipeline(
        estimator=LogisticRegression(max_iter=500, class_weight="balanced"),
        numeric_features=numeric,
        categorical_features=categorical,
    )
    pipe.fit(transactions, transactions["is_fraud"])
    proba = pipe.predict_proba(transactions)
    preds = pipe.predict(transactions)
    assert proba.shape == (len(transactions), 2)
    assert set(np.unique(preds)).issubset({0, 1})


def test_pipeline_predict_before_fit_raises() -> None:
    pipe = fu.FraudModelPipeline(
        estimator=LogisticRegression(),
        numeric_features=["amount"],
        categorical_features=["device_type"],
    )
    with pytest.raises(RuntimeError):
        pipe.predict(pd.DataFrame({"amount": [1.0], "device_type": ["web"]}))


def test_pipeline_missing_columns_raises(transactions: pd.DataFrame) -> None:
    pipe = fu.FraudModelPipeline(
        estimator=LogisticRegression(max_iter=200),
        numeric_features=["amount"],
        categorical_features=["device_type"],
    )
    pipe.fit(transactions, transactions["is_fraud"])
    with pytest.raises(KeyError):
        pipe.predict(pd.DataFrame({"amount": [10.0]}))  # device_type missing
