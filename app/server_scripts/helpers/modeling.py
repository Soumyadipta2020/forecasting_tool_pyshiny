"""Small modeling utilities shared by the Shiny server and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(actual, predicted, k=2, training_series=None, lower=None, upper=None):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    empty = {
        "MAE": np.nan,
        "RMSE": np.nan,
        "MAPE": np.nan,
        "sMAPE": np.nan,
        "MASE": np.nan,
        "WAPE": np.nan,
        "MdAPE": np.nan,
        "R2": np.nan,
        "BIC": np.nan,
        "Coverage": np.nan,
        "Avg Width": np.nan,
        "Width/Mean": np.nan,
    }
    if not mask.any():
        return empty

    actual = actual[mask]
    predicted = predicted[mask]
    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    ssr = float(np.sum(errors**2))
    n = len(actual)
    rmse = float(np.sqrt(ssr / n))
    nonzero = actual != 0
    mape = float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100) if nonzero.any() else np.nan
    mdape = float(np.median(np.abs(errors[nonzero] / actual[nonzero])) * 100) if nonzero.any() else np.nan
    smape_denominator = np.abs(actual) + np.abs(predicted)
    smape_mask = smape_denominator != 0
    smape = float(np.mean(2 * np.abs(errors[smape_mask]) / smape_denominator[smape_mask]) * 100) if smape_mask.any() else np.nan
    actual_sum = float(np.sum(np.abs(actual)))
    wape = float(np.sum(np.abs(errors)) / actual_sum * 100) if actual_sum else np.nan

    scale_source = np.asarray(training_series if training_series is not None else actual, dtype=float)
    scale_source = scale_source[np.isfinite(scale_source)]
    naive_scale = float(np.mean(np.abs(np.diff(scale_source)))) if len(scale_source) > 1 else np.nan
    mase = float(mae / naive_scale) if np.isfinite(naive_scale) and naive_scale != 0 else np.nan

    total = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = float(1 - ssr / total) if total else np.nan
    bic = float(n * np.log(ssr / n) + k * np.log(n)) if ssr > 0 and n > 0 else np.nan

    coverage = np.nan
    avg_width = np.nan
    width_mean = np.nan
    if lower is not None and upper is not None:
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        interval_mask = mask.copy()
        if len(lower) == len(mask) and len(upper) == len(mask):
            lower = lower[interval_mask]
            upper = upper[interval_mask]
        elif len(lower) != len(actual) or len(upper) != len(actual):
            lower = upper = np.asarray([], dtype=float)
        if len(lower) == len(actual) and len(upper) == len(actual):
            interval_valid = np.isfinite(lower) & np.isfinite(upper) & (upper >= lower)
            if interval_valid.any():
                coverage = float(np.mean((actual[interval_valid] >= lower[interval_valid]) & (actual[interval_valid] <= upper[interval_valid])) * 100)
                widths = upper[interval_valid] - lower[interval_valid]
                avg_width = float(np.mean(widths))
                denom = float(np.nanmean(np.abs(actual[interval_valid])))
                width_mean = float(avg_width / denom * 100) if denom else np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "sMAPE": smape,
        "MASE": mase,
        "WAPE": wape,
        "MdAPE": mdape,
        "R2": r2,
        "BIC": bic,
        "Coverage": coverage,
        "Avg Width": avg_width,
        "Width/Mean": width_mean,
    }


def classification_metrics(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    mask = pd.notna(actual) & pd.notna(predicted)
    if not mask.any():
        return {"Accuracy": np.nan, "F1": np.nan, "Precision": np.nan, "Recall": np.nan}

    try:
        from sklearn.metrics import f1_score, precision_score, recall_score

        average = "weighted"
        return {
            "Accuracy": float(np.mean(actual[mask] == predicted[mask]) * 100),
            "F1": float(f1_score(actual[mask], predicted[mask], average=average, zero_division=0) * 100),
            "Precision": float(precision_score(actual[mask], predicted[mask], average=average, zero_division=0) * 100),
            "Recall": float(recall_score(actual[mask], predicted[mask], average=average, zero_division=0) * 100),
        }
    except Exception:
        return {"Accuracy": float(np.mean(actual[mask] == predicted[mask]) * 100), "F1": np.nan, "Precision": np.nan, "Recall": np.nan}


def interval_quality_label(metrics):
    coverage = metrics.get("Coverage", np.nan)
    width_mean = metrics.get("Width/Mean", np.nan)
    if pd.isna(coverage):
        return "Unavailable", "Validation intervals were not available for this model.", "warning"
    if coverage >= 85 and (pd.isna(width_mean) or width_mean <= 80):
        return "Reliable intervals", "Coverage is close to the 95% target without excessive width.", "success"
    if coverage >= 70:
        return "Usable intervals", "Coverage is moderate; check whether the interval width is acceptable.", "warning"
    return "Weak intervals", "Coverage is low on validation data; treat uncertainty bands cautiously.", "danger"
