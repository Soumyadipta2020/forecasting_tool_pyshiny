"""Small modeling utilities shared by the Shiny server and tests."""

from __future__ import annotations

import warnings

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


def fit_theta_forecast(values, horizon, seasonal_period=1):
    try:
        from statsmodels.tsa.forecasting.theta import ThetaModel
    except ImportError as exc:
        raise RuntimeError("Theta requires a statsmodels version with ThetaModel support.") from exc

    y = pd.Series(np.asarray(values, dtype=float))
    period = max(2, int(seasonal_period or 2))
    if len(y) <= period * 2:
        period = 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ThetaModel(y, period=period).fit()
        future = np.asarray(fit.forecast(horizon), dtype=float)

    fitted = _theta_fitted_values(ThetaModel, y, period)
    lower = upper = None
    if hasattr(fit, "prediction_intervals"):
        try:
            intervals = fit.prediction_intervals(horizon, alpha=0.05)
            lower = np.asarray(intervals["lower"], dtype=float)
            upper = np.asarray(intervals["upper"], dtype=float)
        except Exception:
            lower = upper = None

    summary = [
        "Model: Theta",
        "Engine: statsmodels ThetaModel",
        f"Period: {period}",
        "Fitted values: rolling one-step Theta forecasts.",
    ]
    try:
        summary.append(str(fit.summary()))
    except Exception:
        pass
    return fitted, future, "\n".join(summary), lower, upper


def _theta_fitted_values(theta_model, y, period):
    fitted = np.full(len(y), np.nan)
    for idx in range(2, len(y)):
        train = y.iloc[:idx]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted[idx] = float(np.asarray(theta_model(train, period=period).fit().forecast(1), dtype=float)[0])
        except Exception:
            fitted[idx] = float(train.iloc[-1])
    return fitted


def fit_volatility_forecast(values, horizon, model_name):
    y = np.asarray(values, dtype=float)
    if len(y) < 3:
        raise RuntimeError("ARCH/GARCH models need at least 3 valid observations.")

    try:
        return _fit_arch_package_forecast(y, horizon, model_name)
    except ImportError:
        return _fit_arch_garch_fallback(y, horizon, model_name, "optional arch package is not installed")
    except Exception as exc:
        return _fit_arch_garch_fallback(y, horizon, model_name, f"arch package fit failed: {exc}")


def _fit_arch_package_forecast(y, horizon, model_name):
    from arch import arch_model

    vol = "ARCH" if model_name == "ARCH" else "GARCH"
    p = 1
    q = 0 if model_name == "ARCH" else 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = arch_model(y, mean="AR", lags=1, vol=vol, p=p, q=q, rescale=False).fit(disp="off")

    params = fit.params
    const = float(params.get("Const", params.get("mu", np.nanmean(y))))
    lag_name = next((name for name in params.index if str(name).endswith("[1]")), None)
    lag_coef = float(params.get(lag_name, 0.0)) if lag_name is not None else 0.0
    fitted = np.full(len(y), np.nan)
    fitted[1:] = const + lag_coef * y[:-1]

    forecast = fit.forecast(horizon=horizon, reindex=False)
    future = np.asarray(forecast.mean.iloc[-1].to_numpy(), dtype=float)
    variance = np.asarray(forecast.variance.iloc[-1].to_numpy(), dtype=float)
    radius = 1.96 * np.sqrt(np.maximum(variance, 0))
    summary = f"Model: {model_name}\nEngine: arch.arch_model\nVolatility: {vol}({p}, {q})\n{fit.summary()}"
    return fitted, future, summary, future - radius, future + radius


def _fit_arch_garch_fallback(y, horizon, model_name, reason):
    fitted, future, residuals = _ar1_mean_forecast(y, horizon)
    residuals = residuals[np.isfinite(residuals)]
    base_var = float(np.var(residuals, ddof=1)) if len(residuals) > 1 else float(np.var(y, ddof=1))
    if not np.isfinite(base_var) or base_var <= 0:
        diffs = np.diff(y)
        base_var = float(np.var(diffs, ddof=1)) if len(diffs) > 1 else 1.0
    base_var = max(base_var, 1e-6)

    if model_name == "ARCH":
        alpha = 0.35
        beta = 0.0
    else:
        alpha = 0.12
        beta = 0.82
    omega = base_var * max(1 - alpha - beta, 0.05)
    last_error2 = float(residuals[-1] ** 2) if len(residuals) else base_var
    last_var = base_var
    variances = []
    for _ in range(horizon):
        next_var = omega + alpha * last_error2 + beta * last_var
        next_var = max(float(next_var), 1e-6)
        variances.append(next_var)
        last_error2 = next_var
        last_var = next_var

    variance = np.asarray(variances, dtype=float)
    radius = 1.96 * np.sqrt(variance)
    model_type = "ARCH(1)" if model_name == "ARCH" else "GARCH(1, 1)"
    summary = (
        f"Model: {model_name}\n"
        f"Engine: built-in AR(1) + {model_type} fallback\n"
        f"Fallback reason: {reason}\n"
        "Install the optional 'arch' package to use maximum-likelihood volatility fitting."
    )
    return fitted, future, summary, future - radius, future + radius


def _ar1_mean_forecast(y, horizon):
    fitted = np.full(len(y), np.nan)
    if len(y) >= 2:
        x = np.column_stack([np.ones(len(y) - 1), y[:-1]])
        target = y[1:]
        try:
            const, phi = np.linalg.lstsq(x, target, rcond=None)[0]
        except Exception:
            const, phi = float(np.nanmean(y)), 0.0
        fitted[1:] = const + phi * y[:-1]
    else:
        const, phi = float(y[-1]), 0.0

    future = []
    current = float(y[-1])
    for _ in range(horizon):
        current = float(const + phi * current)
        future.append(current)

    residuals = y - fitted
    return fitted, np.asarray(future, dtype=float), residuals
