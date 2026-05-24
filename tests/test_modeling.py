import math

from server_scripts.helpers.modeling import (
    classification_metrics,
    decode_numeric_logistic_predictions,
    fit_theta_forecast,
    fit_volatility_forecast,
    interval_quality_label,
    numeric_logistic_target,
    regression_metrics,
)


def test_regression_metrics_include_interval_quality():
    metrics = regression_metrics(
        actual=[10, 12, 14, 16],
        predicted=[9, 13, 13, 17],
        training_series=[8, 10, 12, 14],
        lower=[8, 11, 12, 15],
        upper=[12, 15, 16, 19],
    )

    assert metrics["MAE"] == 1
    assert metrics["Coverage"] == 100
    assert metrics["Avg Width"] == 4
    assert math.isclose(metrics["Width/Mean"], 30.76923076923077)


def test_classification_metrics_include_f1_precision_recall():
    metrics = classification_metrics(["yes", "no", "yes", "no"], ["yes", "no", "no", "no"])

    assert metrics["Accuracy"] == 75
    assert metrics["F1"] > 0
    assert metrics["Precision"] > 0
    assert metrics["Recall"] > 0


def test_interval_quality_label_flags_weak_coverage():
    title, _, color = interval_quality_label({"Coverage": 40, "Width/Mean": 25})

    assert title == "Weak intervals"
    assert color == "danger"


def test_theta_forecast_returns_fitted_forecast_and_intervals():
    fitted, future, summary, lower, upper = fit_theta_forecast(
        [120, 128, 135, 150, 168, 180, 194, 210, 225, 240, 260, 282],
        horizon=3,
        seasonal_period=4,
    )

    assert len(fitted) == 12
    assert len(future) == 3
    assert len(lower) == 3
    assert len(upper) == 3
    assert "statsmodels ThetaModel" in summary
    assert all(math.isfinite(value) for value in future)


def test_arch_and_garch_forecasts_fallback_when_optional_package_is_missing():
    values = [500, 659, 453, 756, 823, 983, 821, 1040, 1175, 1210, 1305, 1420]

    for model_name in ("ARCH", "GARCH"):
        fitted, future, summary, lower, upper = fit_volatility_forecast(values, horizon=3, model_name=model_name)

        assert len(fitted) == len(values)
        assert len(future) == 3
        assert len(lower) == 3
        assert len(upper) == 3
        assert model_name in summary
        assert all(math.isfinite(value) for value in future)
        assert all(lo <= mid <= hi for lo, mid, hi in zip(lower, future, upper))


def test_numeric_logistic_target_bins_continuous_response_for_regression_forecasts():
    labels, label_medians = numeric_logistic_target([500, 659, 453, 756, 823, 983, 821, 1040, 1175])
    decoded = decode_numeric_logistic_predictions(labels[:4], label_medians, fallback=0.0)

    assert labels.nunique() >= 2
    assert len(decoded) == 4
    assert all(math.isfinite(value) for value in decoded)
