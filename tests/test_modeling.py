import math

from server_scripts.helpers.modeling import (
    classification_metrics,
    interval_quality_label,
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
