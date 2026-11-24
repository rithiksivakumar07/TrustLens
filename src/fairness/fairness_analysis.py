
import pandas as pd

from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    selection_rate,
    true_positive_rate
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_group_performance(y_true, y_pred, sensitive_features):
    """Calculate model performance separately for demographic groups."""

    metrics = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "selection_rate": selection_rate,
        "true_positive_rate": true_positive_rate
    }

    frame = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )

    group_results = frame.by_group.reset_index()
    group_results = group_results.rename(
        columns={group_results.columns[0]: "group"}
    )

    group_counts = (
        pd.Series(sensitive_features)
        .value_counts()
        .rename_axis("group")
        .reset_index(name="sample_count")
    )

    group_results = group_results.merge(
        group_counts,
        on="group",
        how="left"
    )

    return frame, group_results


def evaluate_fairness(y_true, y_pred, sensitive_features):
    """Calculate demographic parity and equal opportunity differences."""

    frame, group_results = evaluate_group_performance(
        y_true,
        y_pred,
        sensitive_features
    )

    dp_difference = demographic_parity_difference(
        y_true,
        y_pred,
        sensitive_features=sensitive_features
    )

    tpr_values = frame.by_group["true_positive_rate"]

    eo_difference = (
        tpr_values.max() - tpr_values.min()
    )

    fairness_summary = {
        "demographic_parity_difference": dp_difference,
        "equal_opportunity_difference": eo_difference
    }

    return fairness_summary, group_results
