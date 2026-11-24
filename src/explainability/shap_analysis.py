
import shap
import numpy as np
import pandas as pd


def explain_random_forest(
    model,
    X_background,
    X_data,
    feature_names,
    class_index=0
):
    """
    Generate SHAP explanations for a Random Forest model.

    class_index:
        0 = Malignant
        1 = Benign
    """

    explainer = shap.TreeExplainer(model)

    explanation = explainer(X_data)

    # Select the requested class when SHAP returns
    # a separate output dimension for each class.
    if explanation.values.ndim == 3:
        explanation = explanation[:, :, class_index]

    explanation.feature_names = list(feature_names)

    return explainer, explanation


def explain_logistic_regression(
    model,
    X_background,
    X_data,
    feature_names
):
    """
    Generate SHAP explanations for Logistic Regression.

    For binary classification, the default explanation
    represents the class-1 decision score (log-odds).
    """

    explainer = shap.LinearExplainer(
        model,
        X_background
    )

    explanation = explainer(X_data)

    explanation.feature_names = list(feature_names)

    return explainer, explanation


def get_global_importance(explanation):
    """
    Calculate mean absolute SHAP importance
    for every feature.
    """

    importance = np.abs(
        explanation.values
    ).mean(axis=0)

    results = pd.DataFrame({
        "feature": explanation.feature_names,
        "mean_absolute_shap": importance
    })

    return results.sort_values(
        by="mean_absolute_shap",
        ascending=False
    ).reset_index(drop=True)


def plot_global_importance(
    explanation,
    max_display=15
):
    """
    Generate global SHAP feature importance plot.
    """

    shap.plots.bar(
        explanation,
        max_display=max_display
    )


def plot_beeswarm(
    explanation,
    max_display=15
):
    """
    Generate SHAP beeswarm visualization.
    """

    shap.plots.beeswarm(
        explanation,
        max_display=max_display
    )


def plot_waterfall(
    explanation,
    sample_index,
    max_display=15
):
    """
    Generate an individual prediction explanation.
    """

    shap.plots.waterfall(
        explanation[sample_index],
        max_display=max_display
    )
