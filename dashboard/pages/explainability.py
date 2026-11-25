
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


def render():

    st.title("Model Explainability")
    st.write(
        "Understand which input features contribute most "
        "to model predictions using SHAP analysis."
    )

    reports_dir = Path(__file__).resolve().parents[2] / "reports"

    rf_path = reports_dir / "random_forest_shap_importance.csv"
    lr_path = reports_dir / "logistic_regression_shap_importance.csv"

    if not rf_path.exists() or not lr_path.exists():
        st.error("SHAP report files are missing.")
        return

    rf_df = pd.read_csv(rf_path)
    lr_df = pd.read_csv(lr_path)

    required_columns = {"feature", "mean_absolute_shap"}

    if not required_columns.issubset(rf_df.columns):
        st.error("Unexpected Random Forest SHAP report format.")
        return

    if not required_columns.issubset(lr_df.columns):
        st.error("Unexpected Logistic Regression SHAP report format.")
        return

    rf_df = rf_df.sort_values(
        "mean_absolute_shap", ascending=False
    )

    lr_df = lr_df.sort_values(
        "mean_absolute_shap", ascending=False
    )

    st.info(
        "Dataset: Breast Cancer Wisconsin | "
        "Method: Global SHAP feature importance"
    )

    st.header("Random Forest Explainability")

    st.write(
        "Features are ranked by their mean absolute SHAP values."
    )

    rf_top = rf_df.head(10).sort_values(
        "mean_absolute_shap", ascending=True
    )

    rf_fig = px.bar(
        rf_top,
        x="mean_absolute_shap",
        y="feature",
        orientation="h",
        title="Top 10 Features – Random Forest",
        labels={
            "mean_absolute_shap": "Mean Absolute SHAP Value",
            "feature": "Feature"
        }
    )

    st.plotly_chart(rf_fig, use_container_width=True)

    st.dataframe(
        rf_df.head(10),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.header("Logistic Regression Explainability")

    st.write(
        "Features are ranked by their mean absolute SHAP values."
    )

    lr_top = lr_df.head(10).sort_values(
        "mean_absolute_shap", ascending=True
    )

    lr_fig = px.bar(
        lr_top,
        x="mean_absolute_shap",
        y="feature",
        orientation="h",
        title="Top 10 Features – Logistic Regression",
        labels={
            "mean_absolute_shap": "Mean Absolute SHAP Value",
            "feature": "Feature"
        }
    )

    st.plotly_chart(lr_fig, use_container_width=True)

    st.dataframe(
        lr_df.head(10),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.header("Understanding SHAP Results")

    st.markdown("""
    - **SHAP:** A method for explaining how input features contribute
      to a model's prediction.
    - **Mean absolute SHAP:** Measures the average magnitude of
      feature contributions across evaluated samples.
    - **Higher value:** Indicates a larger average contribution
      to the model output.
    - **Global importance:** Helps identify influential features
      across the evaluated dataset.
    """)

    st.warning(
        "The Logistic Regression and Random Forest SHAP values "
        "use different model output scales. Compare feature rankings "
        "and patterns rather than directly comparing their raw values. "
        "SHAP explanations show model behaviour, not causation."
    )
