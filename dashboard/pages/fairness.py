
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


def render():

    st.title("Fairness Evaluation")

    st.write(
        "Evaluate differences in model outcomes across "
        "sensitive demographic groups using group-wise metrics."
    )

    reports_dir = Path(__file__).resolve().parents[2] / "reports"

    summary_path = reports_dir / "fairness_summary.csv"
    sex_path = reports_dir / "fairness_by_sex.csv"
    race_path = reports_dir / "fairness_by_race.csv"

    if not all(path.exists() for path in [
        summary_path, sex_path, race_path
    ]):
        st.error("One or more fairness report files are missing.")
        return

    summary = pd.read_csv(summary_path)
    sex = pd.read_csv(sex_path)
    race = pd.read_csv(race_path)

    st.info(
        "Dataset: Adult Census Income | "
        "Sensitive attributes: Sex and Race"
    )

    # Fairness summary
    st.header("1. Fairness Metric Summary")

    st.write(
        "Demographic parity difference measures the largest "
        "difference in positive prediction rates between groups. "
        "Equal opportunity difference measures the largest "
        "difference in true-positive rates between groups."
    )

    summary_display = summary.rename(columns={
        "demographic_parity_difference": "Demographic Parity Difference",
        "equal_opportunity_difference": "Equal Opportunity Difference"
    })

    st.dataframe(
        summary_display,
        use_container_width=True,
        hide_index=True
    )

    summary_long = summary.melt(
        id_vars="Attribute",
        value_vars=[
            "demographic_parity_difference",
            "equal_opportunity_difference"
        ],
        var_name="Metric",
        value_name="Difference"
    )

    summary_long["Metric"] = summary_long["Metric"].map({
        "demographic_parity_difference": "Demographic Parity",
        "equal_opportunity_difference": "Equal Opportunity"
    })

    summary_fig = px.bar(
        summary_long,
        x="Attribute",
        y="Difference",
        color="Metric",
        barmode="group",
        title="Fairness Differences by Sensitive Attribute",
        text_auto=".3f"
    )

    st.plotly_chart(summary_fig, use_container_width=True)
    st.divider()

    # Select attribute
    st.header("2. Group-wise Evaluation")

    attribute = st.selectbox(
        "Select sensitive attribute",
        ["Sex", "Race"]
    )

    group_df = sex.copy() if attribute == "Sex" else race.copy()

    st.write(f"Group-wise results for **{attribute}**.")

    # Selection rates
    selection_fig = px.bar(
        group_df,
        x="group",
        y="selection_rate",
        text_auto=".3f",
        title=f"Positive Prediction Rate by {attribute}"
    )

    selection_fig.update_layout(
        xaxis_title=attribute,
        yaxis_title="Selection Rate",
        yaxis_tickformat=".0%"
    )

    st.plotly_chart(selection_fig, use_container_width=True)

    # True positive rates
    tpr_fig = px.bar(
        group_df,
        x="group",
        y="true_positive_rate",
        text_auto=".3f",
        title=f"True Positive Rate by {attribute}"
    )

    tpr_fig.update_layout(
        xaxis_title=attribute,
        yaxis_title="True Positive Rate",
        yaxis_tickformat=".0%"
    )

    st.plotly_chart(tpr_fig, use_container_width=True)

    # Sample count by group
    st.subheader("Sample Distribution by Group")

    sample_fig = px.bar(
        group_df,
        x="group",
        y="sample_count",
        text="sample_count",
        title=f"Number of Evaluated Samples by {attribute}"
    )

    sample_fig.update_layout(
        xaxis_title=attribute,
        yaxis_title="Number of Samples"
    )

    st.plotly_chart(sample_fig, use_container_width=True)

    # Other group metrics
    st.subheader("Classification Metrics by Group")

    metric_long = group_df.melt(
        id_vars="group",
        value_vars=["accuracy", "precision", "recall", "f1"],
        var_name="Metric",
        value_name="Score"
    )

    metric_fig = px.bar(
        metric_long,
        x="group",
        y="Score",
        color="Metric",
        barmode="group",
        title=f"Classification Metrics by {attribute}"
    )

    metric_fig.update_layout(
        xaxis_title=attribute,
        yaxis_title="Score",
        yaxis_tickformat=".0%"
    )

    st.plotly_chart(metric_fig, use_container_width=True)

    # Group table
    st.subheader("Detailed Group Results")

    st.dataframe(
        group_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # Interpretation
    st.header("3. Experimental Interpretation")

    st.markdown("""
    - **Demographic parity difference:** Compares positive prediction
      rates across demographic groups.
    - **Equal opportunity difference:** Compares true-positive rates
      across demographic groups.
    - **Selection rate:** Proportion of samples receiving a positive
      prediction.
    - **True-positive rate:** Proportion of actual positive cases
      correctly identified.
    - **Sample count:** Number of evaluated observations in each group.
    """)

    st.warning(
        "These metrics describe disparities in this model's predictions "
        "on the evaluated test data. They do not establish causation, "
        "prove discrimination, or determine whether a model is fair "
        "under every definition. Small group sample sizes, especially "
        "for some race categories, make estimates less stable."
    )
