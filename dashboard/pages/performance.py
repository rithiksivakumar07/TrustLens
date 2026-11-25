
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"


def load_report(filename):
    path = REPORTS / filename

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def render():

    st.title("Model Performance Evaluation")

    st.markdown(
        "Evaluate classical machine learning models and "
        "a PyTorch neural network on the Breast Cancer Wisconsin dataset."
    )

    st.info(
        "Dataset: Breast Cancer Wisconsin | "
        "Test split: 20% | Random state: 42"
    )

    classical = load_report("model_comparison.csv")
    neural = load_report("neural_network_results.csv")

    if classical.empty or neural.empty:
        st.error("Required experiment reports are missing.")
        return

    # --------------------------------------------------
    # PREPARE COMPARISON DATA
    # --------------------------------------------------

    comparison = pd.DataFrame([
        {
            "Model": row["model"],
            "Accuracy": row["accuracy"] * 100,
            "Benign Precision": row["precision"] * 100,
            "Benign Recall": row["recall"] * 100
        }
        for _, row in classical.iterrows()
    ])

    nn = neural.iloc[0]

    comparison = pd.concat([
        comparison,
        pd.DataFrame([{
            "Model": nn["model"],
            "Accuracy": nn["accuracy"] * 100,
            "Benign Precision": nn["benign_precision"] * 100,
            "Benign Recall": nn["benign_recall"] * 100
        }])
    ], ignore_index=True)

    # --------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------

    best_model = comparison.loc[
        comparison["Accuracy"].idxmax()
    ]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Best Accuracy",
        f'{best_model["Accuracy"]:.2f}%'
    )

    col2.metric(
        "Models Evaluated",
        len(comparison)
    )

    col3.metric(
        "Best Model",
        best_model["Model"]
    )

    st.divider()

    # --------------------------------------------------
    # ACCURACY COMPARISON
    # --------------------------------------------------

    st.subheader("Accuracy Comparison")

    accuracy_chart = px.bar(
        comparison,
        x="Model",
        y="Accuracy",
        text="Accuracy",
        title="Test Accuracy Across Models",
        labels={"Accuracy": "Accuracy (%)"},
        color="Model"
    )

    accuracy_chart.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    accuracy_chart.update_layout(
        yaxis_range=[0, 105],
        showlegend=False
    )

    st.plotly_chart(
        accuracy_chart,
        use_container_width=True
    )

    # --------------------------------------------------
    # PRECISION AND RECALL
    # --------------------------------------------------

    st.subheader("Benign Class: Precision and Recall")

    metric_data = comparison.melt(
        id_vars="Model",
        value_vars=[
            "Benign Precision",
            "Benign Recall"
        ],
        var_name="Metric",
        value_name="Score"
    )

    metric_chart = px.bar(
        metric_data,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Benign Class Precision vs Recall",
        labels={"Score": "Score (%)"}
    )

    metric_chart.update_layout(
        yaxis_range=[0, 105]
    )

    st.plotly_chart(
        metric_chart,
        use_container_width=True
    )

    st.caption(
        "Precision and recall refer to the benign class "
        "(class label 1) for all three models."
    )

    # --------------------------------------------------
    # MODEL RESULTS TABLE
    # --------------------------------------------------

    st.subheader("Detailed Model Results")

    display_data = comparison.copy()

    for column in [
        "Accuracy",
        "Benign Precision",
        "Benign Recall"
    ]:
        display_data[column] = display_data[column].map(
            lambda x: f"{x:.2f}%"
        )

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------
    # F1 SCORE NOTES
    # --------------------------------------------------

    st.subheader("F1 Score Results")

    f1_data = pd.DataFrame([
        {
            "Model": row["model"],
            "F1 Score": row["f1_score"],
            "F1 Definition": "Benign-class F1"
        }
        for _, row in classical.iterrows()
    ])

    f1_data = pd.concat([
        f1_data,
        pd.DataFrame([{
            "Model": nn["model"],
            "F1 Score": nn["macro_f1"],
            "F1 Definition": "Macro F1"
        }])
    ], ignore_index=True)

    st.dataframe(
        f1_data,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Note: Classical model F1 scores are for the benign class, "
        "whereas the neural network report provides macro F1. "
        "They are not directly equivalent."
    )

    st.divider()

    # --------------------------------------------------
    # CONFUSION MATRICES
    # --------------------------------------------------

    st.subheader("Confusion Matrix Analysis")

    st.markdown(
        "The following matrices are available in the saved "
        "classical model evaluation report."
    )

    col1, col2 = st.columns(2)

    for column, (_, row) in zip(
        [col1, col2],
        classical.iterrows()
    ):

        matrix = ast.literal_eval(row["confusion_matrix"])

        with column:
            st.markdown(f"**{row['model']}**")

            fig = ff.create_annotated_heatmap(
                z=matrix,
                x=["Predicted Malignant", "Predicted Benign"],
                y=["Actual Malignant", "Actual Benign"],
                colorscale="Blues",
                showscale=True
            )

            fig.update_layout(
                xaxis_title="Predicted Class",
                yaxis_title="Actual Class",
                height=400
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.caption(
        "Confusion matrices are currently available for Logistic "
        "Regression and Random Forest. The neural network's "
        "confusion matrix was not saved in the existing report."
    )
