
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


def render():

    st.title("Model Robustness Evaluation")

    st.write(
        "Evaluate how machine learning models respond to "
        "Gaussian noise and adversarial perturbations."
    )

    reports_dir = Path(__file__).resolve().parents[2] / "reports"

    gaussian_path = reports_dir / "gaussian_noise_robustness.csv"
    fgsm_path = reports_dir / "fgsm_robustness.csv"

    if not gaussian_path.exists() or not fgsm_path.exists():
        st.error("Robustness report files are missing.")
        return

    gaussian = pd.read_csv(gaussian_path)
    fgsm = pd.read_csv(fgsm_path)

    st.info(
        "Dataset: Breast Cancer Wisconsin | "
        "Evaluation: Gaussian noise and FGSM"
    )

    # Gaussian noise analysis
    st.header("1. Gaussian Noise Robustness")

    st.write(
        "Gaussian noise is added to input features at increasing "
        "noise levels. Accuracy is measured for each model."
    )

    gaussian_fig = px.line(
        gaussian,
        x="Noise Level",
        y="Accuracy",
        color="Model",
        markers=True,
        title="Model Accuracy Under Gaussian Noise"
    )

    gaussian_fig.update_layout(
        xaxis_title="Noise Level (Standardized Feature Scale)",
        yaxis_title="Test Accuracy",
        yaxis_tickformat=".0%"
    )

    st.plotly_chart(gaussian_fig, use_container_width=True)

    gaussian_drop_fig = px.line(
        gaussian,
        x="Noise Level",
        y="Accuracy Drop (%)",
        color="Model",
        markers=True,
        title="Accuracy Change Under Gaussian Noise"
    )

    gaussian_drop_fig.update_layout(
        xaxis_title="Noise Level",
        yaxis_title="Accuracy Change (%)"
    )

    st.plotly_chart(gaussian_drop_fig, use_container_width=True)

    st.subheader("Gaussian Noise Results")

    st.dataframe(
        gaussian,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # FGSM analysis
    st.header("2. FGSM Adversarial Robustness")

    st.write(
        "Fast Gradient Sign Method (FGSM) perturbs input features "
        "using the gradient of the model loss. Epsilon controls "
        "the perturbation magnitude."
    )

    fgsm_fig = px.line(
        fgsm,
        x="Epsilon",
        y="Accuracy",
        markers=True,
        title="Neural Network Accuracy Under FGSM"
    )

    fgsm_fig.update_layout(
        xaxis_title="FGSM Epsilon",
        yaxis_title="Test Accuracy",
        yaxis_tickformat=".0%"
    )

    st.plotly_chart(fgsm_fig, use_container_width=True)

    loss_fig = px.line(
        fgsm,
        x="Epsilon",
        y="Adversarial Loss",
        markers=True,
        title="Adversarial Loss Under FGSM"
    )

    loss_fig.update_layout(
        xaxis_title="FGSM Epsilon",
        yaxis_title="Adversarial Loss"
    )

    st.plotly_chart(loss_fig, use_container_width=True)

    st.subheader("FGSM Results")

    st.dataframe(
        fgsm,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # Interpretation
    st.header("3. Experimental Interpretation")

    baseline = fgsm.loc[fgsm["Epsilon"].idxmin()]
    strongest = fgsm.loc[fgsm["Epsilon"].idxmax()]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Baseline Accuracy",
        f'{baseline["Accuracy"] * 100:.2f}%'
    )

    col2.metric(
        "Accuracy at Maximum Epsilon",
        f'{strongest["Accuracy"] * 100:.2f}%'
    )

    col3.metric(
        "Accuracy Drop",
        f'{strongest["Accuracy Drop (%)"]:.2f} pp'
    )

    st.markdown("""
    **How to interpret these experiments**

    - Gaussian noise evaluates sensitivity to random input disturbances.
    - FGSM evaluates sensitivity to gradient-based adversarial perturbations.
    - Higher FGSM epsilon represents a larger perturbation.
    - Accuracy changes under Gaussian noise may not be monotonic;
      random perturbations can occasionally correct existing errors.
    """)

    st.warning(
        "FGSM results apply only to the PyTorch neural network. "
        "Epsilon is measured in standardized feature space. "
        "These experiments do not establish robustness against "
        "all possible real-world attacks or disturbances."
    )
