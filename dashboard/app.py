
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="TrustLens | ML Evaluation",
    page_icon="🔍",
    layout="wide"
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

@st.cache_data
def load_report(filename):
    path = REPORTS / filename

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


classical = load_report("model_comparison.csv")
neural = load_report("neural_network_results.csv")
cnn = load_report("cnn_experiment_comparison.csv")
resnet = load_report("resnet18_experiment_comparison.csv")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🔍 TrustLens")
st.sidebar.caption("ML Evaluation Framework")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Model Performance",
        "Explainability",
        "Robustness",
        "Fairness",
        "Deep Learning"
    ]
)

st.sidebar.divider()
st.sidebar.caption("TrustLens | Day 9 Development")

# --------------------------------------------------
# OVERVIEW PAGE
# --------------------------------------------------

if page == "Overview":

    st.title("🔍 TrustLens Dashboard")
    st.subheader(
        "Explainable, Robust and Fair Machine Learning Evaluation"
    )

    st.markdown(
        "Explore model performance, explainability, "
        "robustness, fairness and deep learning experiments."
    )

    st.divider()

    # Classical ML metrics
    if not classical.empty:
        best_classical = classical.loc[
            classical["accuracy"].idxmax()
        ]

        classical_accuracy = best_classical["accuracy"] * 100
    else:
        classical_accuracy = 0
        best_classical = None

    # Neural network metrics
    neural_accuracy = (
        neural["accuracy"].iloc[0] * 100
        if not neural.empty else 0
    )

    # CNN metrics are already percentages
    best_cnn_accuracy = (
        cnn["test_accuracy"].max()
        if not cnn.empty else 0
    )

    # ResNet metrics are fractions
    best_resnet_accuracy = (
        resnet["accuracy"].max() * 100
        if not resnet.empty else 0
    )

    st.subheader("Experiment Highlights")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Best Classical ML",
        f"{classical_accuracy:.2f}%",
        help="Breast Cancer dataset"
    )

    col2.metric(
        "PyTorch Neural Network",
        f"{neural_accuracy:.2f}%",
        help="Breast Cancer dataset"
    )

    col3.metric(
        "Best Custom CNN",
        f"{best_cnn_accuracy:.2f}%",
        help="CIFAR-10 dataset"
    )

    col4.metric(
        "Best ResNet18",
        f"{best_resnet_accuracy:.2f}%",
        help="CIFAR-10 dataset"
    )

    st.divider()

    # Classical ML section
    st.subheader("Classical Machine Learning")
    st.caption("Dataset: Breast Cancer Wisconsin")

    if not classical.empty:
        classical_chart = px.bar(
            classical,
            x="model",
            y="accuracy",
            title="Classical Model Accuracy",
            text_auto=".2%"
        )

        classical_chart.update_layout(
            yaxis_title="Accuracy",
            xaxis_title="Model",
            yaxis_tickformat=".0%"
        )

        st.plotly_chart(
            classical_chart,
            use_container_width=True
        )

        st.dataframe(
            classical.drop(columns=["confusion_matrix"]),
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # Deep learning section
    st.subheader("Deep Learning Experiments")
    st.caption("Dataset: CIFAR-10")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Custom CNN experiments**")

        if not cnn.empty:
            cnn_chart = px.bar(
                cnn,
                x="experiment",
                y="test_accuracy",
                title="CNN Test Accuracy",
                text_auto=".2f"
            )

            cnn_chart.update_layout(
                xaxis_title="Experiment",
                yaxis_title="Test Accuracy (%)"
            )

            st.plotly_chart(
                cnn_chart,
                use_container_width=True
            )

    with col2:
        st.markdown("**ResNet18 experiments**")

        if not resnet.empty:
            resnet_display = resnet.copy()
            resnet_display["accuracy_percent"] = (
                resnet_display["accuracy"] * 100
            )

            resnet_chart = px.bar(
                resnet_display,
                x="experiment",
                y="accuracy_percent",
                title="ResNet18 Test Accuracy",
                text_auto=".2f"
            )

            resnet_chart.update_layout(
                xaxis_title="Experiment",
                yaxis_title="Test Accuracy (%)"
            )

            st.plotly_chart(
                resnet_chart,
                use_container_width=True
            )

    st.divider()

    st.caption(
        "TrustLens | Experimental results from saved project reports. "
        "Metrics from different datasets represent separate tasks "
        "and should not be interpreted as a common benchmark."
    )

# --------------------------------------------------
# OTHER PAGES - TO BE IMPLEMENTED
# --------------------------------------------------

elif page == "Model Performance":
    from pages.performance import render
    render()

elif page == "Explainability":
    from pages.explainability import render
    render()

elif page == "Robustness":
    from pages.robustness import render
    render()

elif page == "Fairness":
    from pages.fairness import render
    render()

elif page == "Deep Learning":
    from pages.deep_learning import render
    render()
else:
    st.title(page)

    st.info(
        f"The {page} module will be implemented next."
    )
