
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


def render():

    st.title("Deep Learning Evaluation")

    st.write(
        "Explore custom CNN experiments and ResNet18 "
        "transfer learning on the CIFAR-10 dataset."
    )

    reports_dir = Path(__file__).resolve().parents[2] / "reports"

    cnn_path = reports_dir / "cnn_experiment_comparison.csv"
    resnet_path = reports_dir / "resnet18_experiment_comparison.csv"
    class_path = reports_dir / "resnet18_classification_report.csv"

    required_files = [cnn_path, resnet_path, class_path]

    if not all(path.exists() for path in required_files):
        st.error("One or more deep learning report files are missing.")
        return

    cnn = pd.read_csv(cnn_path)
    resnet = pd.read_csv(resnet_path)
    class_report = pd.read_csv(class_path)

    st.info(
        "Dataset: CIFAR-10 | "
        "Models: Custom CNN and ResNet18"
    )

    # CNN experiments
    st.header("1. Custom CNN Experiments")

    st.write(
        "Compare the baseline CNN, data augmentation, "
        "and augmentation with regularization."
    )

    cnn_fig = px.bar(
        cnn,
        x="experiment",
        y="test_accuracy",
        text="test_accuracy",
        title="CNN Test Accuracy Comparison",
        labels={
            "experiment": "Experiment",
            "test_accuracy": "Test Accuracy (%)"
        }
    )

    cnn_fig.update_layout(
        yaxis_title="Test Accuracy (%)"
    )

    st.plotly_chart(cnn_fig, use_container_width=True)

    # CNN training versus testing
    st.subheader("Training vs Test Accuracy")

    cnn_training = cnn.melt(
        id_vars="experiment",
        value_vars=["training_accuracy", "test_accuracy"],
        var_name="Dataset Split",
        value_name="Accuracy (%)"
    )

    cnn_training["Dataset Split"] = cnn_training[
        "Dataset Split"
    ].map({
        "training_accuracy": "Training",
        "test_accuracy": "Testing"
    })

    training_fig = px.bar(
        cnn_training,
        x="experiment",
        y="Accuracy (%)",
        color="Dataset Split",
        barmode="group",
        title="CNN Training and Test Accuracy"
    )

    st.plotly_chart(training_fig, use_container_width=True)

    st.subheader("CNN Classification Metrics")

    st.dataframe(
        cnn,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ResNet experiments
    st.header("2. ResNet18 Transfer Learning")

    st.write(
        "Compare frozen-backbone feature extraction with "
        "fine-tuning of the final ResNet18 block."
    )

    resnet_display = resnet.copy()

    resnet_display["accuracy_percent"] = (
        resnet_display["accuracy"] * 100
    )

    resnet_display["macro_f1_percent"] = (
        resnet_display["macro_f1"] * 100
    )

    resnet_long = resnet_display.melt(
        id_vars="experiment",
        value_vars=[
            "accuracy_percent",
            "macro_f1_percent"
        ],
        var_name="Metric",
        value_name="Score (%)"
    )

    resnet_long["Metric"] = resnet_long["Metric"].map({
        "accuracy_percent": "Accuracy",
        "macro_f1_percent": "Macro F1"
    })

    resnet_fig = px.bar(
        resnet_long,
        x="experiment",
        y="Score (%)",
        color="Metric",
        barmode="group",
        title="ResNet18 Experiment Comparison"
    )

    st.plotly_chart(resnet_fig, use_container_width=True)

    st.subheader("ResNet18 Results")

    st.dataframe(
        resnet_display[
            ["experiment", "accuracy_percent", "macro_f1_percent"]
        ],
        use_container_width=True,
        hide_index=True
    )

    # Class-wise evaluation
    st.divider()

    st.header("3. ResNet18 Class-wise Evaluation")

    class_report = class_report.rename(
        columns={"Unnamed: 0": "class"}
    )

    cifar_classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]

    class_report = class_report[
        class_report["class"].isin(cifar_classes)
    ].copy()

    if not {"class", "precision", "recall", "f1-score", "support"}.issubset(
        class_report.columns
    ):
        st.warning("Unexpected classification report format.")
    else:
        class_long = class_report.melt(
            id_vars="class",
            value_vars=["precision", "recall", "f1-score"],
            var_name="Metric",
            value_name="Score"
        )

        class_fig = px.bar(
            class_long,
            x="class",
            y="Score",
            color="Metric",
            barmode="group",
            title="ResNet18 Performance Across CIFAR-10 Classes"
        )

        class_fig.update_layout(
            xaxis_title="CIFAR-10 Class",
            yaxis_title="Score",
            yaxis_tickformat=".0%"
        )

        st.plotly_chart(class_fig, use_container_width=True)

        st.dataframe(
            class_report,
            use_container_width=True,
            hide_index=True
        )

    # Interpretation
    st.divider()

    st.header("4. Experimental Interpretation")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Best Custom CNN",
            f'{cnn["test_accuracy"].max():.2f}%'
        )

    with col2:
        st.metric(
            "ResNet18 Fine-tuned",
            f'{resnet_display["accuracy_percent"].max():.2f}%'
        )

    st.markdown("""
    **Key observations**

    - Data augmentation improved CNN test accuracy over the baseline.
    - Adding regularization produced a small additional accuracy increase.
    - ResNet18 fine-tuning outperformed feature extraction in this experiment.
    - Class-wise metrics reveal differences in performance across CIFAR-10 categories.
    """)

    st.warning(
        "CNN and ResNet18 results come from different architectures "
        "and training configurations. ResNet18 uses ImageNet "
        "pretraining and 224×224 resized CIFAR-10 images. "
        "These results should be interpreted as separate experiments, "
        "not as a controlled architecture-only comparison."
    )
