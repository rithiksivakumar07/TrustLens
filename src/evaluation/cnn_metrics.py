
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def evaluate_cnn(model, test_loader, device, class_names):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device, non_blocking=True)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    results = {
        "accuracy": accuracy_score(all_labels, all_predictions),

        "macro_precision": precision_score(
            all_labels, all_predictions,
            average="macro",
            zero_division=0
        ),

        "macro_recall": recall_score(
            all_labels, all_predictions,
            average="macro",
            zero_division=0
        ),

        "macro_f1": f1_score(
            all_labels, all_predictions,
            average="macro",
            zero_division=0
        ),

        "classification_report": classification_report(
            all_labels,
            all_predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        ),

        "confusion_matrix": confusion_matrix(
            all_labels, all_predictions
        )
    }

    return results
