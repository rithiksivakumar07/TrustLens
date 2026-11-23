
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained binary classification model."""

    predictions = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, pos_label=1
        ),
        "recall": recall_score(
            y_test, predictions, pos_label=1
        ),
        "f1_score": f1_score(
            y_test, predictions, pos_label=1
        ),
        "malignant_recall": recall_score(
            y_test, predictions, pos_label=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist()
    }

    return results
