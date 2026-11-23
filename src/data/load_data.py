
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_breast_cancer_data(test_size=0.20, random_state=42):
    """
    Load and split the Breast Cancer Wisconsin dataset.

    Returns:
        X_train, X_test, y_train, y_test, dataset
    """

    dataset = load_breast_cancer()

    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, dataset
