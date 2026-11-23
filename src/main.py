
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.data.load_data import load_breast_cancer_data

from src.models.train import (
    train_logistic_regression,
    train_random_forest
)

from src.evaluation.metrics import evaluate_model


def main():

    print("=" * 50)
    print("          TRUSTLENS ML ENGINE")
    print("=" * 50)

    # 1. Load dataset
    X_train, X_test, y_train, y_test, dataset = (
        load_breast_cancer_data()
    )

    print("\\nDataset loaded successfully!")

    # 2. Scale data for Logistic Regression
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Train models
    lr_model = train_logistic_regression(
        X_train_scaled, y_train
    )

    rf_model = train_random_forest(
        X_train, y_train
    )

    print("Models trained successfully!")

    # 4. Evaluate models
    lr_results = evaluate_model(
        lr_model, X_test_scaled, y_test
    )

    rf_results = evaluate_model(
        rf_model, X_test, y_test
    )

    # 5. Prepare comparison
    results = pd.DataFrame([
        {"model": "Logistic Regression", **lr_results},
        {"model": "Random Forest", **rf_results}
    ])

    # 6. Save results
    results.to_csv(
        "reports/model_comparison.csv",
        index=False
    )

    print("\\nMODEL COMPARISON")
    print(results.to_string(index=False))

    print("\\nResults saved successfully!")


if __name__ == "__main__":
    main()
