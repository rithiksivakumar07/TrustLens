
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score


def generate_gaussian_noise(X, feature_std, sigma, random_state=42):
    """Generate Gaussian-noise perturbed raw input features."""
    rng = np.random.default_rng(random_state)
    noise = rng.normal(0.0, 1.0, size=X.shape)
    return X + sigma * feature_std * noise


def generate_fgsm_examples(model, X, y, epsilon):
    """
    Generate FGSM adversarial examples for a binary PyTorch classifier.

    Inputs are expected to be in the same standardized feature space
    used when training the model.
    """
    model.eval()

    X_input = X.clone().detach().requires_grad_(True)
    y_input = y.clone().detach()

    criterion = nn.BCEWithLogitsLoss()

    model.zero_grad()
    outputs = model(X_input)
    loss = criterion(outputs, y_input)
    loss.backward()

    perturbation = epsilon * X_input.grad.sign()
    X_adversarial = (X_input + perturbation).detach()

    return X_adversarial, loss.item()


def evaluate_pytorch_model(model, X, y):
    """Evaluate a binary PyTorch classifier."""
    model.eval()

    with torch.no_grad():
        outputs = model(X)
        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities >= 0.5).int()

    accuracy = accuracy_score(
        y.cpu().numpy().flatten(),
        predictions.cpu().numpy().flatten()
    )

    return accuracy


def evaluate_fgsm(model, X, y, epsilon_values):
    """Evaluate a PyTorch model across multiple FGSM strengths."""
    clean_accuracy = evaluate_pytorch_model(model, X, y)

    criterion = nn.BCEWithLogitsLoss()
    results = []

    for epsilon in epsilon_values:
        X_adversarial, _ = generate_fgsm_examples(
            model, X, y, epsilon
        )

        with torch.no_grad():
            outputs = model(X_adversarial)
            adversarial_loss = criterion(outputs, y).item()

        adversarial_accuracy = evaluate_pytorch_model(
            model, X_adversarial, y
        )

        results.append({
            "Model": "Neural Network",
            "Epsilon": epsilon,
            "Accuracy": adversarial_accuracy,
            "Accuracy Drop (%)": (
                clean_accuracy - adversarial_accuracy
            ) * 100,
            "Adversarial Loss": adversarial_loss
        })

    return pd.DataFrame(results)
