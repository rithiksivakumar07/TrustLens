
import torch
import torch.nn as nn


class TrustLensCNN(nn.Module):
    """
    Basic CNN for CIFAR-10 image classification.
    """

    def __init__(self, dropout=0.0):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        classifier_layers = [
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU()
        ]

        if dropout > 0:
            classifier_layers.append(nn.Dropout(dropout))

        classifier_layers.append(nn.Linear(128, 10))

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_cnn_model(dropout=0.0, device="cpu"):
    model = TrustLensCNN(dropout=dropout)
    return model.to(device)
