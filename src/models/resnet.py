
import torch
import torch.nn as nn
from torchvision.models import resnet18


def create_resnet18(num_classes=10, fine_tune=True, device="cpu"):
    """
    Create a ResNet18 model for image classification.

    fine_tune=True:
        Train layer4 and the final classifier.

    fine_tune=False:
        Freeze the pretrained backbone and train only the classifier.
    """

    model = resnet18(weights=None)

    # Replace ImageNet's 1000-class output layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Always train the final classifier
    for param in model.fc.parameters():
        param.requires_grad = True

    # Optionally fine-tune the final residual block
    if fine_tune:
        for param in model.layer4.parameters():
            param.requires_grad = True

    return model.to(device)


def count_trainable_parameters(model):
    """Count parameters that will receive gradient updates."""

    return sum(
        param.numel()
        for param in model.parameters()
        if param.requires_grad
    )
