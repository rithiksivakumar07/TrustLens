
import torch
import torch.nn as nn
import torch.optim as optim


class TrustLensNeuralNetwork(nn.Module):

    def __init__(self, input_features):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_features, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)

        )

    def forward(self, x):

        return self.network(x)


def train_neural_network(
    model,
    train_loader,
    epochs=50,
    learning_rate=0.001
):

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    training_losses = []

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for batch_X, batch_y in train_loader:

            optimizer.zero_grad()

            outputs = model(batch_X)

            loss = criterion(outputs, batch_y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        training_losses.append(average_loss)

    return model, training_losses
