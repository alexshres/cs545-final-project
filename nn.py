import torch.nn as nn
import torch.nn.functional as F


class MNISTNeural(nn.Module):
    def __init__(self):
        super().__init__()

        # Will turn our 28 x 28 MNIST dataset into a 784 x 1 vector
        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 300),
                nn.ReLU(),
                nn.Linear(300, 300),
                nn.ReLU(),
                nn.Linear(300, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)

        # Using softmax to predict digit
        pred_probab = nn.Softmax(dim=1)(logits)
        y_pred = pred_probab.argmax(1)

        return y_pred



