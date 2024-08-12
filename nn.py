import torch.nn as nn
import torch.nn.functional as F


class MNISTNeural(nn.Module):
    def __init__(self):
        super().__init__()

        # Will turn our 28 x 28 MNIST dataset into a 784 x 1 vector
        self.flatten = nn.Flatten()

        # Network has 3 layers and we will be using the ReLU as an activation
        # function
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 300),
                nn.ReLU(),
                nn.Linear(300, 100),
                nn.ReLU(),
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, 10),
        )

    def forward(self, x):
        # Might not need to flatten data depending on how we are ingesting each picture
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)

        # Using softmax to predict digit
        pred_probab = nn.Softmax(dim=1)(logits)
        y_pred = pred_probab.argmax(1)

        return y_pred



