import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
)

test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        tranform=ToTensor()
)

class NeuralNetwork(nn.Module):
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

        return logits

        """
        # Using softmax to predict digit
        pred_probab = nn.Softmax(dim=1)(logits)
        y_pred = pred_probab.argmax(1)

        return y_pred
        """


class MNISTNeural():
    def __init__(self, learning_rate=0, batches=1, epochs=10):
        self.model = NeuralNetwork()
        self.epochs = epochs
        self.batch_size = batches

        self.train_dataloader = DataLoader(training_data, batch_size=batches)
        self.test_dataloader = DataLoader(test_data, batch_size=batches)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
    def train(self): 
        size = len(self.train_dataloader.dataset)
        self.model.train()

        for batch, (X, y) in enumerate(self.train_dataloader):
            pred = self.model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * self.batch_size + len(X) 
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


    def test(self):
        self.model.eval()

        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        test_loss, correct = 0, 0


        with torch.no_grad():
            for X, y in self.test_dataloader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
        test_loss /= num_batches
        correct /= size

        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


    def run(self):
        for t in range(self.epochs):
            print(f"Epoch {t+1}\n-----------------------------------")
            self.train()
            self.test()

        print("Finished")

