import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
        transform=ToTensor()
)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # Will turn our 28 x 28 MNIST dataset into a 784 x 1 vector
        self.flatten = nn.Flatten()

        # Network has 3 layers and we will be using the ReLU as an activation
        # function
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 300),
                nn.ReLU(),
                nn.Dropout(),             # adding dropout to second layer
                nn.Linear(300, 100),
                nn.ReLU(),
                nn.Dropout(),             # adding dropout to third layer
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, 10),
        )


    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)

        return logits

class MNISTNeural():
    def __init__(self, learning_rate=0, batches=1, epochs=3):
        self.model = NeuralNetwork()
        self.epochs = epochs
        self.batch_size = batches
        self.lr = learning_rate

        self.train_dataloader = DataLoader(training_data, batch_size=batches)
        self.test_dataloader = DataLoader(test_data, batch_size=batches)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_per_epoch = [0] * epochs
        self.acc_per_epoch = [0] * epochs

        
    def train(self): 
        size = len(self.train_dataloader.dataset)
        # Put model in train mode: implements dropout 
        self.model.train()
        loss_amt = 0 

        for batch, (X, y) in enumerate(self.train_dataloader):
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation; zero_grad zeroes out the gradients so that parameters are 
            # updated accordingly, otherwise gradients include old gradients
            self.optimizer.zero_grad()
            loss.backward()                 # Computes gradients
            self.optimizer.step()           # updates parameters

            if batch % 100 == 0:
                loss_amt, current = loss.item(), batch * self.batch_size + len(X) 
                print(f"loss: {loss_amt:>7f} [{current:>5d}/{size:>5d}]")


    def test(self):
        # Puts model in eval mode, does not do dropout
        self.model.eval()

        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        test_loss, correct = 0, 0


        # no_grad disables gradient calculation so we are only testing on the data (no training
        # is happening)
        with torch.no_grad():
            for X, y in self.test_dataloader:
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
        test_loss /= num_batches
        correct /= size

        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        return [test_loss, 100 * correct]


    def run(self):
        accuracy = 0
        for t in range(self.epochs):
            print(f"Epoch {t+1}\n-----------------------------------")
            self.train()
            metrics = self.test()
            self.loss_per_epoch[t] = metrics[0]  
            self.acc_per_epoch[t] = metrics[1]

        print("Finished")
        self.plot_accuracy()

        return [self.loss_per_epoch, self.acc_per_epoch]


    def plot_accuracy(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.epochs + 1), self.acc_per_epoch, marker='o', linestyle='-', color='b')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(f"Model Accuracy by Epoch Learning Rate={self.lr}")
        plt.grid(True)
        plt.savefig(f"acc_plot_{self.lr}.png")

