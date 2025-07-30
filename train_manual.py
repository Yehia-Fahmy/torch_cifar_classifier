# imports
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# setup
# select device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
# load and prepare dataset
batch_size  = 64
num_workers = 4
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size, True, num_workers=num_workers)

testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size, True, num_workers=num_workers)

classes = trainset.classes
print(f"Number of classes: {len(classes)}")
print(classes)
print(f"Number of training examples: {len(trainset)}")
print(f"Number of testing examples: {len(testset)}")

# define network class
class CifarClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 16, 3)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        # exit()
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

network = CifarClassifier().to(device)

# specify loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters())

# define training function
num_epochs = 10
for epoch in range(num_epochs):
    # run one pass through the data
    network.train()
    running_loss = 0.0
    for batch_id, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        y_hat = network(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
    print(f"Training Loss for epoch {epoch+1}: {running_loss / len(train_loader)}")

    network.eval()
    running_loss = 0.0
    for batch_id, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        y_hat = network(x)
        running_loss += criterion(y_hat, y).item()  # Fixed variable name and order
    print(f"Testing Loss for epoch {epoch+1}: {running_loss / len(test_loader)}")

