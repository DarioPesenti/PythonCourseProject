import os
import torch
from torch import nn
import torch.nn.functional as F
import torchvision


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device")

class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Update in_channels to match the out_channels of conv1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Update the input size for fc1 based on the output dimensions of conv2
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x, return_embedding=False):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # Extract and return embeddings after the second convolutional layer
        if return_embedding:
            # Flatten the output from the convolutional layers
            embedding = x.view(-1, 64 * 16 * 16)
            return embedding
        # Continue with the rest of the network if not returning embedding
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MyNeuralNetwork().to(device)

embedding_classifier = nn.Linear(16384, 10).to(device) #used for convexity analyses

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(model.parameters()) + list(embedding_classifier.parameters()), lr=0.001)