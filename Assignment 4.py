import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as AF
import matplotlib.pyplot as plt
import time


from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Training Function
def train_ANN(
    model,
    model_type,
    num_epochs,
    dataloader,
    loss_function,
    optimizer,
):
    time0 = time.perf_counter()

    # use gpu if available
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built()
        else "cpu"
    )
    if torch.cuda.is_available():
        print("The CUDA version is", torch.version.cuda)
        cuda_id = torch.cuda.current_device()
        print("ID of the CUDA device:", cuda_id)
        print("The name of the CUDA device:", torch.cuda.get_device_name(cuda_id))

        print("Training with CUDA")
        model = model.to(device=device)

    elif device.type == "mps" or device.type == "cuda":
        print("Training with Apple Silicon")
        device = torch.device("mps")
        model = model.to(device)
    else:
        print("Training with CPU")

    for e in range(num_epochs):
        running_loss = 0
        for images, labels in dataloader:
            if device.type == "mps" or device.type == "cuda":
                images = images.to(device)
                labels = labels.to(device)

            if model_type == "mlp":
                images = images.view(images.shape[0], -1)

            # set the cumulated gradient to zero
            optimizer.zero_grad()

            # feedforward images as input to the network
            output = model(images)
            loss = loss_function(output, labels)

            # calculating gradients backward using Autograd
            loss.backward()

            # updating all parameters after every iteration through backpropagation
            optimizer.step()

            running_loss += loss.item()
        else:
            print(f"Epoch {e} - Training loss: {running_loss / len(train_dataloader)}")
    print(f"Training Time: {time.perf_counter() - time0}")


def test_ANN(model, model_type, dataloader):
    # use gpu if available
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built()
        else "cpu"
    )
    if torch.cuda.is_available():
        model = model.to(device=device)

    elif device.type == "mps" or device.type == "cuda":
        device = torch.device("mps")
        model = model.to(device)

    # torch.no_grad() is a decorator for the step method
    # making "require_grad" false since no need to keeping track of gradients
    predicted_digits = []
    num_samples = 0
    num_correct = 0
    # torch.no_grad() deactivates Autogra engine (for weight updates)
    with torch.no_grad():
        # set the model in testing mode
        model.eval()
        for batch_cnt, (images, labels) in enumerate(dataloader):
            if model_type == "mlp":
                images = images.reshape(-1, 784)

            if device.type == "mps" or device.type == "cuda":
                images = images.to(device)
                labels = labels.to(device)

            # returns the max value of all elements in the input tensor
            output = model(images)
            _, prediction = torch.max(output, 1)
            predicted_digits.append(prediction)
            num_samples += labels.shape[0]
            num_correct += (prediction == labels).sum().item()
        accuracy = num_correct / num_samples
        print(f"Number of samples: {num_samples}")
        print(f"Number of correct prediction: {num_correct}")
        print(f"Accuracy: {accuracy}")


mini_batch_size = 1000

transformer = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transformer, download=True
)
test_dataset = datasets.MNIST(
    root="./data", train=False, transform=transformer, download=False
)

train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=mini_batch_size, shuffle=True
)
test_dataloader = DataLoader(
    dataset=test_dataset, batch_size=mini_batch_size, shuffle=True
)

print(f"Mini batch size: {mini_batch_size}")
print(f"Number of batches loaded for training: {len(train_dataloader)}")
print(f"Number of batches loaded for testing: {len(test_dataloader)}")

print("-" * 80)
print("Modeling with MLP")
print("-" * 80)

model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
    nn.LogSoftmax(dim=1),
)
print(model)

train_ANN(
    model,
    "mlp",
    15,
    dataloader=train_dataloader,
    loss_function=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=0.003),
)

test_ANN(model, "mlp", dataloader=test_dataloader)

print("-" * 80)
print("Modeling with CNN")
print("-" * 80)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 28, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout_conv1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128, 10)
        self.hidden = nn.Linear(28 * 13 * 13, 128)
        self.out = nn.ReLU()

    def forward(self, x):
        x = self.out(self.conv1(x))
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.out(self.hidden(x))
        x = self.dropout_conv1(x)
        x = self.fc1(x)
        return x


model = CNN()
print(model)

train_ANN(
    model,
    "cnn",
    15,
    dataloader=train_dataloader,
    loss_function=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=0.003),
)

test_ANN(model, "cnn", dataloader=test_dataloader)
