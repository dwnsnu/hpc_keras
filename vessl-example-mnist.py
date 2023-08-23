import datetime
import os

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import vessl


# Define model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d(0.25)
        self.drop2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.LogSoftmax(1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop2(x)
        x = self.fc2(x)
        return self.softmax(x)
    

# Train function with VESSL logging
def train(model, device, train_loader, optimizer, epoch, start_epoch):
    model.train()
    loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, label)
        loss.backward()
        optimizer.step()
        if batch_idx % 128 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch + 1, len(train_loader.dataset), len(train_loader.dataset), 100, loss.item()))

    # Logging loss metrics to Vessl
    vessl.log(
        step=epoch + start_epoch + 1,
        payload={'loss': loss.item()}
    )
    
    
# Test function (with vessl plot uploading)
def test(model, device, test_loader, save_image):
    model.eval()
    test_loss = 0
    correct = 0
    test_images = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_images.append(vessl.Image(
                data[0], caption="Pred: {} Truth: {}".format(pred[0].item(), target[0])))

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_accuracy))

    # Saving inference results with caption into Vessl
    if save_image:
        vessl.log({"Examples": test_images})

    return test_accuracy

def main():
    # Download the MNIST training data
    train_data = datasets.MNIST(
        root = '/tmp/example-data',
        train = True,                         
        transform = ToTensor(), 
        download = True,            
    )
    test_data = datasets.MNIST(
        root = '/tmp/example-data', 
        train = False, 
        transform = ToTensor()
    )

    # Downsize the dataset to run fast.
    train_data.data = train_data.data[:200]
    test_data.data = test_data.data[:1000]
    print(f'The shape of train data: {train_data.data.shape}')
    print(f'The shape of test data: {test_data.data.shape}')

    # Define data loader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, num_workers=1)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    print(model)

    # Initialize new experiment via VESSL SDK 
    vessl.init()

    # Hyperparameters
    epochs = 5
    batch_size = 128
    learning_rate = 0.01
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    # Train the model
    for epoch in range(epochs):
        train(model, device, train_loader, optimizer, epoch, 0)
        test(model, device, test_loader, True)
        scheduler.step()

    # Finish experiment
    vessl.finish()

if __name__ == "__main__":
    if os.environ.get("VESSL_EXPERIMENT_ID", None) is None:
        print("Please run with `vessl run python vessl-example-mnist.py`")
    else:
        print("\n")
        print("---- This process is running on the remote cluster you selected. ----")
        print("---- ^C will not abort this remote experiment process. ----")
        print("\n")
        main()
