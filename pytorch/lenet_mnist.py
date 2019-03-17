import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt

train_ds = datasets.mnist.MNIST('data', train=True, transform=transforms.ToTensor(), download=True)
test_ds = datasets.mnist.MNIST('data', train=False, transform=transforms.ToTensor(), download=True)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=True)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1)
        self.avg1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.avg2 = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.avg1(x)
        x = self.conv2(x)
        x = self.avg2(x)
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


model = ConvNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
epochs = 5
train_ds_size = len(train_ds)
valid_ds_size = len(test_ds)

for i in range(epochs):
    print(f'Epoch: {i}/{epochs-1}')
    print('-' * 10)
    for phase in ['train', 'valid']:
        if phase == 'train':
            model.train()
            dl = train_dl
            ds_size = train_ds_size
        else:
            model.eval()
            dl = test_dl
            ds_size = valid_ds_size
        running_loss = 0.0
        running_correct = 0
        for input, target in dl:
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                output = model(input)
                _, preds = torch.max(output, 1)
                loss = loss_fn(output, target)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * input.size()[0]
            running_correct += torch.sum(preds == target.data)
    
        epoch_loss = round(running_loss / ds_size, 3)
        epoch_accuracy = round(running_correct.double().numpy() / ds_size, 3)

        print(f'{phase}: loss: {epoch_loss}, acc: {epoch_accuracy}')