# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:05:32 2024

@author: BeButton
"""

#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim
from torchvision import datasets, transforms 
import seaborn as sns
import matplotlib.pyplot as plt

#%%
transform = transforms.ToTensor()  
train_dataset = datasets.MNIST(root ='.\Data', train=True, download=True, transform=transform) 
test_dataset = datasets.MNIST(root ='.\Data', train=False, download=True, transform=transform) 

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True) 
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

#%%
input_size = 28 * 28
num_classes = 10

#%%

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        #self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=input_size, out_features=64, bias=True)
        self.fc2 = nn.Linear(in_features=64, out_features=32, bias=True)
        self.fc3 = nn.Linear(in_features=32, out_features=num_classes, bias=True)
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        #x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#%%
model = MyNet()

#%%
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01) # without momentum to show learning curve.

#%%
num = 20
train_losses = []
val_losses = []
epoches = [i for i in range(num)]

#%%
for epoch in range(num):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    train_losses.append(loss.item())
    print(f"Epoch {epoch+1}/{num}, Loss: {loss.item()}")

    model.eval()  
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output.data, dim=1)
            total += target.size(0)
            correct += (pred == target).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    val_losses.append(loss.item())

#%%
torch.save(model.state_dict(), 'spyder-env\deep_learning\Model\mnist.pth')

#%%
fig = plt.figure(figsize=(8, 6))
plt.plot(epoches, train_losses, color='blue') #  label="Train Loss" can be as param
plt.scatter(epoches, val_losses, color='red') # label="Val Loss" also here
plt.legend(['Train Loss', 'Val Loss'], loc='upper right')
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.title('Loss Curve')
#plt.savefig('graph.png')
fig

#%%
image_tensor, label = train_dataset[0]
print(image_tensor.shape, label)

#%%
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_data.shape)

#%%
for i in range(6):
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
    plt.show()

#%%

