# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:14:56 2024

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

#%% Data
transform = transforms.ToTensor()  
train_dataset = datasets.MNIST(root ='.\spyder-env\deep_learning\Data', train=True, download=True, transform=transform) 
test_dataset = datasets.MNIST(root ='.\spyder-env\deep_learning\Data', train=False, download=True, transform=transform) 

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True) 
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

#%%
input_size = 28 * 28
num_classes = 10

#%% Model

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=64, bias=True)
        self.fc2 = nn.Linear(in_features=64, out_features=32, bias=True)
        self.fc3 = nn.Linear(in_features=32, out_features=num_classes, bias=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x
    
class DeepNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=64, bias=True)
        self.fc2 = nn.Linear(in_features=64, out_features=32, bias=True)
        self.fc3 = nn.Linear(in_features=32, out_features=28, bias=True)
        self.fc4 = nn.Linear(in_features=28, out_features=16, bias=True)
        self.fc5 = nn.Linear(in_features=16, out_features=num_classes, bias=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.softmax(self.fc5(x))
        return x
    
class WideNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=392, bias=True)
        self.fc2 = nn.Linear(in_features=392, out_features=32, bias=True)
        self.fc3 = nn.Linear(in_features=32, out_features=num_classes, bias=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x
    
class OutNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(in_features=input_size, out_features=64, bias=True)
        self.fc2 = nn.Linear(in_features=64, out_features=32, bias=True)
        self.fc3 = nn.Linear(in_features=32, out_features=num_classes, bias=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

#%%
def plotting(train_losses, val_losses, epoches, name):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(epoches, train_losses, color='blue') #  label="Train Loss" can be as param
    plt.scatter(epoches, val_losses, color='red') # label="Val Loss" also here
    plt.legend(['Train Loss', 'Val Loss'], loc='upper right')
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.title('Loss Curve: ' +name)
    fig

#%%
def training(cl_model, e_num, name, mom=0.0):
    model = cl_model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=mom)
    num = e_num
    train_losses = []
    val_losses = []
    epoches = [i for i in range(num)]
    for epoch in range(num):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            lambda1, lambda2 = 0.5, 0.01
            if name == 'l2':
                loss += lambda2 * torch.norm(model.fc2.weight, p=2)
            if name == 'l1':
                loss += lambda1 * torch.norm(model.fc2.weight, p=1)
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
                if name == 'l2':
                    loss += lambda2 * torch.norm(model.fc2.weight, p=2)
                if name == 'l1':
                    loss += lambda1 * torch.norm(model.fc2.weight, p=2)
                _, pred = torch.max(output.data, dim=1)
                total += target.size(0)
                correct += (pred == target).sum().item()
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        val_losses.append(loss.item())
    plotting(train_losses, val_losses, epoches, name)
    return model

#%%
training(MyNet(), 20, 'original')

#%% Regularization
training(MyNet(), 20, 'l2')
training(MyNet(), 20, 'l1')

#%% Dropout
training(OutNet(), 20, 'dropout')
training(DeepNet(), 20, 'deep')

#%% Layers
training(WideNet(), 20, 'wide')

#%% Momentum
training(MyNet(), 20, 'mom1', 0.1)
training(MyNet(), 20, 'mom5', 0.5)
training(MyNet(), 20, 'mom9', 0.9)
