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

#%%
import os
if not os.path.exists('Model'):
    os.makedirs('Model')
    print('Folder created!')
else:
    print("The folder allready exists.")

#%% Data
transform = transforms.ToTensor()  
train_dataset = datasets.MNIST(root ='.\Data', train=True, download=True, transform=transform) 
test_dataset = datasets.MNIST(root ='.\Data', train=False, download=True, transform=transform) 

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
        #self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def apply_regularization(self, layers, regularization):
        regularization_loss = 0  
        for layer in layers:
            if hasattr(layer, 'weight'):
                if regularization == "L1":
                    p = 1
                    L_lambda = 0.001
                elif regularization == "L2":
                    p = 2
                    L_lambda = 0.005
                else:
                    return 0
                L_reg_layer = L_lambda * torch.norm(layer.weight, p=p)
                regularization_loss += L_reg_layer
            return regularization_loss
    
    
class DeepNet(nn.Module):
    def __init__(self):
        super(DeepNet, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=64, bias=True)
        self.fc2 = nn.Linear(in_features=64, out_features=32, bias=True)
        self.fc3 = nn.Linear(in_features=32, out_features=28, bias=True)
        self.fc4 = nn.Linear(in_features=28, out_features=16, bias=True)
        self.fc5 = nn.Linear(in_features=16, out_features=num_classes, bias=True)
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
    def apply_regularization(self, layers, regularization):
        regularization_loss = 0  
        for layer in layers:
            if hasattr(layer, 'weight'):
                if regularization == "L1":
                    p = 1
                    L_lambda = 0.001
                elif regularization == "L2":
                    p = 2
                    L_lambda = 0.005
                else:
                    return 0
                L_reg_layer = L_lambda * torch.norm(layer.weight, p=p)
                regularization_loss += L_reg_layer
            return regularization_loss
    
    
class WideNet(nn.Module):
    def __init__(self):
        super(WideNet, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=392, bias=True)
        self.fc2 = nn.Linear(in_features=392, out_features=32, bias=True)
        self.fc3 = nn.Linear(in_features=32, out_features=num_classes, bias=True)
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def apply_regularization(self, layers, regularization):
        regularization_loss = 0  
        for layer in layers:
            if hasattr(layer, 'weight'):
                if regularization == "L1":
                    p = 1
                    L_lambda = 0.001
                elif regularization == "L2":
                    p = 2
                    L_lambda = 0.005
                else:
                    return 0
                L_reg_layer = L_lambda * torch.norm(layer.weight, p=p)
                regularization_loss += L_reg_layer
            return regularization_loss
    
    
class OutNet(nn.Module):
    def __init__(self):
        super(OutNet, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=64, bias=True)
        self.fc2 = nn.Linear(in_features=64, out_features=32, bias=True)
        self.fc3 = nn.Linear(in_features=32, out_features=num_classes, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        #self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def apply_regularization(self, layers, regularization):
        regularization_loss = 0  
        for layer in layers:
            if hasattr(layer, 'weight'):
                if regularization == "L1":
                    p = 1
                    L_lambda = 0.001
                elif regularization == "L2":
                    p = 2
                    L_lambda = 0.005
                else:
                    return 0
                L_reg_layer = L_lambda * torch.norm(layer.weight, p=p)
                regularization_loss += L_reg_layer
            return regularization_loss
    

#%%
def training(cl_model, num, name, reg='None', mom=0.00):
    model = cl_model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=mom)
    train_losses = []
    val_losses = []
    accuracy_list = []
    epoches = [i for i in range(num)]
    
    for epoch in range(num):
        model.train()
        train_loss_list = []
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            train_loss = loss
            loss += model.apply_regularization(layers=[model.fc2], regularization=reg)    
            loss.backward()
            optimizer.step()
            train_loss_list.append(train_loss.item())
        train_loss = sum(train_loss_list) / len(train_loss_list)
        train_losses.append(train_loss)
        #train_losses.append(loss.item())
        accuracy, val_loss = test(model=model, criterion=criterion)
        accuracy_list.append(accuracy)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{num}, Train Loss: {train_loss}")
    torch.save(model.state_dict(), 'Model/'+name+ '_' +reg+'.pth')
    plotting(train_losses, val_losses, epoches, accuracy_list, name)
    return model

def test(model, criterion):
    model.eval()  
    correct = 0
    total = 0
    val_losses = []
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            loss = criterion(outputs, target)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            val_losses.append(loss.item())
    val_loss = sum(val_losses) / len(val_losses)
    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")
    return accuracy, val_loss

def plotting(train_losses, val_losses, epoches, accus, name):
    fig = plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epoches, train_losses, color='blue') #  label="Train Loss" can be as param
    plt.scatter(epoches, val_losses, color='red') # label="Val Loss" also here
    plt.legend(['Train Loss', 'Val Loss'], loc='upper right')
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.title('Loss Curve: ' +name)
    
    plt.subplot(1, 2, 2)
    plt.plot(accus, label='Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('Model/'+name+'.png')
    fig

#%%
training(MyNet(), 20, 'Original')

#%% Regularization
training(MyNet(), 20, 'L_2', reg='L2')
training(MyNet(), 20, 'L_1', reg='L1')

#%% Dropout
training(OutNet(), 20, 'Dropout')

#%% Layers
training(WideNet(), 20, 'Wide')
training(DeepNet(), 20, 'Deep')

#%% Momentum
training(MyNet(), 20, 'Mom_1', mom=0.1)
training(MyNet(), 20, 'Mom_5', mom=0.5)
training(MyNet(), 20, 'Mom_9', mom=0.9)

#%%

