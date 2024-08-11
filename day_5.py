# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:11:13 2024

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

#%% Folder
import os
if not os.path.exists('Img'):
    os.makedirs('Img')
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
print('Train: ', train_dataset.data.shape)
print('Test: ', test_dataset.data.shape)
print('Image: ', test_dataset.data[0].shape)

#%%
num_count = {}
for image_tensor, label in train_dataset:
    if label in num_count.keys():
        num_count[label] += 1
    else:
        num_count[label] = 1
        
#%%
plt.bar(*zip(*num_count.items()))
plt.xticks(list(num_count.keys()))
plt.title('Train Labels')
plt.xlabel('Number')
plt.ylabel('Count')
plt.show()

#%%



#%%
image_tensor, label = test_dataset[0]
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
input_size = 28 * 28
num_classes = 10

#%% Model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3)    # Output size   :([64, 64, 26, 26])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.fc2 = nn.Linear(in_features=10*13*13, out_features=64, bias=True)
        self.fc3 = nn.Linear(in_features=64, out_features=num_classes, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = x.view(x.size(0), -1)
        #x = x.flatten(start_dim=1)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
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
    
#%% Functions
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
model = training(ConvNet(), 20, 'Convolutional')

#%%
def feature_map(cl_model, row_size, col_size):
    img_batch = next(iter(test_loader))[0]
    conv1_output = cl_model.conv1(img_batch[0])
    layer_visualization = conv1_output.data
    
    for i, feature_map in enumerate(layer_visualization):
        plt.subplot(row_size, col_size, i+1)
        plt.imshow(feature_map.numpy(), cmap='gray')
        plt.axis('off')
    plt.subplots_adjust(hspace=-0.5, wspace=0.1)
    #plt.tight_layout()
    plt.savefig('Img/feature_maps.png')

#%%
feature_map(model, 2, 5)







