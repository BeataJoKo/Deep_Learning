# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 20:34:38 2024

@author: Group 17
    Anas Byriel Othman
    Beata Joanna Morawska
    Ejvind Heebøll Brandt
    Jakob Kolberg
    Melanie Wittrup Berg
    Tew Rhui Ren
"""

#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn 

#%%

input_dim = (128, 128)
channel_dim = 1

class group_17(nn.Module):
    def __init__(self):
        super(group_17, self).__init__()
        # Define the layers
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=15, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=15, out_channels=30, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=30, out_channels=60, kernel_size=3, stride=1, padding=1)
        # Fully connected layers
        self.fc1 = nn.Linear(60 * 8 * 8, 100)  # Adjust the input size based on image dimensions after pooling
        self.fc2 = nn.Linear(100, 2)  # 2 classes: normal and pneumonia
        # 
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Forward pass through the network
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
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

