# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:22:32 2024

@author: BeButton
"""

#%%
import time
import random
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image 
import torch
import torch.nn as nn 
import torch.optim as optim
from torchvision import datasets, transforms 
from sklearn.model_selection import train_test_split

#%% 
"""   Create empty dataframe with column names to append data: 
    name of image file, and correspondin label   """
    
df = pd.DataFrame(data=None, columns=['idx', 'path', 'label'])
    
#%% 
"""   loop through the data folder to iterate through image paths
    split filename to get label and check 
    if all files are of the same file format  """

path = 'data'
extension = []
for filename in os.listdir(path):
    row = {'idx': int(filename.split('_', 1)[0]),'path': filename, 'label': filename.split('_', 1)[1].split('.', 1)[0]}
    row = pd.DataFrame(row, index=[0])
    df = pd.concat([df, row]).reset_index(drop=True)    
    ext = filename.split('.', 1)[1]
    if ext not in extension:
        print(filename.split('.', 1)[1])
        extension.append(ext)
        print(filename)
        
#%%
"""   sort dataframe through file name and drop unnecessery column, delete Einstein joke image before dropping id?   """

df = df.sort_values('idx').reset_index(drop=True)
df.drop(['idx'], axis=1,  inplace=True)

#%%
"""   ratios for shering data to train, test and validation datasets   """

train_ratio = 0.75
val_ratio = 0.15
test_ratio = 0.10

#%%
"""   devide data to train, validation and test   """

X_train, X_test, y_train, y_test = train_test_split(df['path'], df['label'], test_size=1 - train_ratio, random_state=123)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + val_ratio), random_state=123) 

#%% Folder
"""   creat folders and subfolders   """ 

folders = ['training', 'validation', 'testing']
sub_folder = ['normal', 'pneumonia']
for folder in folders:
    for sub in sub_folder:
        #exist_ok suppresses OSError if the directory already exists. If the directory doesn’t exist, it will be created. more about exist_ok: https://www.geeksforgeeks.org/python-os-makedirs-method/
        os.makedirs('data/'+folder+'/'+sub,exist_ok=True)
#%%
def moveImg(X, parent):
    for e in X:
        if 'normal' in e:
            os.rename('data/'+e, 'data/'+parent+'/normal/'+e)
        else:
            os.rename('data/'+e, 'data/'+parent+'/pneumonia/'+e)         
  
#%%
moveImg(X_train, 'training')
moveImg(X_val, 'validation')
moveImg(X_test, 'testing')

#%%
class ImgDataset(Dataset): 
    # Inicialization
    def __init__(self, df_x, df_y, root_dir=None, transform=None): 
        self.root_dir = root_dir
        self.images = list(df_x) 
        self.names = list(df_y)
        self.transform = transform 
        self.labels = [1 if x == 'pneumonia' else 0 for x in self.names]
        self.data = []
        
        if root_dir:
            self.readImages()
        
    def readImages(self):
        for i in range(0, len(self.images) - 1):
            img_dir = os.path.join(self.root_dir, self.names[i])
            image_path = os.path.join(img_dir, self.images[i]) 
            image = Image.open(image_path)
            # Applying the transformation 
            if self.transform: 
                image = self.transform(image)  
            self.data.append(image)
  
    # Defining the length of the dataset 
    def __len__(self): 
        return len(self.images) 
  
    # Defining the method to get an item from the dataset 
    def __getitem__(self, index): 
        image = self.data[index]
        label = torch.tensor(self.labels[index])
        return image, label
                
#%%
class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return transforms.functional.pad(image, padding, 0, 'constant')

#%%
batch_size = 32
std_transform = transforms.Compose([
    transforms.Grayscale(),
    SquarePad(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((256, 256)),
    transforms.ToTensor(),
])  

aug_transform = transforms.Compose([
    transforms.RandomRotation(degrees=5),
    transforms.ElasticTransform(alpha=15.0, sigma=0.25),
    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
    transforms.Grayscale(),
    SquarePad(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((256, 256)),
    transforms.ToTensor(),
])  

train_dataset = ImgDataset(X_train, y_train, './data/training', aug_transform)
val_dataset = ImgDataset(X_val, y_val, './data/validation', std_transform)
test_dataset = ImgDataset(X_test, y_test, './data/testing', std_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False) 
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#%%
print('Test: ', len(test_dataset))
print('Image_0_name: ', test_dataset.names[0])
print('Image_0_size: ', test_dataset.data[0].size(1), 'x', test_dataset.data[0].size(2))
print('Image_0_label: ', test_dataset.labels[0])
print('Object_0: ', test_dataset[0])

#%%
image_tensor, label = train_dataset[0]
print(image_tensor.shape, label)

#%%
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_data.shape)

#%%
def showImg(data_set, num, title):
    for i in random.choices(range(0, len(data_set)-1), k=num):
        plt.imshow(data_set.data[i][0], cmap='gray', interpolation='none')
        plt.title("{} - Truth: {}".format(title, data_set.names[i]))
        plt.xticks([])
        plt.yticks([])
        plt.show()
    
#%%
showImg(train_dataset, 6, 'Train')
showImg(val_dataset, 6, 'Val')
showImg(train_dataset, 6, 'Train')
showImg(test_dataset, 6, 'Test')

plt.xticks([])
plt.yticks([])
plt.show()


#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Define the layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 128 * 128, 512)  # Adjust the input size based on image dimensions after pooling
        self.fc2 = nn.Linear(512, 2)  # 2 classes: normal and pneumonia

    def forward(self, x):
        # Forward pass through the network
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x





#%%
import torch.optim as optim

# Initialize the model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()  # Appropriate for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, val_loader, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
        # Validate
        validate_model(model, val_loader)

def validate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')
