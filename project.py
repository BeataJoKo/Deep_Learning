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
batch_size = 32
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])  

train_dataset = ImgDataset(X_train, y_train, './data/training', transform)
val_dataset = ImgDataset(X_val, y_val, './data/validation', transform)
test_dataset = ImgDataset(X_test, y_test, './data/testing', transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False) 
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


#%%
print('Test: ', len(test_dataset))
print('Image_0_name: ', test_dataset.names[0])
print('Image_0_size: ', test_dataset.data[0].shape)
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
for i in range(6):
    plt.imshow(train_dataset.data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(train_dataset.names[i]))
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
#%%
random_idx = random.randint(0, len(train_dataset) - 1) 

plt.imshow(train_dataset.data[random_idx][0], cmap='gray', interpolation='none') 
plt.xticks([])
plt.yticks([])
plt.show()
