# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:22:32 2024

@author: BeButton
"""

#%%
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
"""   sort dataframe through file name and drop unnecessery column   """

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
        if not os.path.exists('data/'+folder):
            os.makedirs('data/'+folder)
            os.makedirs('data/'+folder+'/'+sub)
            print(folder.title()+' subfolder '+sub.title()+' created!')                
        else:
            print("The "+folder+" allready exists.")
            if not os.path.exists('data/'+folder+'/'+sub):
                os.makedirs('data/'+folder+'/'+sub)
            else:
                print('The '+folder.title()+' subfolder '+sub.title()+' allready exists.')
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
class ImgDataset(torch.utils.data.Dataset): 
    def __init__(self, root_dir, data, labels, transform=None): 
        self.root_dir = root_dir
        self.images = data 
        self.labels = labels
        self.transform = transform 
  
    # Defining the length of the dataset 
    def __len__(self): 
        return len(self.images) 
  
    # Defining the method to get an item from the dataset 
    def __getitem__(self, index): 
        img_dir = os.path.join(self.root_dir, self.labels[index])
        image_path = os.path.join(img_dir, self.images[index]) 
        image = np.array(Image.open(image_path)) 
        label = self.labels[index]
  
        # Applying the transform 
        if self.transform: 
            image = self.transform(image) 
          
        return image, label

#%%
batch_size = 32
transform = transforms.ToTensor()  
train_dataset = ImgDataset('./data/training', X_train, y_train, transform)
val_dataset = ImgDataset('./data/validation', X_val, y_val, transform)
test_dataset = ImgDataset('./data/testing', X_test, y_test, transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False) 
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


#%%
print('Train: ', len(train_dataset))
print('Image_0_name: ', train_dataset.images[0])
print('Image_0_size: ', train_dataset[0][0].shape)
print('Label_0: ', train_dataset.labels[0])
print('Image_0_label: ', train_dataset[0][1])

#%%







#%%

        
#%%


    
#%%

        
        
        
        
        
        
        
