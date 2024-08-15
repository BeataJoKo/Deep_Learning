# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:22:32 2024

@author: BeButton
"""

"""For data handeling we import the following liberaries and packages"""
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
from torchvision import transforms 
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

#%% 
"""   Create empty dataframe with column names to append data: 
    name of image file, and correspondin label   """
    
df = pd.DataFrame(data=None, columns=['idx', 'path', 'label'])
    
#%% 
"""   loop through the data folder to iterate through image paths
    split filename to get label and checked, if all files are of the same file format"""


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
""" We sort the dataframe through file names and drop unnecessery columns """

df = df.sort_values('idx').reset_index(drop=True)
df.drop(['idx'], axis=1,  inplace=True)

#%%
"""	Ratios for splitting data into train, test and validation datasets. We prioritize 
	a variety in the training data so we split the ratio, In 75% of images for training, 
	15% for validation and 10% for testing"""

train_ratio = 0.75
val_ratio = 0.15
test_ratio = 0.10

#%%
""" Devide data to train, validation and test.
    For the seed, we put it to random_state = 123, 
    so the picures are all in the same place"""

X_train, X_test, y_train, y_test = train_test_split(df['path'], df['label'], test_size=1 - train_ratio, random_state=123)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio/(test_ratio + val_ratio), random_state=123) 

#%% Folder
"""   creat folders for the training , validation and testing datasets, 
    and subfolders that contains the classes pneumonia and normal """ 

folders = ['training', 'validation', 'testing']
sub_folder = ['normal', 'pneumonia']
for folder in folders:
    for sub in sub_folder:
# exist_ok suppresses OS Error, if the directory already exists. If the directory does not exist, a new diectory will be created. More about exist_ok: https://www.geeksforgeeks.org/python-os-makedirs-method/
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
        for i in range(0, len(self.images)):
            img_dir = os.path.join(self.root_dir, self.names[i])
            image_path = os.path.join(img_dir, self.images[i]) 
            image = Image.open(image_path)
            # Applying the transformation 
            if self.transform: 
                image = self.transform(image)  
            self.data.append(image)
  
    # Defining the length of the dataset 
    def __len__(self): 
        return len(self.data) 
  
    # Defining the method to get an item from the dataset 
    def __getitem__(self, index): 
        image = self.data[index]
        label = self.labels[index]
        label = torch.tensor(label)
        
        return image, label
                
#%%
"""   https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/5   
	To avoid a potential false classification based on distortion of the images, we decide to
 	preserve the original ratio and fill the square input by adding padding to the image border"""

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
img_size = (128, 128)

#%%
"""   https://pytorch.org/vision/stable/transforms.html   
	To add more variation into the data, we transform the data. The transformation should
 	simulate naturally occcuring noise. After we look at the data and observe rotation,
  	different perspepectives on the chest, and variation in brightness and contrast,
   	we aim to replicate this noise with our transformation. We also decide to use
    	greyscale, as the original X-ray image is recorded in grayscale"""

std_transform = transforms.Compose([
    transforms.Grayscale(),
    SquarePad(),
    transforms.Resize(img_size),
    transforms.ToTensor(),
])  

aug_transform = transforms.Compose([
    transforms.RandomRotation(degrees=5),
    transforms.ElasticTransform(alpha=15.0, sigma=0.25),
    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
    transforms.Grayscale(),
    SquarePad(),
    transforms.Resize(img_size),
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
        plt.savefig('img/'+title+'_'+str(i)+'_'+data_set.names[i]+'.png')
        plt.show()
    
#%%
os.makedirs('img', exist_ok=True)
os.makedirs('model', exist_ok=True)

#%%
#We choose how many pictures we wanna see from each folder.
showImg(train_dataset, 6, 'Train')
showImg(val_dataset, 6, 'Val')
showImg(train_dataset, 6, 'Train')
showImg(test_dataset, 6, 'Test')

#%%
"""
We use convolutional and pooling to identify small scale features and more complex patterns. 
"""
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Define the layers
        # Convolutional layers
	self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5, stride=1, padding=2)
        # Fully connected layers
        self.fc1 = nn.Linear(40 * 16 * 16, 40)  # Adjust the input size based on image dimensions after pooling
        self.fc2 = nn.Linear(40, 2)  # 2 classes: normal and pneumonia
        # 
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Forward pass through the network
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
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
            loss += model.apply_regularization(layers=[model.fc1], regularization=reg)    
            loss.backward()
            optimizer.step()
            train_loss_list.append(train_loss.item())
        train_loss = sum(train_loss_list) / len(train_loss_list)
        train_losses.append(train_loss)
        #train_losses.append(loss.item())
        accuracy, val_loss = validating(model=model, criterion=criterion)
        accuracy_list.append(accuracy)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{num}, Train Loss: {train_loss}")
    torch.save(model.state_dict(), 'model/'+name+ '_' +reg+'.pth')
    plotting(train_losses, val_losses, epoches, accuracy_list, name)
    return model

def validating(model, criterion):
    model.eval()  
    correct = 0
    total = 0
    val_losses = []
    with torch.no_grad():
        for data, target in val_loader:
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
    plt.savefig('img/'+name+'.png')
    fig

#%%
model = training(SimpleCNN(), 20, 'Convolutional')

