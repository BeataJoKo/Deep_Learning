# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:22:32 2024

@author: Group 17
    Anas Byriel Othman
    Beata Joanna Morawska
    Ejvind Heebøll Brandt
    Jakob Kolberg
    Melanie Wittrup Berg
    Tew Rhui Ren
""" 
# We have went through a highly iterative process. It's really hard for us to describe/pinpoint who worked on what. 
# Please assume we worked equally on all parts.


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
	a variety in the training data so we split the ratio, In 75% of images for training since we need more testing, 
 	compared to validation and testing, 15% for validation and 10% for testing"""

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
"""The images gets moved to the appropriate folders"""

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

""" By using Pytorch's Datasets, the image processing, loading and applied transformations gets handled, 
and returnes the images with the corresponding labels"""

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
	class Squarepad makes sure that the images are padded to square dimentions before the resizing.	
	To avoid a potential false classification based on distortion of the images, we decide to
 	preserve the original ratio and fill the square input by adding padding to the image border."""

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

""" 
Different transformations. The parameters are not particularily finely tuned, we tried tweaking them slightly. 
Our assumptioon is that the RandomRotation fairly important. The others, like ElasticTransform, RandomAffine and ColorJitter
are mostly experimental. They could probably also be used (we did not do this) to produce more data/images and use those in 
addition to the original images as part of training process. 
"""
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
showImg(train_dataset, 2, 'Train')
showImg(val_dataset, 2, 'Val')
showImg(train_dataset, 2, 'Train')
showImg(test_dataset, 2, 'Test')

#%%
"""
We use convolutional and pooling to identify small scale features and more complex patterns. 
"""
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
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
	    
    """
    Regularization, where L1 is Lasso and L2 is Ridge. 
    L1 forces weights closer to zero and can with a high enough lambda/punishing term. L2 on the other hand distrubutes 
    the weights more evenly. Here different hyperparameters were tested and made our model inaccurate. In our we had the 
    best results with L2. The training class assumes no regularization if none are specified in the function call. 
    """
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
    criterion = nn.CrossEntropyLoss() # Appropriate for non-binary classification 
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
model = training(SimpleCNN(), 40, 'Convolutional_15conv_40e_0.9mom', reg='L2', mom=0.9) 

#%%
def save_filters(cl_model, r, filter_file):
    #plt.figure(figsize=(10, 6))
    for i, filter in enumerate(cl_model):
        plt.subplot(r, 5, i + 1)
        plt.imshow(filter[0, :, :].cpu().detach(), cmap='gray')
        plt.axis('off')
        #plt.subplots_adjust(hspace=-0.2, wspace=0.1)
        #plt.tight_layout()
        plt.savefig('img/'+filter_file)

#%%
save_filters(model.conv1.weight, 1, 'Convolutional_conv1_Tr.png')
save_filters(model.conv2.weight, 3, 'Convolutional_conv2_Tr.png')
save_filters(model.conv3.weight, 6, 'Convolutional_conv3_Tr.png')
save_filters(model.conv4.weight, 12, 'Convolutional_conv4_Tr.png')

#%%
"""   https://www.geeksforgeeks.org/visualizing-feature-maps-using-pytorch/   """

total_conv_layers = 0
conv_weights = []
conv_layers = []

for module in model.children():
    if isinstance(module, nn.Conv2d):
        total_conv_layers += 1
        conv_weights.append(module.weight)
        conv_layers.append(module)

print(f"Total convolution layers: {total_conv_layers}")

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_model = model.to(device)
 
# Preprocess the image and move it to GPU
img_batch = next(iter(val_loader))[0]
input_image = img_batch[0].unsqueeze(0)  # Add a batch dimension
input_image = input_image.to(device)

feature_maps = []  # List to store feature maps
layer_names = []  # List to store layer names
for layer in conv_layers:
    input_image = layer(input_image)
    feature_maps.append(input_image)
    layer_names.append(str(layer))

#%%
# Process and visualize feature maps
processed_feature_maps = []  # List to store processed feature maps
for feature_map in feature_maps:
    feature_map = feature_map.squeeze(0)  # Remove the batch dimension
    mean_feature_map = torch.sum(feature_map, 0) / feature_map.shape[0]  # Compute mean across channels
    processed_feature_maps.append(mean_feature_map.data.cpu().numpy())

#%%
# Plot the feature maps
fig = plt.figure(figsize=(30, 10))
for i in range(len(processed_feature_maps)):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(processed_feature_maps[i], cmap='gray')
    ax.axis("off")
    plt.tight_layout()
    ax.set_title(layer_names[i].split('(')[0], fontsize=30)
plt.savefig('img/Convolutional_Layers_avg.png')

#%%
count = 0
fig = plt.figure(figsize=(60, 120))
for i in range(0, len(feature_maps)):
    for j in range(0, len(feature_maps[i][0])):
        count += 1
        ax = fig.add_subplot(14, 8, count)
        ax.imshow(feature_maps[i][0][j].detach().numpy(), cmap='gray')
        ax.axis("off")
        plt.tight_layout()
        print(i, j)
plt.savefig('img/Convolutional_Layers_all.png')

#%%

