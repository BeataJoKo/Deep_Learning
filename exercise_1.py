import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=False)




class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features =20)
        self.fc2 = nn.Linear(in_features=20, out_features =20)
        self.fc3 = nn.Linear(in_features=20, out_features =20)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.flatten = nn.Flatten()
    
        


    def forward(self, x):
        x = self.flatten(x)
    
        #relu is also the activation layer
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        #softmax is the final activation layer
        x = self.softmax(self.fc3(x))
        return x
net = MyNetwork()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)
num_epochs = 20
losses = []
for epoch in range(num_epochs):
    net.train()
    for data, targets in train_loader:
        optimizer.zero_grad()
        outputs = net(data)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
    losses.append(loss.detach())

PATH = "model.pth" 
#til træning fjernes alt under torch.save og torch.save gøre synlig. der efter puttes alt kode tilbage og torch save 
#torch.save(net.state_dict(), PATH)

net.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, targets in test_loader:
        outputs = net(data)
        _, predicted = torch.max(outputs.detach(),dim=1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
print(accuracy)

#print(train_dataset.data.shape)
    



net1 = MyNetwork()
net1.load_state_dict(torch.load(PATH))
net1.eval()


epochs = list(range(num_epochs))
plt.plot(epochs, losses, "bo", label = "Training loss")
plt.title("training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()

