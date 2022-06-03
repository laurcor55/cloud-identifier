import random
from lib import MlpNet
import lib
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt

batch_size = 100
validation_split = .2

random_seed = 2
data_path = 'CCSN'
transform = transforms.Compose([transforms.Resize(400), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.Grayscale()])#, lib.MyCropTransform([70, 0, 400, 330])])

train_set = datasets.ImageFolder(data_path, transform=transform)

# Create validation split by taking a percentage of the training set:
train_set_size = len(train_set)
indices = list(range(train_set_size))
split = int(np.floor(validation_split * train_set_size))
np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
labels_reference = list(train_set.class_to_idx.keys())

# Create dataloader objects - will be used during training and inference
# to iterate over the data.
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=val_sampler)

lib.data_preview(train_set)

# Create a neural network object with the specified number of input neurons,
# hidden neurons, and output neurons (or total classes)
unique_classes_count = len(labels_reference)
input_neurons_count = 400*400
hidden_neurons_count = 16
net = MlpNet(input_neurons_count, hidden_neurons_count, unique_classes_count)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

num_epochs = 50
loss_tracker = []

for epoch in range(num_epochs):
  # Initialize random value for loss just for displaying purposes
  loss = torch.tensor([batch_size])
  i = 0
  for images, labels in train_loader:
    images = Variable(images.view(-1, input_neurons_count))
    labels = Variable(labels)

    optimizer.zero_grad()
    outputs = net(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    loss_tracker.append(loss.data)                                  

    if (i+1) % batch_size == 0 or (i+1) == len(train_loader):   
      print('Epoch [%d/%d],  Val Acc: %d, Training Loss: %.4f'
              %(epoch+1, num_epochs, \
                lib.get_accuracy(val_loader, net), loss.data))
    i += 1

train_acc = lib.get_accuracy(train_loader, net)
test_acc = lib.get_accuracy(val_loader, net)

print('Accuracy of the network on train images: %d %%' % (train_acc))
print('Accuracy of the network on val images: %d %%' % (test_acc))

# Plot the loss over time
plt.plot(loss_tracker)
plt.ylabel("Loss")
plt.xlabel("Step Number")
plt.title("Loss over time")
plt.show()