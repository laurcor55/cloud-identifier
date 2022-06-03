from torch import nn
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
import numpy as np

import os
import random
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

import tqdm
import math
import time

class CnnNet(nn.Module):
  def __init__(self, num_classes):
    super(CnnNet, self).__init__()
    size_1 = 96
    size_2 = 256
    size_3 = 384
    kernel_1 = 11
    kernel_2 = 5
    kernel_3 = 3
    size_linear =  9216
    self.cnn_layers = nn.Sequential(
      nn.Conv2d(3, size_1, kernel_size=kernel_1, stride=4, padding=0),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2),
      
      nn.Conv2d(size_1, size_2, kernel_size=kernel_2, stride=1, padding=2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2),

      nn.Conv2d(size_2, size_3, kernel_size=kernel_3, stride=1, padding=1),
      nn.ReLU(),

      nn.Conv2d(size_3, size_3, kernel_size=kernel_3, stride=1, padding=1),
      nn.ReLU(),

      nn.Conv2d(size_3, size_2, kernel_size=kernel_3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2))

      #nn.ReLU(inplace=True),
      
     # nn.BatchNorm2d(size_1),
     # nn.Dropout(0.2),
      
     # nn.MaxPool2d(kernel_size=2, stride=2),
      
      
     
    self.linear_layers = nn.Sequential(
      nn.Dropout(0.5),
      nn.Linear(in_features=size_linear, out_features=4096),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(in_features=4096, out_features=4096),
      nn.ReLU(),
      nn.Linear(in_features=4096, out_features=num_classes)
    )
    

  def forward(self, x):
    x = self.cnn_layers(x)
    x = torch.flatten(x, 1)
    output = self.linear_layers(x)
    return output

class CropTransform:
  def __init__(self, region):
    self.region = region
  
  def __call__(self, x):
    return transforms.functional.crop(x, self.region[0], self.region[1], self.region[2], self.region[3])

def data_preview(data_set):
  labels_reference = list(data_set.class_to_idx.keys())
  data_loader = torch.utils.data.DataLoader(data_set,shuffle=True, batch_size=9)
  images, labels = next(iter(data_loader))
  ii = 1
  for image, label in zip(images, labels):
    plt.subplot(3, 3, ii)
    image_transposed = np.transpose(image.numpy(), (1, 2, 0))
    if image_transposed.shape[2] == 1:
      plt.imshow(image_transposed[:, :, 0])
    else:
      plt.imshow(image_transposed)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    plt.title(labels_reference[label])
    ii += 1
  plt.show()

def get_accuracy(loader, my_net):
  correct = 0
  total = 0
  for images, labels in loader:
    images = Variable(images)
    outputs = my_net(images)
    _, predicted = torch.max(outputs.data, 1)  
    total += labels.size(0)               
    correct += (predicted == labels).sum()
  return 100 * correct // total

def label_preview(data_set, data_loader, my_net):
  labels_reference = list(data_set.class_to_idx.keys())
  images, labels = next(iter(data_loader))
  images = Variable(images)
  outputs = my_net(images)
  _, predicted = torch.max(outputs.data, 1)  
  labels_predict = predicted.numpy()
  for ii in range(8):
    image = images[ii]
    label = labels[ii]
    actual_label = labels_reference[label.numpy()]
    plt.subplot(2, 4, ii+1)
    image_transposed = np.transpose(image.numpy(), (1, 2, 0))
    if image_transposed.shape[2] == 1:
      plt.imshow(image_transposed[:, :, 0])
    else:
      plt.imshow(image_transposed)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    title = 'Predicted: ' + str(labels_reference[labels_predict[ii]]) + '\n Actual: ' + str(actual_label)
    plt.title(title)
  plt.show()


batch_size = 100
validation_split = 0.1
lr = 3e-4
random_seed = int(time.time())
data_path = 'CCSN'
image_size = 255
transform = transforms.Compose([ transforms.ToTensor(), transforms.Resize(image_size)])

train_set_no_transform = datasets.ImageFolder(data_path, transform=transform)
data_preview(train_set_no_transform)

transform = transforms.Compose([ transforms.ToTensor(), transforms.Resize(image_size), transforms.RandomHorizontalFlip(), transforms.RandomRotation([-20, 20])]) #, CropTransform([0, 0, image_size-50, image_size]),
train_set = datasets.ImageFolder(data_path, transform=transform)
data_preview(train_set)

train_set_size = len(train_set)
indices = list(range(train_set_size))
split = int(np.floor(validation_split * train_set_size))

np.random.seed(random_seed)
np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

labels_reference = list(train_set.class_to_idx.keys())

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=val_sampler)

unique_classes_count = len(labels_reference)
net = CnnNet(unique_classes_count)

criterion = nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(net.parameters(), lr=lr)
num_epochs = 50

loss_tracker = []

for epoch in range(num_epochs):
  loss = torch.tensor([100])
  ii = 0
  for images, labels in train_loader:
    images = Variable(images)
    labels = Variable(labels)

    optimizer.zero_grad()
    outputs = net(images)
    loss = criterion(outputs, labels)
    loss.backward()      
    optimizer.step()

    loss_tracker.append(loss.data)                                  

    if (ii+1) % batch_size == 0 or (ii+1) == len(train_loader):   
      print('Epoch [%d/%d],  Val Acc: %d %%, Training Loss: %.4f'
              %(epoch+1, num_epochs, \
                get_accuracy(val_loader, net), loss.data))
    ii += 1

train_acc = get_accuracy(train_loader, net)
test_acc = get_accuracy(val_loader, net)

print('Accuracy of the network on train images: %d %%' % (train_acc))
print('Accuracy of the network on val images: %d %%' % (test_acc))

# Plot the loss over time
plt.plot(loss_tracker)
plt.ylabel("Loss")
plt.xlabel("Step Number")
plt.title("Loss over time")
plt.show()

label_preview(train_set, val_loader, net)