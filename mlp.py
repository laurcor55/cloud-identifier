import os
import random
from mlp_lib import Net
import mlp_lib as mlp
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import tqdm

batch_size = 100
validation_split = .2

shuffle_dataset = True
random_seed = 2
data_path = 'CCSN_v2'
transform = transforms.Compose([transforms.Resize(255),  transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_set = datasets.ImageFolder(data_path, transform=transform)

# Create validation split by taking a percentage of the training set:
train_set_size = len(train_set)
indices = list(range(train_set_size))
split = int(np.floor(validation_split * train_set_size))

# Shuffle the dataset to increase generalization and speed training
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

# Use the SubsetRandomSampler to randomly sample the training dataset for 
# training and validation data. We will feed this into the dataloader below.
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

labels_reference = list(train_set.class_to_idx.keys())

# Create dataloader objects - will be used during training and inference
# to iterate over the data.
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=val_sampler)

images, labels = next(iter(train_loader))

plt.imshow(images[0].numpy().transpose((1, 2, 0)))
plt.title(labels_reference[labels[0]])
plt.show()

# Create a neural network object with the specified number of input neurons,
# hidden neurons, and output neurons (or total classes)
unique_classes_count = len(labels_reference)
input_neurons_count = 255*255*3
hidden_neurons_count = 16
net = Net(input_neurons_count, hidden_neurons_count, unique_classes_count)

# We use cross-entropy loss with the Adam optimizer. No need to understand what
# these two mean just yet, we will go over cross-entropy soon and the Adam 
# optimizer in a later lecture.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


# Here we create our training loop, which runs for the number of epochs
# specified below. 

# 1 Epoch represents one entire pass through your training dataset, so 20 epochs
# means that your model will see each data point ~20 times during training.
num_epochs = 50

loss_tracker = []

for epoch in range(num_epochs):
  # Initialize random value for loss just for displaying purposes
  loss = torch.tensor([batch_size])
  i = 0
  for images, labels in train_loader:
    # Convert torch tensor to a vector of size 784 in order to send it to input
    # layer
    images = Variable(images.view(-1, input_neurons_count))
    labels = Variable(labels)

    optimizer.zero_grad()
    outputs = net(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # Track losses for plotting later
    loss_tracker.append(loss.data)                                  

    # Visualization code
    if (i+1) % batch_size == 0 or (i+1) == len(train_loader):   
      print('Epoch [%d/%d],  Val Acc: %d, Training Loss: %.4f'
              %(epoch+1, num_epochs, \
                mlp.get_accuracy(val_loader, net), loss.data))
    i += 1

# Print the training and test accuracy
train_acc = mlp.get_accuracy(train_loader, net)
test_acc = mlp.get_accuracy(val_loader, net)

print('Accuracy of the network on train images: %d %%' % (train_acc))
print('Accuracy of the network on val images: %d %%' % (test_acc))

# Plot the loss over time
plt.plot(loss_tracker)
plt.ylabel("Loss")
plt.xlabel("Step Number")
plt.title("Loss over time")
plt.show()