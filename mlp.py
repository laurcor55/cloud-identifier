import random
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt


class MlpNet(nn.Module):

  # We initialize our neural net by defining the input size, number of nodes
  # in hidden layers, and the total number of classes we are detecting.
  def __init__(self, input_size, hidden_size, num_classes):
    super(MlpNet, self).__init__() # The Net class expands upon the nn.Module
    
    self.mlp_layers = nn.Sequential(
      nn.Linear(input_size, hidden_size),
      #nn.Sigmoid(),
      nn.LeakyReLU(0.1),
      nn.Linear(hidden_size, hidden_size),
      nn.LeakyReLU(0.1),
      #nn.Sigmoid(),
      nn.Linear(hidden_size, num_classes),
      nn.LeakyReLU(0.1)
    )

  def forward(self, x):
    out = self.mlp_layers(x)
    return out

def get_accuracy(loader, my_net):
  correct = 0
  total = 0
  for images, labels in loader:
    images = Variable(images.view(-1, 255*255*3))
    outputs = my_net(images)
    _, predicted = torch.max(outputs.data, 1)  
    total += labels.size(0)               
    correct += (predicted == labels).sum()
  return 100 * correct // total

def data_preview(data_set):
  labels_reference = list(data_set.class_to_idx.keys())
  data_loader = torch.utils.data.DataLoader(data_set, batch_size=9)
  images, labels = next(iter(data_loader))
  ii = 1
  for image, label in zip(images, labels):
    plt.subplot(3, 3, ii)
    plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    
    plt.title(labels_reference[label])
    ii += 1
  plt.show()


def label_preview(data_set, data_loader, my_net):
  labels_reference = list(data_set.class_to_idx.keys())
  images, labels = next(iter(data_loader))
  images_original = Variable(images)
  images = Variable(images.view(-1, input_neurons_count))
  outputs = my_net(images)
  _, predicted = torch.max(outputs.data, 1)  
  labels_predict = predicted.numpy()
  for ii in range(8):
    image = images_original[ii]
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
validation_split = .1

random_seed = 2
data_path = 'CCSN'
transform = transforms.Compose([transforms.Resize(255), transforms.ToTensor()])#,, transforms.Normalize((0.5,), (0.5,), transforms.Grayscale() MyCropTransform([70, 0, 400, 330])])

train_set = datasets.ImageFolder(data_path, transform=transform)
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

data_preview(train_set)
unique_classes_count = len(labels_reference)
input_neurons_count = 255**2*3
hidden_neurons_count = 200
net = MlpNet(input_neurons_count, hidden_neurons_count, unique_classes_count)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

num_epochs = 100
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
                get_accuracy(val_loader, net), loss.data))
    i += 1

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