import lib
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np

batch_size = 100
data_path = 'CCSN'
input_neurons_count = 400**2
validation_split = .05
random_seed = 25
transform = transforms.Compose([transforms.Resize(400), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.Grayscale()])

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

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(train_set, sampler=val_sampler)

lib.data_preview(train_set)

correct_count = 0
total_count = 0
val_labels = np.zeros(len(val_loader)).astype(int)

for train_image, train_label in train_loader:
  train_image = train_image.numpy().reshape(len(train_label), input_neurons_count)
  
  distance = np.zeros((len(train_label), len(val_loader)))
  val_ind = 0

  for val_image, val_label in val_loader:
    val_image = val_image.numpy().reshape(1, input_neurons_count)
    diff = np.sum(np.subtract(train_image, val_image)**2, axis=1)
    distance[:, val_ind] = diff
    val_labels[val_ind] = val_label.numpy()

    val_ind += 1
  batch_min = np.min(distance, axis=0)
  batch_labels = train_label.numpy()[np.argmin(distance, axis=0)]
  if (total_count==0):
    min_distance = batch_min
    min_labels = batch_labels
  else:
    ind_to_update = np.argwhere(batch_min < min_distance)
    min_distance[ind_to_update] = batch_min[ind_to_update]
    min_labels[ind_to_update] = batch_labels[ind_to_update]
  total_correct = sum(min_labels==val_labels)
  total_tested = len(val_labels)
  correct = total_correct/total_tested
  print(correct)
  total_count += 1
