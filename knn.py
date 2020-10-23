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
shuffle_dataset = True
data_path = 'CCSN_v2'
input_neurons_count = 255*255*3
validation_split = .2
random_seed = 2
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
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

labels_reference = list(train_set.class_to_idx.keys())

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(train_set, sampler=val_sampler)

images, labels = next(iter(train_loader))
plt.imshow(images[0].numpy().transpose((1, 2, 0)))
plt.title(labels_reference[labels[0]])
plt.show()
correct_count = 0
total_count = 0
for val_image, val_label in val_loader:
  val_image = val_image.numpy().reshape(1, input_neurons_count)
  distance = []
  train_label_all = []
  for train_image, train_label in train_loader:
    train_image = train_image.numpy().reshape(len(train_label), input_neurons_count)
    distance.extend(np.sum(np.subtract(train_image, val_image)**2, axis=1))
    train_label_all.extend(train_label.numpy())
  ind = distance.index(min(distance))
  val_label_predict = train_label_all[ind]
  total_count += 1
  if (val_label_predict == val_label.numpy()):
    correct_count += 1
  print(correct_count/total_count)
