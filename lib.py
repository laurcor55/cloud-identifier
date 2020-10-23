from torch import nn
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
import numpy as np


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


# Here we are going to create our neural network class by inheiriting the 
# nn.Module class. The main functions to notice are the initialization function 
# __init__() and forward propagation function forward().

class CnnNet(nn.Module):

  # We initialize our neural net by defining the input size, number of nodes
  # in hidden layers, and the total number of classes we are detecting.
  def __init__(self, input_size, hidden_size, num_classes):

    # IMPORTANT: This is often forgotten, but make sure to make a super call.
    super(Net, self).__init__()
    #not sequential. forward pass is sequential.
    self.conv1 = nn.Conv2d(1, 10, kernel_size=3) # 1 input channel since black and white. 10 output channels.
    self.conv2 = nn.Conv2d(10, 20, 5) # input must be the same as output of previous. size of kernel is 5. output size is 20
    self.maxpool = nn.MaxPool2d(2) # after first convolution.
    self.fc = nn.Linear(320, 10) # fully connected layer. 320 from the output of the second maxpool. 10 is number of classes.
    self.relu = nn.functional.relu

  
  # Below we construct the output of the forward propagation pass of the neural
  # net using the layers and activation functions defined in the Net constructor.
  def forward(self, x):
    input_size = x.size(0)
    out = self.relu(self.conv1(x))
    out = self.maxpool(out)
    out = self.relu(self.conv2(out))
    out = self.maxpool(out)

    fc_input = out.view(input_size, -1) # flattens into vector of correct size (320)
    fc_output = self.fc(fc_input)

    return fc_output

class MlpNet(nn.Module):

  # We initialize our neural net by defining the input size, number of nodes
  # in hidden layers, and the total number of classes we are detecting.
  def __init__(self, input_size, hidden_size, num_classes):
    super(MlpNet, self).__init__() # The Net class expands upon the nn.Module
    
    # Hidden Layer 1
    self.fc1 = nn.Linear(input_size, hidden_size) 
    self.sigmoid = nn.Sigmoid() 

    # Hidden Layer 2
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    #already defined sigmoid

    # Final Output Layer
    self.fc3 = nn.Linear(hidden_size, num_classes)


  # Below we construct the output of the forward propagation pass of the neural
  # net using the layers and activation functions defined in the Net constructor.
  def forward(self, x):

    # Hidden Layer 1
    out = self.fc1(x) # takes input, passes it through first layer then gives output.
    out = self.sigmoid(out) #activation function. Ready to go to second layer

    # Hidden Layer 2
    out = self.fc2(out)
    out = self.sigmoid(out)

    # Pass to Output layer
    out = self.fc3(out)

    return out

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



