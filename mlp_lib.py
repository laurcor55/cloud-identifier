from torch import nn
from torch.autograd import Variable
import torch


# Function for getting accuracy, adapted from: 
# https://towardsdatascience.com/a-simple-starter-guide-to-build-a-neural-network-3c2cf07b8d7c

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

class Net(nn.Module):

  # We initialize our neural net by defining the input size, number of nodes
  # in hidden layers, and the total number of classes we are detecting.
  def __init__(self, input_size, hidden_size, num_classes):
    super(Net, self).__init__() # The Net class expands upon the nn.Module
    
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


