import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(3, 1, 1, bias=False)
        self.fc1 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x



net = Net()
def is_trainable(net):
    zeros = torch.zeros((1, 3, 32, 32))  # input tensor of shape (batch_size, channels, height, width)

    # Check that we can pass a dummy input through the network without errors.
    try:
        output = net(zeros)
    except Exception as e:
        return False
    
    # Network output shape must match number of classes in CIFAR-10.
    if output.shape != (1, 10):
        return False

    return True

def main():
    print(is_trainable(net))

if __name__ == '__main__':
    main()