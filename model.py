import torch.nn as nn
import torch.nn.functional as F

INPUT_CHANNELS = 3
SPECIES_SIZE = 12


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # An affine operation: y = Wx + b
        self.fc1 = nn.Linear(77 * 77 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, SPECIES_SIZE)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        # All dimensions except the batch dimension
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__":
    # Create network
    net = Net()
    net.cuda()
    print(net)
