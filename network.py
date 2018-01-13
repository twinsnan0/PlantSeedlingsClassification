import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from seedlingsdata import SeedlingsData

BATCH_SIZE = 128
IMAGE_SIZE = 320
LEARN_RATE = 0.01
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
        self.fc1 = nn.Linear(94864, 120)
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

    def num_flat_features(self, x):
        # All dimensions except the batch dimension
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__":
    # Create network
    net = Net()
    print(net)
    # Input data
    test_output_resize_file_path = r"D:\Project\Space\Python\Data\seedings_data\train\resize"
    test_output_rotate_file_path = r"D:\Project\Space\Python\Data\seedings_data\train\rotate"
    test_output_crop_file_path = r"D:\Project\Space\Python\Data\seedings_data\train\crop"

    data = SeedlingsData()
    data.load([test_output_resize_file_path, test_output_rotate_file_path, test_output_crop_file_path])
    for epoch in range(0, 101):
        for batch_index, images, labels in data.generate_train_data():
            batch_x = Variable(torch.from_numpy(images)).float()
            batch_y = Variable(torch.from_numpy(labels)).long()

            output = net(batch_x)

            optimizer = optim.SGD(net.parameters(), lr=LEARN_RATE)
            optimizer.zero_grad()
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            print("epoch:{} ,batch index:{}, loss:{}".format(epoch, batch_index, loss))
