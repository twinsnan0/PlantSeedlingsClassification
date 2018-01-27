import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

INPUT_CHANNELS = 3
SPECIES_SIZE = 12


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = models.resnet101(pretrained=True)
        self.resnet.fc = F.relu(nn.Linear(2048, 1024))
        for param in self.resnet.parameters():
            param.requires_grad = True
        self.fc1 = nn.Linear(1024, 120)
        self.fc2 = nn.Linear(120, SPECIES_SIZE)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Create network
    net = Net()
    print(net)
