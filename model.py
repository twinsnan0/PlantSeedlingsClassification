import torch.nn as nn
import torchvision.models as models

INPUT_CHANNELS = 3
SPECIES_SIZE = 12


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(2048, SPECIES_SIZE)

    def forward(self, x):
        x = self.resnet50(x)
        return x


if __name__ == "__main__":
    # Create network
    net = Net()
    print(net)
