import torch.nn as nn
import torchvision.models as models

INPUT_CHANNELS = 3
SPECIES_SIZE = 12


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = models.resnet101(pretrained=True)
        self.resnet.fc = nn.Linear(2048, SPECIES_SIZE)
        for param in self.resnet.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.resnet(x)
        return x


if __name__ == "__main__":
    # Create network
    net = Net()
    print(net)
