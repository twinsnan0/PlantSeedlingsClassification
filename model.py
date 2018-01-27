import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

INPUT_CHANNELS = 3
SPECIES_SIZE = 12


class Net(nn.Module):
    def __init__(self, model_name):
        super(Net, self).__init__()
        self.model_name = model_name
        if model_name == 'resnet101':
            self.model = models.resnet101(pretrained=True)
            self.model.fc = nn.Linear(2048, 1024)

        elif model_name == 'resnet152':
            self.model = models.resnet152(pretrained=True)
            self.model.fc = nn.Linear(2048, 1024)

        elif model_name == 'densenet161':
            self.model = models.densenet161(pretrained=True)
            self.model.classifier = nn.Linear(1920, 1024)

        elif model_name == 'densenet201':
            self.model = models.densenet201(pretrained=True)
            self.model.classifier = nn.Linear(1920, 1024)

        elif model_name == 'inception_v3':
            self.model = models.inception_v3(pretrained=True)
            self.model.fc = nn.Linear(2048, 1024)

        else:
            print("wrong model, select 'resnet101', 'resnet152', 'densenet161', 'densenet201' or 'inception_v3'")

        for param in self.model.parameters():
            param.requires_grad = True
        self.fc1 = nn.Linear(1024, 120)
        self.fc2 = nn.Linear(120, SPECIES_SIZE)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Create network
    net = Net('inception_v3')
    print(net)
