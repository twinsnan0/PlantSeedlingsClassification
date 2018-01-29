import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

INPUT_CHANNELS = 3
SPECIES_SIZE = 12


class Net(nn.Module):
    def __init__(self, model_name):
        super(Net, self).__init__()
        self.model_name = model_name

        if self.model_name == 'resnet50+':
            self.model = models.resnet50(pretrained=True)
            self.model.fc = nn.Linear(2048, 1024)

        if self.model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            self.model.fc = nn.Linear(2048, 1024)

        elif self.model_name == 'resnet152':
            self.model = models.resnet152(pretrained=True)
            self.model.fc = nn.Linear(2048, 1024)

        elif self.model_name == 'densenet161':
            self.model = models.densenet161(pretrained=True)
            self.model.classifier = nn.Linear(1920, 1024)

        elif self.model_name == 'densenet201':
            self.model = models.densenet201(pretrained=True)
            self.model.classifier = nn.Linear(1920, 1024)

        elif self.model_name == 'inception_v3':
            self.model = models.inception_v3(pretrained=True)
            self.model.fc = nn.Linear(2048, 1024)

        else:
            print(
                "wrong model, select 'resnet50+', 'resnet50','resnet101', 'resnet152', 'densenet161', 'densenet201' or "
                "'inception_v3'")

        for param in self.model.parameters():
            param.requires_grad = True
        if self.model_name == 'resnet50+':
            self.fc1 = nn.Linear(1027, 120)
        else:
            self.fc1 = nn.Linear(1024, 120)
        self.fc2 = nn.Linear(120, SPECIES_SIZE)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, plant_area=None, avg_prob=None, avg_green=None):
        x = self.model(x)
        x = x.view(-1, 1024)
        if self.model_name == 'resnet50+':
            x = torch.cat((x, plant_area, avg_prob, avg_green), dim=1)
        else:
            pass
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Create network
    net = Net('inception_v3')
    print(net)
