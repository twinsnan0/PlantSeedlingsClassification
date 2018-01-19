import constants

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from model import Net
from seedlingsdata import SeedlingsData


def train():
    data = SeedlingsData()
    data.load([constants.test_output_resize_file_path, constants.test_output_rotate_file_path,
               constants.test_output_crop_file_path], validate=0.05)
    data.set_batch_size(32)

    # Create network
    net = Net()
    print(net)

    for epoch in range(0, 50):
        train_epoch(net, data, epoch)
        validate_epoch(net, data, epoch)


def train_epoch(net: Net, data: SeedlingsData, epoch: int):
    for batch_index, images, labels in data.generate_train_data():
        batch_x = Variable(torch.from_numpy(images)).float() / 255
        batch_y = Variable(torch.from_numpy(labels)).long()

        output = net(batch_x)

        optimizer = optim.Adam(net.parameters())
        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        print("epoch:{} ,batch index:{}, loss:{}".format(epoch, batch_index, loss.data[0]))
        # Validate
        if batch_index != 0 and batch_index % 100 == 0:
            pass


def validate_epoch(net: Net, data: SeedlingsData, epoch: int):
    validate_total = 0
    validate_right = 0
    for validate_batch_index, validate_images, validate_labels in data.generate_validate_data():
        validate_batch_x = Variable(torch.from_numpy(validate_images)).float() / 255
        validate_batch_y = Variable(torch.from_numpy(validate_labels)).long()
        validate_output = net(validate_batch_x)
        _, predict_batch_y = torch.max(validate_output, 1)
        validate_total += validate_batch_y.size(0)
        validate_right += sum(predict_batch_y.data.cpu().numpy() == validate_batch_y.data.cpu().numpy())
        accuracy = validate_right / validate_total
        print("Epoch:{} ,validate_batch index:{}, validate_accuracy:{}".format(epoch, validate_batch_index,
                                                                               accuracy))

    accuracy = validate_right / float(validate_total)
    print("Epoch:{}, validate accuracy: %.4f".format(epoch, accuracy))


if __name__ == "__main__":
    # Train network
    train()
