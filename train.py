import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms

import constants
from model import Net
from seedlingsdata import SeedlingsData

accuracy_list = [0.0]


def train(save_directory: str, model_path: str = None):
    data = SeedlingsData()
    data.load(train_data_paths=[constants.train_output_resize_file_path, constants.train_output_rotate_file_path,
                                constants.train_output_crop_file_path],
              test_data_paths=[constants.test_output_resize_file_path], validate=0.2)
    data.set_batch_size(64)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Create network
    net = Net()
    net.cuda()
    print(net)

    if model_path is not None:
        load_model(net, model_path)

    for epoch in range(0, 1):
        train_epoch(net, data, epoch, normalize)
        accuracy = validate_epoch(net, data, epoch, normalize)
        accuracy_list.append(accuracy)
        save_model(net, save_directory, accuracy=accuracy)
        if accuracy_list.index(max(accuracy_list)) == len(accuracy_list) - 1:
            save_model(net, save_directory, is_best=True)

    del net


def test(model_path: str = None):
    data = SeedlingsData()
    data.load(train_data_paths=[], test_data_paths=[constants.test_output_resize_file_path])
    data.set_batch_size(32)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Create network
    net = Net()
    net.cuda()
    print(net)

    load_model(net, model_path)
    create_submission_file("submission.txt")

    with open("submission.txt", "a") as submission_file:
        for test_image_dir, test_image in data.generate_test_data():
            test_tensor = normalize(torch.from_numpy(test_image))
            test_x = Variable(test_tensor, volatile=True).cuda().float()
            test_output = net(test_x)
            _, predict_y = torch.max(test_output, 1)
            print("Predict result:")
            print(predict_y)
            file_name = os.path.split(test_image_dir)[1].split("_")[0] + '.' + \
                        os.path.split(test_image_dir)[1].split('.')[1]
            string_to_write = "{},{}\r\n".format(file_name, SeedlingsData.seedlings_labels[predict_y.data[0]])
            submission_file.write(string_to_write)
            print(string_to_write)

    del net


def create_submission_file(filename):
    # Overwrites the existing file if the file exists
    # If the file does not exist, creates a new file for reading and writing
    file = open(filename, "w+")
    file.write("file,species\r\n")


def train_epoch(net: Net, data: SeedlingsData, epoch: int, normalize: transforms.Normalize):
    for batch_index, images, labels in data.generate_train_data():
        tensor = normalize(torch.from_numpy(images))
        batch_x = Variable(tensor).cuda().float()
        batch_y = Variable(torch.from_numpy(labels)).cuda().long()

        output = net(batch_x)
        if batch_index <= 5:
            optimizer = optim.Adam(net.parameters(), lr=0.0001)
        if batch_index > 5:
            optimizer = optim.Adam(net.parameters(), lr=0.00001)
        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        print("epoch:{} ,batch index:{}, loss:{}".format(epoch, batch_index, loss.data[0]))
        # Validate
        if batch_index != 0 and batch_index % 100 == 0:
            pass


def validate_epoch(net: Net, data: SeedlingsData, epoch: int, normalize: transforms.Normalize):
    validate_total = 0
    validate_right = 0
    for validate_batch_index, validate_images, validate_labels in data.generate_validate_data():
        validate_tensor = normalize(torch.from_numpy(validate_images))
        validate_batch_x = Variable(validate_tensor, volatile=True).cuda().float()
        validate_batch_y = Variable(torch.from_numpy(validate_labels), volatile=True).cuda().long()
        validate_output = net(validate_batch_x)
        _, predict_batch_y = torch.max(validate_output, 1)
        validate_total += validate_batch_y.size(0)
        validate_right += sum(predict_batch_y.data.cpu().numpy() == validate_batch_y.data.cpu().numpy())
        accuracy = validate_right / validate_total
        print("Epoch:{} ,validate_batch index:{}, validate_accuracy:{}".format(epoch, validate_batch_index,
                                                                               accuracy))
    net.eval()

    accuracy = validate_right / float(validate_total)
    print("Epoch:{}, validate accuracy: {}".format(epoch, accuracy))
    return accuracy


def save_model(net: Net, save_directory, accuracy=0.0, is_best=False):
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    if not is_best:
        current_time = time.strftime("%Y%m%d_%H%M", time.localtime())
        file_name = current_time + "_" + str(accuracy) + ".pkl"
        file_path = os.path.join(save_directory, file_name)
        torch.save(net.state_dict(), file_path)
    else:
        file_path = os.path.join(constants.save_file_directory, "best.pkl")
        torch.save(net.state_dict(), file_path)


def load_model(net: Net, path: str):
    net.load_state_dict(torch.load(path))


if __name__ == "__main__":
    # Train network
    train(constants.save_file_directory)
    # Test
    best_model_path = os.path.join(constants.save_file_directory, "best.pkl")
    test(best_model_path)
