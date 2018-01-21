import os
import random

import cv2
import numpy as np

import constants
import remove_background

WIDTH = 224
HEIGHT = 224


class PreProcess(object):
    def __init__(self):
        pass

    @staticmethod
    def rotate(source_path: str, output_path: str, rotation_angles: list = None):
        """
        Rotate the images
        :param source_path: source directory path
        :param output_path: output directory path
        :param rotation_angles : what angle should be rotated
        :return:
        """
        if rotation_angles is None:
            rotation_angles = [90, 180, 270]

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        directories = os.listdir(source_path)
        for directory in directories:
            out_directory_path = os.path.join(output_path, directory)
            if not os.path.exists(out_directory_path):
                os.makedirs(out_directory_path)
            for file in os.listdir(os.path.join(source_path, directory)):
                # Rotate
                image = cv2.imread(os.path.join(source_path, directory, file))
                for angle in rotation_angles:
                    rotated = PreProcess._rotate(image, angle)
                    last_index = file.rfind(".")
                    origin_name = file[0:last_index]
                    output_file_name = os.path.join(out_directory_path,
                                                    str(origin_name) + "_rotate_" + str(angle) + ".png")
                    rotated = PreProcess._resize(rotated, WIDTH, HEIGHT)
                    cv2.imwrite(output_file_name, rotated)
                    print("Saved: ", output_file_name)

    @staticmethod
    def resize(source_path: str, output_path: str, resize_width=WIDTH, resize_height=HEIGHT):
        """
        Resize the image
        :param source_path: source directory path
        :param output_path: output directory path
        :param resize_width: resize width
        :param resize_height: resize height
        :return:
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        directories = os.listdir(source_path)
        for directory in directories:
            out_directory_path = os.path.join(output_path, directory)
            if not os.path.exists(out_directory_path):
                os.makedirs(out_directory_path)
            for file in os.listdir(os.path.join(source_path, directory)):
                # Resize
                image = cv2.imread(os.path.join(source_path, directory, file))
                resized = PreProcess._resize(image, resize_width, resize_height)
                last_index = file.rfind(".")
                origin_name = file[0:last_index]
                output_file_name = os.path.join(out_directory_path,
                                                str(origin_name) + "_resize_" + str(resize_width) + ".png")
                cv2.imwrite(output_file_name, resized)
                print("Saved: ", output_file_name)

    @staticmethod
    def remove_background(source_path: str, output_path: str):
        """
        Resize the background of images
        :param source_path: source directory path
        :param output_path: output directory path
        :param resize_width: resize width
        :param resize_height: resize height
        :return:
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        directories = os.listdir(source_path)
        for directory in directories:
            out_directory_path = os.path.join(output_path, directory)
            if not os.path.exists(out_directory_path):
                os.makedirs(out_directory_path)
            for file in os.listdir(os.path.join(source_path, directory)):
                # Resize
                image = cv2.imread(os.path.join(source_path, directory, file))
                removed = remove_background.remove_background(image)
                last_index = file.rfind(".")
                origin_name = file[0:last_index]
                output_file_name = os.path.join(out_directory_path,
                                                str(origin_name) + "_background_removed" + ".png")
                cv2.imwrite(output_file_name, removed)
                print("Saved: ", output_file_name)

    @staticmethod
    def crop(source_path: str, output_path: str, crop_ratio=0.8, resize_width=WIDTH, resize_height=HEIGHT):
        """
        Crop the image then resize to the specified size
        :param source_path: source directory path
        :param output_path: output directory path
        :param crop_ratio: crop ratio
        :param resize_width: resize width
        :param resize_height: resize height
        :return:
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        directories = os.listdir(source_path)
        for directory in directories:
            out_directory_path = os.path.join(output_path, directory)
            if not os.path.exists(out_directory_path):
                os.makedirs(out_directory_path)
            for file in os.listdir(os.path.join(source_path, directory)):
                # Crop
                image = cv2.imread(os.path.join(source_path, directory, file))
                croped = PreProcess._resize(PreProcess._crop(image, crop_ratio), resize_width, resize_height)
                last_index = file.rfind(".")
                origin_name = file[0:last_index]
                output_file_name = os.path.join(out_directory_path,
                                                str(origin_name) + "_crop_" + str(crop_ratio) + ".png")
                cv2.imwrite(output_file_name, croped)
                print("Saved: ", output_file_name)

    @staticmethod
    def _rotate(image, angle, center=None, scale=1.0):
        h, w = image.shape[:2]
        if center is None:
            center = (w / 2, h / 2)

        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, matrix, (w, h))

        return rotated

    @staticmethod
    def _crop(image, crop_ratio):
        offset_height = random.random() * (1 - crop_ratio)
        offset_width = random.random() * (1 - crop_ratio)
        h, w = image.shape[:2]
        return image[int(offset_width * w):int((crop_ratio + offset_width) * w),
               int(offset_height * h):int((crop_ratio + offset_height) * h)]

    @staticmethod
    def _resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        h, w = image.shape[:2]

        if width is None and height is None:
            dim = (w, h)
        elif width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        elif height is None:
            r = width / float(w)
            dim = (width, int(h * r))
        else:
            dim = (width, height)

        return cv2.resize(image, dim, interpolation=inter)


class SeedlingsData(object):

    def __init__(self):
        self.paths = []
        self.labels = {}
        self._data = []
        self.batch_size = 128
        self.validate = 0.0
        self.train_size = 0
        self.validate_size = 0

        seedlings_labels = ["Black-grass", "Charlock", "Cleavers", "Common Chickweed", "Common wheat",
                            "Fat Hen", "Loose Silky-bent", "Maize", "Scentless Mayweed",
                            "Shepherds Purse", "Small-flowered Cranesbill", "Sugar beet"]

        for index, label in enumerate(seedlings_labels):
            self.labels[label] = index

    def load(self, train_data_paths: list, train=True, shuffle=True, validate=0.2):

        """
        Load data and save in a list like this:
        the element's list contains two string,one is the file path of an image, one is the class of the image.
        [['D:\\Project\\Space\\Python\\Data\\seedings_data\\train\\train\\Black-grass\\0050f38b3.png', 'Black-grass']]
        :param train_data_paths: path of the data
        :param train: whether is the train data
        :param shuffle: should shuffle
        :param validate: validation ratio
        :return:
        """
        self._data.clear()
        self.paths = train_data_paths
        self.validate = validate

        if train:
            for data_path in train_data_paths:
                directories = os.listdir(data_path)
                for directory in directories:
                    for root, dirs, files in os.walk(os.path.join(data_path, directory)):
                        for file in files:
                            # Save file path and class
                            self._data.append([os.path.join(root, file), self.labels[directory]])
        else:
            for data_path in train_data_paths:
                for root, dirs, files in os.walk(os.path.join(data_path, data_path)):
                    for file in files:
                        # Save file path and undefined class
                        self._data.append([os.path.join(root, file), None])
        if shuffle:
            random.shuffle(self._data)

        self.train_size = int(len(self._data) * (1 - self.validate))
        self.validate_size = int(len(self._data) * self.validate)

        print("train_size:{}".format(self.train_size))
        print("validate_size:{}".format(self.validate_size))

    def set_batch_size(self, size):
        """
        Set the batch size for training
        :param size: batch size
        :return:
        """
        self.batch_size = size

    @property
    def train_data(self) -> list:
        return self._data[:self.train_size]

    @property
    def validate_data(self) -> list:
        return self._data[self.train_size:]

    def generate_train_data(self):
        current_index = 0
        batch_index = 0
        while current_index + self.batch_size < self.train_size:
            batch_data = self._data[current_index: current_index + self.batch_size]
            h, w, c = cv2.imread(batch_data[0][0]).shape[:]
            batch_images = np.zeros((self.batch_size, c, w, h), dtype=np.float)
            for index, image in enumerate(batch_data):
                image_rgb = cv2.cvtColor(cv2.imread(image[0]), cv2.COLOR_BGR2RGB)
                batch_images[index] = (np.transpose(image_rgb, (2, 0, 1)))
            batch_labels = np.array([image[1] for image in batch_data])

            yield batch_index, batch_images, batch_labels
            current_index += self.batch_size
            batch_index += 1

    def generate_validate_data(self):
        current_index = self.train_size
        batch_index = 0
        while current_index + self.batch_size < len(self._data):
            batch_data = self._data[current_index: current_index + self.batch_size]
            h, w, c = cv2.imread(batch_data[0][0]).shape[:]
            batch_images = np.zeros((self.batch_size, c, w, h), dtype=np.float)
            for index, image in enumerate(batch_data):
                image_rgb = cv2.cvtColor(cv2.imread(image[0]), cv2.COLOR_BGR2RGB)
                batch_images[index] = (np.transpose(image_rgb, (2, 0, 1)))
            batch_labels = np.array([image[1] for image in batch_data])

            yield batch_index, batch_images, batch_labels
            current_index += self.batch_size
            batch_index += 1


if __name__ == "__main__":
    # Replace with your directory
    # First we should pre-process the image data
    # Resize
    pre_process = PreProcess()
    # pre_process.resize(constants.test_file_path, constants.test_output_resize_file_path)

    # Rotate
    # pre_process.rotate(constants.test_output_resize_file_path, constants.test_output_rotate_file_path)

    # Crop
    # pre_process.crop(constants.test_output_resize_file_path, constants.test_output_crop_file_path)

    # Remove background
    pre_process.remove_background(constants.test_output_resize_file_path,
                                  constants.test_output_remove_background_file_path)

    # After pre-processing, we need to input data for training
    # data = SeedlingsData()
    # data.load([constants.test_output_resize_file_path, constants.test_output_rotate_file_path,
    #            constants.test_output_crop_file_path])

    # Iterate method
    # for images, labels in data.generate_train_data():
    #     print(type(images))
    #     print(images.shape)
    #     print(labels.shape)
