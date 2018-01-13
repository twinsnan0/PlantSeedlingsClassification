import os
import random

import cv2


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
                    rotated = PreProcess.__rotate(image, angle)
                    last_index = file.rfind(".")
                    origin_name = file[0:last_index]
                    output_file_name = os.path.join(out_directory_path,
                                                    str(origin_name) + "_rotate_" + str(angle) + ".png")
                    cv2.imwrite(output_file_name, rotated)
                    print("Saved: ", output_file_name)

    @staticmethod
    def resize(source_path: str, output_path: str, resize_width=320, resize_height=320):
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
                resized = PreProcess.__resize(image, resize_width, resize_height)
                last_index = file.rfind(".")
                origin_name = file[0:last_index]
                output_file_name = os.path.join(out_directory_path,
                                                str(origin_name) + "_resize_" + str(resize_width) + ".png")
                cv2.imwrite(output_file_name, resized)
                print("Saved: ", output_file_name)

    @staticmethod
    def crop(source_path: str, output_path: str, crop_ratio=0.5, resize_width=320, resize_height=320):
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
                croped = PreProcess.__resize(PreProcess.__crop(image, crop_ratio), resize_width, resize_height)
                last_index = file.rfind(".")
                origin_name = file[0:last_index]
                output_file_name = os.path.join(out_directory_path,
                                                str(origin_name) + "_crop_" + str(crop_ratio) + ".png")
                cv2.imwrite(output_file_name, croped)
                print("Saved: ", output_file_name)

    @staticmethod
    def __rotate(image, angle, center=None, scale=1.0):
        h, w = image.shape[:2]
        if center is None:
            center = (w / 2, h / 2)

        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, matrix, (w, h))

        return rotated

    @staticmethod
    def __crop(image, crop_ratio):
        h, w = image.shape[:2]
        return image[int(w * ((1 - crop_ratio) / 2)):int(w * (1 - (1 - crop_ratio) / 2)),
               int(h * ((1 - crop_ratio) / 2)):int(h * (1 - (1 - crop_ratio) / 2))]

    @staticmethod
    def __resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        h, w = image.shape[:2]

        if width is None and height is None:
            dim = (w, h)

        elif width is None:
            r = height / float(h)
            dim = (int(w * r), height)

        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=inter)


class SeedlingsData(object):

    def __init__(self):
        self._data = []

        self.paths = []
        self.batch_size = 128

        # define iter index
        self.iter_index = 0

    def load(self, train_data_paths: list, train=True, shuffle=True):

        """
        Load data and save in a list like this:
        the element's list contains two string,one is the file path of an image, one is the class of the image.
        [['D:\\Project\\Space\\Python\\Data\\seedings_data\\train\\train\\Black-grass\\0050f38b3.png', 'Black-grass']]
        :param train_data_paths: path of the data
        :param train: whether is the train data
        :param shuffle: should shuffle
        :return:
        """
        self.data.clear()
        self.paths = train_data_paths
        self.iter_index = 0

        if train:
            for data_path in train_data_paths:
                directories = os.listdir(data_path)
                for directory in directories:
                    for root, dirs, files in os.walk(os.path.join(data_path, directory)):
                        for file in files:
                            # Save file path and class
                            self.data.append([os.path.join(root, file), directory])
        else:
            for data_path in train_data_paths:
                for root, dirs, files in os.walk(os.path.join(data_path, data_path)):
                    for file in files:
                        # Save file path and undefined class
                        self.data.append([os.path.join(root, file), ""])
        if shuffle:
            random.shuffle(self.data)

    def set_batch_size(self, size):
        """
        Set the batch size for training
        :param size:
        :return:
        """
        self.batch_size = size

    @property
    def data(self) -> list:
        return self._data

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_index + self.batch_size >= len(self.data):
            self.iter_index = 0
            raise StopIteration("Iter end")

        batch_data = self.data[self.iter_index: self.iter_index + self.batch_size]
        batch_images = [cv2.imread(image[0]) for image in batch_data]
        batch_labels = [image[1] for image in batch_data]
        self.iter_index += self.batch_size
        return batch_images, batch_labels

    def generate(self):
        current_index = 0
        while current_index + self.batch_size < len(self.data):
            batch_data = self.data[current_index: current_index + self.batch_size]
            batch_images = [cv2.imread(image[0]) for image in batch_data]
            batch_labels = [image[1] for image in batch_data]

            yield batch_images, batch_labels
            current_index += self.batch_size


if __name__ == "__main__":
    # Replace with your directory
    test_file_path = r"D:\Project\Space\Python\Data\seedings_data\train\train"
    test_output_resize_file_path = r"D:\Project\Space\Python\Data\seedings_data\train\resize"
    test_output_rotate_file_path = r"D:\Project\Space\Python\Data\seedings_data\train\rotate"
    test_output_crop_file_path = r"D:\Project\Space\Python\Data\seedings_data\train\crop"

    # First we should pre-process the image data
    # Resize
    preprocess = PreProcess()
    preprocess.resize(test_file_path, test_output_resize_file_path)

    # Rotate
    preprocess.rotate(test_output_resize_file_path, test_output_rotate_file_path)

    # Crop
    preprocess.crop(test_output_resize_file_path, test_output_crop_file_path)

    # After pre-processing, we need to input data for training
    data = SeedlingsData()
    data.load([test_output_resize_file_path, test_output_rotate_file_path, test_output_crop_file_path])

    print(len(data.data))
    print((data.data[0]))

    # Iterate method 1
    # for images, labels in data:
    #     print(type(images))
    #     print(labels)

    # Iterate method 2
    # for images, labels in data.generate():
    #     print(type(images))
    #     print(labels)
