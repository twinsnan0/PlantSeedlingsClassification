import json
import os

from kaggle_data.downloader import KaggleDataDownloader

import constants


def download(competition_name, data_file_name, directory=None, file_name="test"):
    if directory is None:
        directory = os.getcwd()

    file_path = os.path.join(directory, file_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    print(os.path.abspath(file_path))

    # We can not download the data without user info
    username, password = get_user_info()
    print("username:{}, password:{}".format(username, password))
    downloader = KaggleDataDownloader(username, password, competition_name)

    download_file_path = os.path.join(directory, data_file_name)

    if os.path.exists(download_file_path):
        print("Data exists")
    else:
        downloader.download_dataset(data_file_name, directory)

    downloader.decompress(download_file_path, file_path)


def get_user_info() -> (str, str):
    # Warning: you can not push the user_info.json file to github
    with open('user_info.json') as json_file:
        data = json.load(json_file)
        return data["username"], data["password"]


if __name__ == "__main__":
    # Download and decompress data set

    download(constants.train_data_name, "sample_submission.csv.zip", r"../PlantSeedlingsImage/",
             "sample_submission.csv")
    download(constants.train_data_name, "test.zip", r"../PlantSeedlingsImage/", "test")
    # download(constants.train_data_name, "train.zip", r"../PlantSeedlingsImage/", "train")
