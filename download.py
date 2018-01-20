import os
from kaggle_data.downloader import KaggleDataDownloader

import constants

# To run this code please install the Kaggle-data-downloader
# pip install -U git+https://github.com/EKami/kaggle-data-downloader.git


def download(user_pw, competition_name, data_file_name, directory=None):
    if directory is None:
        directory = os.getcwd()

    # We can not download the data without user info
    downloader = KaggleDataDownloader(user_pw[0], user_pw[1], competition_name)

    download_file_path = os.path.join(directory, data_file_name)
    if os.path.exists(download_file_path):
        print("Data exists")
    else:
        downloader.download_dataset(data_file_name, directory)

    downloader.decompress(download_file_path, directory)


def get_user_info() -> (str, str):
    # require user's input
    username = input("Please enter username for Kaggle: ")
    password = input("Password: ")
    return username, password


if __name__ == "__main__":
    # Download and decompress data set
    user_info = get_user_info()
    #download(user_info, constants.train_data_name, "sample_submission.csv.zip", r"../PlantSeedlingsImage/","sample_submission.csv")
    #download(user_info, constants.train_data_name, "test.zip", r"../PlantSeedlingsImage/", "test")
    download(user_info, constants.train_data_name, "train.zip", r"../PlantSeedlingsImage/")
