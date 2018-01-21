import os
from kaggle_data.downloader import KaggleDataDownloader

import constants


# To run this code please install the Kaggle-data-downloader
# pip install -U git+https://github.com/EKami/kaggle-data-downloader.git

def download(user_pwd, competition_name, data_file_name, directory=None, file_name=""):
    if directory is None:
        directory = os.getcwd()

    file_path = os.path.join(directory, file_name)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    print(os.path.abspath(file_path))

    # We can not download the data without user info
    downloader = KaggleDataDownloader(user_pwd[0], user_pwd[1], competition_name)

    download_file_path = os.path.join(directory, data_file_name)
    if os.path.exists(download_file_path):
        print("Data exists")
    else:
        downloader.download_dataset(data_file_name, directory)

        downloader.decompress(download_file_path, file_path)


def get_user_info() -> (str, str):
    # require user's input
    username = input("Please enter username for Kaggle: ")
    password = input("Password: ")
    return username, password


if __name__ == "__main__":
    # Download and decompress data set
    user_info = get_user_info()

    download(user_info, constants.train_data_name, constants.train_data_sample_submission + ".zip",
             constants.test_file_directory,
             constants.train_data_sample_submission)
    download(user_info, constants.train_data_name, constants.train_data_test + ".zip", constants.test_file_directory,
             constants.train_data_test)
    download(user_info, constants.train_data_name, constants.train_data_train + ".zip", constants.test_file_directory,
             constants.train_data_train)
