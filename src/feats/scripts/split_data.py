import os
import numpy as np


def split_train_data(train_data_folder):

    train_data_files = os.listdir(train_data_folder)
    num_train_data = len(train_data_files)

    if ".gitingore" in train_data_files:
        train_data_files.remove(".gitignore")

    # split data into training, validation and test data
    train_data = np.random.choice(num_train_data, int(0.85*num_train_data), replace=False)
    validation_data = np.random.choice(list(set(range(num_train_data)) - set(train_data)), int(0.05*num_train_data), replace=False)
    test_data = list(set(range(num_train_data)) - set(train_data) - set(validation_data))

    print("Number of training data: {}".format(len(train_data)))
    print("Number of validation data: {}".format(len(validation_data)))
    print("Number of test data: {}".format(len(test_data)))

    # copy files in respecitve folders
    for i in validation_data:
        os.system("cp {} {}".format(train_data_folder + train_data_files[i], "../data/labels/val/" + train_data_files[i]))
        os.system("rm {}".format(train_data_folder + train_data_files[i]))

    for i in test_data:
        os.system("cp {} {}".format(train_data_folder + train_data_files[i], "../data/labels/test/" + train_data_files[i]))
        os.system("rm {}".format(train_data_folder + train_data_files[i]))


if __name__ == "__main__":

    train_data_folder = "../data/labels/train/"
    split_train_data(train_data_folder)
