#!/usr/bin/env python3

import os
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


def sort_files(folder):
    """
    Sort files by creation time.

    :param folder: folder containing the files
    :return: list of files sorted by creation time
    """

    def extract_timestamp(file_name):
        timestamp_str = os.path.splitext(file_name)[0]  # Remove the extension
        timestamp_format = "%Y%m%d-%H%M%S"
        return datetime.strptime(timestamp_str, timestamp_format)

    # sort files by timestamp in filename
    files = os.listdir(folder)
    files = sorted(files, key=extract_timestamp)

    return files


def visualize_data(folder):
    """
    Visualize the raw data.

    :param folder: folder containing the data
    :return: None
    """

    # create a new figure
    plt.figure(figsize=(8, 5))
    plt.ion()
    plt.show()

    #files = sort_files(folder)
    files = os.listdir(folder)

    # iterate over all files in the subfolder
    for file in files:

        # check that the file ends with .npy
        if not file.endswith(".npy"):
            continue

        # load data
        data = np.load(os.path.join(folder, file), allow_pickle=True).item()

        # plot image
        plt.title(folder)
        plt.imshow(data["gs_img"].astype(np.uint8))

        # move image to the left in figure
        plt.subplots_adjust(left=0.05, right=0.75, top=0.9, bottom=0.1)

        # print force values below the image
        plt.text(340, 10, "f_x: {} N".format(round(data["f_x"], 4)), color="black", fontsize=12)
        plt.text(340, 30, "f_y: {} N".format(round(data["f_y"], 4)), color="black", fontsize=12)
        plt.text(340, 50, "f_z: {} N".format(round(data["f_z"], 4)), color="black", fontsize=12)
        plt.text(340, 70, "m_x: {} Nm".format(round(data["m_x"], 4)), color="black", fontsize=12)
        plt.text(340, 90, "m_y: {} Nm".format(round(data["m_y"], 4)), color="black", fontsize=12)
        plt.text(340, 110, "m_z: {} Nm".format(round(data["m_z"], 4)), color="black", fontsize=12)
        plt.text(340, 130, "x_0: {} mm".format(data["x_0"]), color="black", fontsize=12)
        plt.text(340, 150, "y_0: {} mm".format(data["y_0"]), color="black", fontsize=12)
        plt.text(340, 170, "z_0: {} mm".format(data["z_0"]), color="black", fontsize=12)
        plt.text(340, 190, "d_x: {} mm".format(data["d_x"]), color="black", fontsize=12)
        plt.text(340, 210, "d_y: {} mm".format(data["d_y"]), color="black", fontsize=12)
        plt.text(340, 230, "d_z: {} mm".format(data["d_z"]), color="black", fontsize=12)

        # pause until the user presses a key
        plt.pause(0.1)
        plt.waitforbuttonpress()

        # reset plot
        plt.clf()


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="Visualize raw data.")
    parser.add_argument("--folder", type=str, default="data", help="folder containing the data")
    args = parser.parse_args()

    # visualize data
    visualize_data(args.folder)
