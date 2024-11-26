import os
import argparse
from datetime import datetime
import numpy as np


def main(root_dir):
    """
    Main function for nomalization of the dataset. This function iterates over all
    files in the root directory and computes the min and max values for each key.
    The results are then saved in npy file in the root directory.

    :param root_dir: path to the dataset directory
    :return: None
    """

    files = os.listdir(root_dir)
    if ".gitignore" in files:
        files.remove(".gitignore")
    normalization = {}

    # iterate over all files
    for f in files:
        # load data from file
        data = np.load(str(root_dir + f), allow_pickle=True).item()

        # iterate over all keys in data
        for key in data.keys():
            # get the data for this key
            d = data[key]

            # get the min and max value
            min_val = np.min(d)
            max_val = np.max(d)

            # check if key is already in normalization
            if key not in normalization:
                # add key to normalization and set min and max values
                normalization[key] = {"min": min_val, "max": max_val}
            else:
                # update min and max values
                if min_val < normalization[key]["min"]:
                    normalization[key]["min"] = min_val
                if max_val > normalization[key]["max"]:
                    normalization[key]["max"] = max_val

    # save normalization
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    save_dir = os.path.dirname(os.path.dirname(root_dir)) + "/normalization_" + str(timestamp) + ".npy"
    np.save(save_dir, normalization)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="Normalize the dataset.")
    parser.add_argument("-f", "--folder", type=str, default="../data/labels/train/", help="Path to the dataset directory.")
    args = parser.parse_args()

    # run main function
    main(args.folder)
