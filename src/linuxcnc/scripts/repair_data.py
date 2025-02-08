#!/usr/bin/env python3

import os
import argparse
import numpy as np


if __name__ == "__main__":
    
    # parse arguments
    parser = argparse.ArgumentParser(description="Repair raw data.")
    parser.add_argument("--folder", type=str, default="data", help="folder containing the data")
    args = parser.parse_args()
    
    # subfolders in the folder
    subfolders = [f.path for f in os.scandir(args.folder) if f.is_dir()]

    # extract indenter name from folder name
    indenter_name = os.path.basename(args.folder)

    for sf in subfolders:

        # files in subfolder
        files = os.listdir(sf)
        print("Starting to repair data in folder '{}'".format(sf))

        subfolder_name = os.path.basename(sf)
        deg = subfolder_name.split('_')[-1].replace('deg', '')

        for file in files:
            
            # check that the file ends with .npy
            if not file.endswith(".npy"):
                continue

            # load data
            data = np.load(os.path.join(sf, file), allow_pickle=True).item()

            # update indenter name
            data["indenter"] = indenter_name

            # add degree value
            data["deg"] = deg

            # save data
            np.save(os.path.join(sf, file), data)

        print("Finished repairing data in folder '{}'".format(sf))