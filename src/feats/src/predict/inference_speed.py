import os.path
import sys; sys.path.append("../")
import argparse
import yaml
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

from data.dataloader import FEATSDataset, normalize, unnormalize
from models.unet import UNet


def setup_device():
    """
    Sets up the device to be used for training.

    :return: device
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device: {}".format(device))

    return device


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="Live prediction of UNet")
    parser.add_argument("-c", "--config", type=str, default="predict_config.yaml", help="Path to config file for live prediction.")
    args = parser.parse_args()

    # load config file
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    # setup device
    device = setup_device()

    #  initialize network
    model = UNet(enc_chs=config["enc_chs"], dec_chs=config["dec_chs"], out_sz=config["output_size"])
    model.load_state_dict(torch.load(config["model"], map_location=torch.device("cpu"), weights_only=True))
    model.eval()
    model.to(device)

    # prepare dummy input
    dummy_input = torch.randn(1, 3, 240, 320, dtype=torch.float).to(device)

    # init loggers
    if device == torch.device("cuda") or device == torch.device("cpu"):
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    elif device == torch.device("mps"):
        starter, ender = torch.mps.event.Event(enable_timing=True), torch.mps.event.Event(enable_timing=True)

    # measure performance over multiple repetitions
    repetitions = 300
    timings=np.zeros((repetitions,1))

    # GPU warm-up
    for _ in range(10):
        _ = model(dummy_input)

    # start measuring
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()

            # wait for GPU sync
            if device == torch.device("cuda") or device == torch.device("cpu"):
                torch.cuda.synchronize()
            elif device == torch.device("mps"):
                torch.mps.synchronize()

            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)

    print("Mean Inference Time: ", mean_syn, " +/- ", std_syn, " ms")
    print("Mean Inference Frequency: ", 1000.0 / mean_syn, " Hz")
