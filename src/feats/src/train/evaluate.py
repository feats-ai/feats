import os.path
import sys; sys.path.append("../")
import argparse
import yaml
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

from data.dataloader import FEATSDataset, normalize, unnormalize
from models.unet import UNet


def main(config):
    """
    Main function for evaluating of the UNet. This function loads the model,
    creates a test dataset and iterates over the dataset to get the predictions
    of the model. The predictions are then plotted against the ground truth.

    :param config: dictionary containing the configuration for evaluation
    :return: None
    """

    # create dataset
    test_dataset = FEATSDataset(config["test_data"], config["norm_file"], normalize, train=False)

    # create data loader for test dataset
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # initialize network
    model = UNet(enc_chs=config["enc_chs"], dec_chs=config["dec_chs"], out_sz=config["output_size"])
    model.load_state_dict(torch.load(config["model"], map_location=torch.device("cpu"), weights_only=True))
    model.eval()

    # load calibration file
    if config["calibration_file"] is not None:
        calibration = np.load(config["calibration_file"])
        rows, cols = 240, 320
        M = np.float32([[1, 0, calibration[0]], [0, 1, calibration[1]]])

    # keep track of the mae for each channel
    criterion_MAE = torch.nn.L1Loss()
    mae_fx = []; mae_fy = []; mae_fz = []
    mae_fx_guf = []; mae_fy_guf = []; mae_fz_guf = []

    # disable gradient computation and reduce memory consumption
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            
            # prepare input data
            if config["calibration_file"] is not None:
                inputs_prewarp = data["gs_img"].numpy().squeeze(0)
                inputs_warp = cv2.warpAffine(inputs_prewarp,  M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
                inputs = torch.from_numpy(inputs_warp).permute(2, 0, 1).unsqueeze(0)
            else:
                inputs = data["gs_img"].permute(0, 3, 1, 2)

            # get model prediction
            outputs = model(inputs)

            # unnormalize gt data
            gt_grid_x = unnormalize(data["grid"][:, :, :, 0].squeeze(0), "grid_x", config["norm_file"])
            gt_grid_y = unnormalize(data["grid"][:, :, :, 1].squeeze(0), "grid_y", config["norm_file"])
            gt_grid_z = unnormalize(data["grid"][:, :, :, 2].squeeze(0), "grid_z", config["norm_file"])
            gt_total_force_x = unnormalize(data["total_force"][:, 0].squeeze(0), "f_x", config["norm_file"])
            gt_total_force_y = unnormalize(data["total_force"][:, 1].squeeze(0), "f_y", config["norm_file"])
            gt_total_force_z = unnormalize(data["total_force"][:, 2].squeeze(0), "f_z", config["norm_file"])

            # unnormalize output data
            outputs = outputs.squeeze(0).permute(1, 2, 0)
            pred_grid_x = unnormalize(outputs[:, :, 0], "grid_x", config["norm_file"])
            pred_grid_y = unnormalize(outputs[:, :, 1], "grid_y", config["norm_file"])
            pred_grid_z = unnormalize(outputs[:, :, 2], "grid_z", config["norm_file"])

            # calculate mae for each channel of the total force
            z_threshold = -40
            if gt_total_force_z > z_threshold:
                mae_fx.append(criterion_MAE(gt_total_force_x, torch.sum(pred_grid_x, dim=[0, 1])))
                mae_fy.append(criterion_MAE(gt_total_force_y, torch.sum(pred_grid_y, dim=[0, 1])))
                mae_fz.append(criterion_MAE(gt_total_force_z, torch.sum(pred_grid_z, dim=[0, 1])))
                mae_fx_guf.append(criterion_MAE(gt_grid_x, pred_grid_x))
                mae_fy_guf.append(criterion_MAE(gt_grid_y, pred_grid_y))
                mae_fz_guf.append(criterion_MAE(gt_grid_z, pred_grid_z))
                                       
    # print mean mae for each channel
    print("Mean MAE for x-axis: {:.4f} +/- {:.4f}".format(torch.mean(torch.stack(mae_fx)), torch.std(torch.stack(mae_fx))))
    print("Mean MAE for y-axis: {:.4f} +/- {:.4f}".format(torch.mean(torch.stack(mae_fy)), torch.std(torch.stack(mae_fy))))
    print("Mean MAE for z-axis: {:.4f} +/- {:.4f}".format(torch.mean(torch.stack(mae_fz)), torch.std(torch.stack(mae_fz))))

    print("Mean MAE for x-axis (GUF): {:.4f} +/- {:.4f}".format(torch.mean(torch.stack(mae_fx_guf)), torch.std(torch.stack(mae_fx_guf))))
    print("Mean MAE for y-axis (GUF): {:.4f} +/- {:.4f}".format(torch.mean(torch.stack(mae_fy_guf)), torch.std(torch.stack(mae_fy_guf))))
    print("Mean MAE for z-axis (GUF): {:.4f} +/- {:.4f}".format(torch.mean(torch.stack(mae_fz_guf)), torch.std(torch.stack(mae_fz_guf))))


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="Visualize UNet")
    parser.add_argument("-c", "--config", type=str, default="visualize_config.yaml", help="Path to config file for visualization")
    args = parser.parse_args()

    # load config file
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    # visualize network predictions
    main(config)
