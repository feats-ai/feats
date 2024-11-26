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
    Main function for visualization of the UNet. This function loads the model,
    creates a test dataset and iterates over the dataset to get the predictions
    of the model. The predictions are then plotted against the ground truth.

    :param config: dictionary containing the configuration for visualization
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

    # disable gradient computation and reduce memory consumption
    with torch.no_grad():
        for i, data in enumerate(test_loader):

            # prepare input data
            inputs = data["gs_img"].permute(0, 3, 1, 2)

            # get model prediction
            outputs = model(inputs)

            # unnormalize gt data
            gt_grid_x = unnormalize(data["grid"][:, :, :, 0].squeeze(0), "grid_x", config["test_data"], config["norm_file"])
            gt_grid_y = unnormalize(data["grid"][:, :, :, 1].squeeze(0), "grid_y", config["test_data"], config["norm_file"])
            gt_grid_z = unnormalize(data["grid"][:, :, :, 2].squeeze(0), "grid_z", config["test_data"], config["norm_file"])
            gt_total_force_x = unnormalize(data["total_force"][:, 0].squeeze(0), "f_x", config["test_data"], config["norm_file"])
            gt_total_force_y = unnormalize(data["total_force"][:, 1].squeeze(0), "f_y", config["test_data"], config["norm_file"])
            gt_total_force_z = unnormalize(data["total_force"][:, 2].squeeze(0), "f_z", config["test_data"], config["norm_file"])

            # unnormalize output data
            outputs = outputs.squeeze(0).permute(1, 2, 0)
            pred_grid_x = unnormalize(outputs[:, :, 0], "grid_x", config["test_data"], config["norm_file"])
            pred_grid_y = unnormalize(outputs[:, :, 1], "grid_y", config["test_data"], config["norm_file"])
            pred_grid_z = unnormalize(outputs[:, :, 2], "grid_z", config["test_data"], config["norm_file"])

            # create figure
            fig = plt.figure()
            fig.set_size_inches(12, 7)

            # create grid for 3x2 subplots
            grid = AxesGrid(fig, 121,
                    nrows_ncols=(2, 3),
                    axes_pad=0.50,
                    share_all=True,
                    label_mode="all",
                    cbar_location="bottom",
                    cbar_mode="edge",
                    cbar_pad=0.25,
                    cbar_size="15%",
                    direction="column"
                    )

            # set color limits
            clim_xy = [-0.07, 0.07]
            clim_z = [-0.3, 0.0]

            # plot each channel
            gt_x = grid[0].imshow(gt_grid_x, cmap="viridis")
            gt_x.set_clim(clim_xy)
            pred_x = grid[1].imshow(pred_grid_x, cmap="viridis")
            pred_x.set_clim(clim_xy)
            grid.cbar_axes[0].colorbar(gt_x)

            gt_y = grid[2].imshow(gt_grid_y, cmap="viridis")
            gt_y.set_clim(clim_xy)
            pred_y = grid[3].imshow(pred_grid_y, cmap="viridis")
            pred_y.set_clim(clim_xy)
            grid.cbar_axes[1].colorbar(gt_y)

            gt_z = grid[4].imshow(gt_grid_z, cmap="viridis")
            gt_z.set_clim(clim_z)
            pred_z = grid[5].imshow(pred_grid_z, cmap="viridis")
            pred_z.set_clim(clim_z)
            grid.cbar_axes[2].colorbar(gt_z)

            # set labels for each axis
            j = 0
            direction = ["x", "y", "z"]
            for cax in grid.cbar_axes:
                cax.axis[cax.orientation].set_label("{}-axis [N]".format(direction[j%3]))
                j += 1

            # adjust the size of the plot
            plt.subplots_adjust(left=-.52, right=4)

            # use total force as title for each channel
            grid[0].set_title("{:.2f}N".format(gt_total_force_x))
            grid[1].set_title("{:.2f}N".format(torch.sum(pred_grid_x, dim=[0, 1])))
            grid[2].set_title("{:.2f}N".format(gt_total_force_y))
            grid[3].set_title("{:.2f}N".format(torch.sum(pred_grid_y, dim=[0, 1])))
            grid[4].set_title("{:.2f}N".format(gt_total_force_z))
            grid[5].set_title("{:.2f}N".format(torch.sum(pred_grid_z, dim=[0, 1])))

            # set titel for the rows
            grid[0].set_ylabel("Ground Truth", fontsize=12)
            grid[1].set_ylabel("Prediction", fontsize=12)

            # show image and make it bigger
            gs_img = data["gs_img"].numpy().squeeze(0)
            gs_img = cv2.cvtColor(gs_img, cv2.COLOR_BGR2RGB)
            gs_img = cv2.resize(gs_img, (int(320*1.5), int(240*1.5)))
            cv2.imshow("GelSight Mini", gs_img)
            cv2.moveWindow("GelSight Mini", 1320, 400)

            # show plot and close it after key press
            plt.show(block=False)
            plt.pause(0.1)
            plt.waitforbuttonpress()
            plt.close()

            # close image
            cv2.destroyAllWindows()


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="Visualize UNet")
    parser.add_argument("-c", "--config", type=str, default="visualize_config.yaml", help="Path to config file for visualization")
    args = parser.parse_args()

    # load config file
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    # visualize network predictions
    main(config)
