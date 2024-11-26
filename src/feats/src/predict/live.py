#!/usr/bin/env python3

import sys; sys.path.append("../")
import argparse
import yaml

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from data.dataloader import normalize, unnormalize
from models.unet import UNet


def capture_image(cam, imgw=320, imgh=240):
    """
    Capture image from GelSight Mini and process it.

    :param cam: video capture object
    :param imgw: width of the image
    :param imgh: height of the image
    :return: processed image
    """

    # capture image
    ret, f0 = cam.read()

    if ret:
        # resize, crop and resize back
        img = cv2.resize(f0, (895, 672))  # size suggested by janos to maintain aspect ratio
        border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(np.floor(img.shape[1] * (1 / 7)))  # remove 1/7th of border from each size
        img = img[border_size_x:img.shape[0] - border_size_x, border_size_y:img.shape[1] - border_size_y]
        img = img[:, :-1]  # remove last column to get a popular image resolution
        img = cv2.resize(img, (imgw, imgh))  # final resize for 3d
    else:
        print("ERROR! reading image from camera")

    # convert bgr to rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def make_prediction(img, model, device, config):
    """
    Make prediction using the unet model.

    :param img: input image
    :param model: unet model
    :param device: device to run the model
    :param config: configuration file
    :return: predicted grid x, y, z
    """

    # store data in dictionary
    data = {}
    data["gs_img"] = img

    # normalize data
    data = normalize(data, config["data_folder"], config["norm_file"])

    # convert to torch tensor
    gs_img = torch.from_numpy(data["gs_img"]).float()
    gs_img = gs_img.unsqueeze(0).permute(0, 3, 1, 2).to(device)

    outputs = model(gs_img)

    # unnormalize the outputs
    outputs_transf = outputs.squeeze(0).permute(1, 2, 0)
    pred_grid_x = unnormalize(outputs_transf[:, :, 0], "grid_x", config["data_folder"], config["norm_file"])
    pred_grid_y = unnormalize(outputs_transf[:, :, 1], "grid_y", config["data_folder"], config["norm_file"])
    pred_grid_z = unnormalize(outputs_transf[:, :, 2], "grid_z", config["data_folder"], config["norm_file"])

    # convert to numpy
    pred_grid_x = pred_grid_x.cpu().detach().numpy()
    pred_grid_y = pred_grid_y.cpu().detach().numpy()
    pred_grid_z = pred_grid_z.cpu().detach().numpy()

    return pred_grid_x, pred_grid_y, pred_grid_z


def animate(frame, cam, model, device, config, ims, axs):
    """
    Animate the prediction with matplotlib.

    :param frame: frame number
    :param cam: camera object
    :param mdoel: unet model
    :param device: device to run the model
    :param config: configuration dictionary
    :param ims: image objects
    :param axs: axis objects
    :return: None
    """

    # capture image and make prediction
    gs_img = capture_image(cam)
    pred_grid_x, pred_grid_y, pred_grid_z = make_prediction(gs_img, model, device, config)

    # update the image data
    ims[0].set_data(pred_grid_x)
    ims[1].set_data(pred_grid_y)
    ims[2].set_data(pred_grid_z)
    ims[3].set_data(gs_img.astype(np.uint8))

    # update titles with the new force values
    axs[0].set_title("x-force: {:.2f}N".format(np.sum(pred_grid_x)), fontsize=14)
    axs[1].set_title("y-force: {:.2f}N".format(np.sum(pred_grid_y)), fontsize=14)
    axs[2].set_title("z-force: {:.2f}N".format(np.sum(pred_grid_z)), fontsize=14)

    return None


def main(config):
    """
    Main function to run the live prediction.

    :param config: configuration dictionary containing model parameters and paths
    :return: None
    """

    # initialize camera
    cam = cv2.VideoCapture(0)

    # specify device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = UNet(enc_chs=config["enc_chs"], dec_chs=config["dec_chs"], out_sz=config["output_size"])
    model.load_state_dict(torch.load(config["model"], map_location=torch.device("cpu"), weights_only=True))
    model.eval().to(device)

    # Set up the figure and initial images
    fig, axs = plt.subplots(1, 4, figsize=(16, 5))
    fig.canvas.manager.set_window_title("FEATS")
    fig.suptitle("\nFEATS LIVE DEMO", fontsize=16)

    # reduce whitespace of plot
    plt.subplots_adjust(left=0.05, right=0.95, top=2.7, bottom=0.1)

    # capture first image of the camera
    gs_img = capture_image(cam)

    # make initial prediction
    pred_grid_x, pred_grid_y, pred_grid_z = make_prediction(gs_img, model, device, config)

    # hardcode the color limits
    clim_x = (-0.029, 0.029)
    clim_y = (-0.029, 0.029)
    clim_z = (-0.17, 0.0)

    # initialize the plots
    im_x = axs[0].imshow(pred_grid_x, origin="upper", vmin=clim_x[0], vmax=clim_x[1], animated=True)
    im_y = axs[1].imshow(pred_grid_y, origin="upper", vmin=clim_y[0], vmax=clim_y[1], animated=True)
    im_z = axs[2].imshow(pred_grid_z, origin="upper", vmin=clim_z[0], vmax=clim_z[1], animated=True)
    im_gs = axs[3].imshow(gs_img.astype(np.uint8), animated=True)

    # store the images in a list
    ims = [im_x, im_y, im_z, im_gs]

    # set titles
    axs[0].set_title("x-force: {:.2f}N".format(np.sum(pred_grid_x)), fontsize=14)
    axs[1].set_title("y-force: {:.2f}N".format(np.sum(pred_grid_y)), fontsize=14)
    axs[2].set_title("z-force: {:.2f}N".format(np.sum(pred_grid_z)), fontsize=14)
    axs[3].set_title("GelSight Mini Image", fontsize=14)

    # turn off axis
    axs[0].axis("off")
    axs[1].axis("off")
    axs[2].axis("off")
    axs[3].axis("off")

    # show colorbars
    fig.colorbar(im_x, ax=axs[0], orientation="horizontal", fraction=0.046, pad=0.01, ticks=[-0.02, -0.01, 0.0, 0.01, 0.02])
    fig.colorbar(im_y, ax=axs[1], orientation="horizontal", fraction=0.046, pad=0.01, ticks=[-0.02, -0.01, 0.0, 0.01, 0.02])
    fig.colorbar(im_z, ax=axs[2], orientation="horizontal", fraction=0.046, pad=0.01, ticks=[-0.15, -0.1, -0.05, 0.0])

    # add fake colorbar for axs[3] so that gs_img is on the same height
    fake_cbar = fig.colorbar(im_gs, ax=axs[3], orientation="horizontal", fraction=0.046, pad=0.01)
    fake_cbar.ax.set_visible(False)

    # animate the predictions
    ani = FuncAnimation(fig, animate, frames=None, interval=0, blit=False, fargs=(cam, model, device, config, ims, axs))

    plt.show()


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="Live prediction of UNet")
    parser.add_argument("-c", "--config", type=str, default="predict_config.yaml", help="Path to config file for live prediction.")
    args = parser.parse_args()

    # load config file
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    # make live prediction
    main(config)
