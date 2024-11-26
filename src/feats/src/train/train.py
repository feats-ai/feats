import sys; sys.path.append("../")
import os
import argparse
import yaml
from datetime import datetime

import torch
torch.multiprocessing.set_sharing_strategy("file_system")
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb
wandb.require("core")

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


def train_one_epoch(model, optimizer, epoch_index, run, train_loader, device):
    """
    Trains the Unet for one epoch.

    :param model: Unet model
    :param optimizer: optimizer
    :param epoch_index: index of the current epoch
    :param run: wandb logger
    :param train_loader: data loader for training data
    :param device: device to use for training
    :return: average loss over the last 10 batches
    """

    # initialize running and last loss
    running_loss = 0.
    last_loss = 0.

    # loop over the dataset multiple times (one epoch)
    for i, data in enumerate(train_loader):

        inputs = data["gs_img"].permute(0, 3, 1, 2).to(device)
        gt_grid = data["grid"].permute(0, 3, 1, 2).to(device)
        gt_total_force = data["total_force"].to(device)

        # zero gradients for every batch
        optimizer.zero_grad()

        # make predictions for this batch
        outputs = model(inputs)

        # compute the loss and its gradients
        loss = criterion(outputs, gt_grid, gt_total_force)
        loss.backward()

        # adjust learning weights
        optimizer.step()

        # gather data and report
        running_loss += loss.item()
        if (i+1) % 10 == 0:
            last_loss = running_loss / 10  # loss averaged over the last 10 batches
            print("  batch {} loss: {}".format(i + 1, last_loss))

            wb_ct = epoch_index * len(train_loader) + i + 1
            run.log({"epoch": epoch_index, "train-loss": last_loss}, step=wb_ct)

            # reset running loss
            running_loss = 0.

    return last_loss


def criterion(outputs, gt_grid, gt_total_force):
    """
    Computes the loss for the given outputs and ground truth values.

    :param outputs: output tensor of the network
    :param gt_grid: ground truth grid
    :param gt_total_force: ground truth total force
    :return: loss
    """

    criterion_MSE = torch.nn.MSELoss()
    loss = criterion_MSE(outputs, gt_grid)

    return loss


def loss_total_force(gt_total_force, outputs):
    """
    Calculate the loss for the total force in unnormalized space.

    :param gt_total_force: ground truth total force
    :param outputs: output of the network
    :return: loss_tf_x, loss_tf_y, loss_tf_z
    """

    # unnormalize the outputs
    outputs_transf = outputs.squeeze(0).permute(1, 2, 0)
    pred_grid_x = unnormalize(outputs_transf[:, :, 0], "grid_x", config["train_data"], config["norm_file"])
    pred_grid_y = unnormalize(outputs_transf[:, :, 1], "grid_y", config["train_data"], config["norm_file"])
    pred_grid_z = unnormalize(outputs_transf[:, :, 2], "grid_z", config["train_data"], config["norm_file"])

    # unnormalize the ground truth total force
    gt_total_force_x = unnormalize(gt_total_force[:, 0].squeeze(0), "f_x", config["train_data"], config["norm_file"])
    gt_total_force_y = unnormalize(gt_total_force[:, 1].squeeze(0), "f_y", config["train_data"], config["norm_file"])
    gt_total_force_z = unnormalize(gt_total_force[:, 2].squeeze(0), "f_z", config["train_data"], config["norm_file"])

    # compute the total force for the predictions and ground truth
    pred_total_force_x = torch.sum(pred_grid_x, dim=(0, 1))
    pred_total_force_y = torch.sum(pred_grid_y, dim=(0, 1))
    pred_total_force_z = torch.sum(pred_grid_z, dim=(0, 1))

    # calculate the error for the total force
    criterion_MAE = torch.nn.L1Loss()
    loss_tf_x = criterion_MAE(pred_total_force_x, gt_total_force_x)
    loss_tf_y = criterion_MAE(pred_total_force_y, gt_total_force_y)
    loss_tf_z = criterion_MAE(pred_total_force_z, gt_total_force_z)

    return loss_tf_x, loss_tf_y, loss_tf_z


def main(config):
    """
    Main function for training the UNet. This functions sets up the device, creates the datasets and data loaders,
    initializes the network, criterion and optimizer, and then trains the network for a number of epochs. After
    each epoch, the validation loss is computed and the model's state is saved if the validation loss is lower
    than the best validation loss so far.

    References:
    - https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

    :param config: configuration dictionary
    :return: None
    """

    # set device
    device = setup_device()

    # create datasets
    train_dataset = FEATSDataset(config["train_data"], config["norm_file"], normalize, train=True)
    val_dataset = FEATSDataset(config["val_data"], config["norm_file"], normalize, train=False)

    # create data loaders for our datasets; shuffle for training, not for validation
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # initialize network and optimizer
    model = UNet(enc_chs=config["enc_chs"], dec_chs=config["dec_chs"], out_sz=config["output_size"]).to(device)

    # load pre-trained model
    if config["pretrained_model"]:
        model.load_state_dict(torch.load(config["pretrained_model"], map_location=device))
        print("Pre-trained model loaded: {}".format(config["pretrained_model"]))

    #optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], weight_decay = 0.001, momentum = 0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, "min")

    # create a timestamped subdirectory for this run
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    run = wandb.init(project="forcesense", name="unet_{}".format(timestamp),
                     dir=config["run_path"],
                     config={"norm_file": config["norm_file"],
                             "enc_chs": config["enc_chs"],
                             "dec_chs": config["dec_chs"],
                             "output_size": config["output_size"],
                             "batch_size": config["batch_size"],
                             "epochs": config["epochs"],
                             "learning_rate": config["learning_rate"]
                            }
                    )

    # create directory for model
    os.mkdir(config["model_path"] + "unet_{}".format(timestamp))

    # set number of epochs
    EPOCHS = config["epochs"]

    # track the best validation loss, and the current epoch number
    best_vloss = 1_000_000.
    epoch_number = 0

    # loop over the dataset multiple times
    for epoch in range(EPOCHS):
        print("EPOCH {}:".format(epoch_number + 1))

        # make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(model, optimizer, epoch_number, run, train_loader, device)

        # initilialize the validation loss for this epoch
        running_vloss = 0.0
        running_vloss_tf_x = 0.0
        running_vloss_tf_y = 0.0
        running_vloss_tf_z = 0.0

        # set the model to evaluation mode, disabling dropout and using population statistics for batch normalization.
        model.eval()

        # disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs = vdata["gs_img"].permute(0, 3, 1, 2).to(device)
                vgt_grid = vdata["grid"].permute(0, 3, 1, 2).to(device)
                vgt_total_force = vdata["total_force"].to(device)

                voutputs = model(vinputs)

                vloss = criterion(voutputs, vgt_grid, vgt_total_force)
                running_vloss += vloss.item()

                vloss_tf_x, vloss_tf_y, vloss_tf_z = loss_total_force(vgt_total_force, voutputs)
                running_vloss_tf_x += vloss_tf_x.item()
                running_vloss_tf_y += vloss_tf_y.item()
                running_vloss_tf_z += vloss_tf_z.item()

        # compute the average validation loss for this epoch
        avg_vloss = running_vloss / (i + 1)
        avg_vloss_tf_x = running_vloss_tf_x / (i + 1)
        avg_vloss_tf_y = running_vloss_tf_y / (i + 1)
        avg_vloss_tf_z = running_vloss_tf_z / (i + 1)
        print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

        # adjust learning rate
        scheduler.step(avg_vloss_tf_x + avg_vloss_tf_y + avg_vloss_tf_z)

        # log the running loss averaged per batch for both training and validation
        run.log({"epoch": epoch_number + 1, "avg-train-loss": avg_loss, "val-loss": avg_vloss})
        run.log({"epoch": epoch_number + 1, "val-MAE-tf-x": avg_vloss_tf_x, "val-MAE-tf-y": avg_vloss_tf_y, "val-MAE-tf-z": avg_vloss_tf_z})

        # track best performance, and save the model's state
        if avg_vloss_tf_x + avg_vloss_tf_y + avg_vloss_tf_z < best_vloss:
            best_vloss = avg_vloss_tf_x + avg_vloss_tf_y + avg_vloss_tf_z
            model_path = config["model_path"] + "unet_{}/".format(timestamp) + "unet_{}_{}.pt".format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

    run.finish()

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="Train UNet")
    parser.add_argument("-c", "--config", type=str, default="train_config.yaml", help="Path to config file for training")
    args = parser.parse_args()

    # load config file
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    # train network
    main(config)
