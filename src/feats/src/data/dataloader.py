import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def normalize(data, norm_file):
    """
    Normalize the data in the FEATS dataset for the given range.

    :param data: dictionary containing the data
    :param norm_file: name of the normalization file
    :return: dictionary containing the normalized data
    """

    # define ranges for normalization
    range_xy = (-1, 1)
    range_z = (0, 1)

    # load normalization
    norm = np.load(norm_file, allow_pickle=True).item()

    # iterate over all keys in sample
    for key in data.keys():
        if key == "gs_img":
            # normalize data
            data[key] = data[key] / 255
        elif key == "grid_x" or key == "grid_y" or key == "f_x" or key == "f_y":
            # normalize data
            data[key] = (data[key] - norm[key]["min"]) / (norm[key]["max"] - norm[key]["min"])
            # scale data
            data[key] = data[key] * (range_xy[1] - range_xy[0]) + range_xy[0]
        elif key == "grid_z" or key == "f_z":
            # normalize data
            data[key] = (data[key] - norm[key]["min"]) / (norm[key]["max"] - norm[key]["min"])
            # scale data
            data[key] = data[key] * (range_z[1] - range_z[0]) + range_z[0]

    return data


def unnormalize(data, key, norm_file):
    """
    Unnormalize the data in the FEATS dataset for the given range.

    :param data: dictionary containing the data
    :param key: key of the data to unnormalize
    :param norm_file: name of the normalization file
    :return: dictionary containing the unnormalized data
    """

    # define ranges for normalization
    range_xy = (-1, 1)
    range_z = (0, 1)

    # load normalization
    norm = np.load(norm_file, allow_pickle=True).item()

    if key == "gs_img":
        # undo normalization
        data = data * 255
    elif key == "grid_x" or key == "grid_y" or key == "f_x" or key == "f_y":
        # undo normalization
        data = (data - range_xy[0]) / (range_xy[1] - range_xy[0])
        # undo scaling
        data = data * (norm[key]["max"] - norm[key]["min"]) + norm[key]["min"]
    elif key == "grid_z" or key == "f_z":
        # undo normalization
        data = (data - range_z[0]) / (range_z[1] - range_z[0])
        # undo scaling
        data = data * (norm[key]["max"] - norm[key]["min"]) + norm[key]["min"]

    return data


def augment(img):
    """
    Augment the image.

    :param img: image to be augmented
    :return: augmented image
    """

    # add Gaussian noise
    img = img + torch.randn(img.shape) * 0.05

    # convert image to PIL
    img = TF.to_pil_image(img.permute(2, 0, 1))

    # randomly change brightness, contrast, saturation and hue
    img = TF.adjust_brightness(img, random.uniform(0.75, 1.25))
    img = TF.adjust_contrast(img, random.uniform(0.75, 1.25))
    img = TF.adjust_saturation(img, random.uniform(0.75, 1.25))
    img = TF.adjust_hue(img, random.uniform(-0.1, 0.1))

    return TF.to_tensor(img).permute(1, 2, 0).float()


class FEATSDataset(Dataset):

    def __init__(self, root_dir, norm_file, transform=None, train=False):
        """
        Dataset class for the FEATS dataset.
        One sample is stored in one numpy array.

        :param root_dir: path to the dataset directory
        :param norm: normalization file
        :param transform: transformations to apply to the data
        :param train: boolean to indicate if the dataset is used for training
        :return: None
        """

        self.root_dir = root_dir  # path to the dataset directory
        self.norm_file = norm_file  # name of the normalization file
        self.files = os.listdir(self.root_dir)  # list of files in the dataset directory
        if ".gitignore" in self.files:
            self.files.remove(".gitignore")
        self.transform = transform  # transformations to apply to the data
        self.train = train


    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        :param None
        :return: int
        """

        return len(self.files)


    def __getitem__(self, idx):
        """
        Return a sample from the dataset.

        :param idx: index of the sample
        :return: dict
        """

        # load data from file
        data = np.load(str(self.root_dir + self.files[idx]), allow_pickle=True).item()

        # apply transformations
        if self.transform:
            data = self.transform(data, self.norm_file)

        if self.train:
            # augment every fifth sample
            if idx % 5 == 0:
                gs_img = augment(torch.from_numpy(data["gs_img"]).float())
            else:
                gs_img = torch.from_numpy(data["gs_img"]).float()
        else:
            gs_img = torch.from_numpy(data["gs_img"]).float()

        # create sample
        sample = {
            "gs_img": gs_img,
            "grid": torch.stack((torch.from_numpy(data["grid_x"]).float(), torch.from_numpy(data["grid_y"]).float(), torch.from_numpy(data["grid_z"]).float()), 2),
            "total_force": torch.stack((torch.from_numpy(np.array(data["f_x"])).float(), torch.from_numpy(np.array(data["f_y"])).float(), torch.from_numpy(np.array(data["f_z"])).float()), 0),
            "filename": self.files[idx]
        }

        return sample
