#!/usr/bin/env python3

import argparse
import yaml

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def estimate_homography(world_points, image_points):
    """
    Estimate homography from world points to image points.

    :param world_points: world points as numpy array
    :param image_points: image points as numpy array
    :return: homography as numpy array
    """

    H = cv2.findHomography(world_points, image_points)[0]

    return H


def visualize_homography(img, world_points, image_points, H):
    """
    Visualize homography.

    :param img: image as numpy array
    :param world_points: world points as numpy array
    :param image_points: image points as numpy array
    :param H: homography as numpy array
    :return: None
    """

    # project world_points to image using homography
    projected_points = cv2.perspectiveTransform(np.array([world_points]), H)[0]

    # print homography
    print("Homography: \n{}".format(H))

    # plot image with projected points
    plt.imshow(img)
    plt.scatter(image_points[:, 0], image_points[:, 1], c="r", s=100)
    plt.scatter(projected_points[:, 0], projected_points[:, 1], c="b", s=100, marker="x")
    plt.legend(["image points", "projected points"])
    plt.title("Homography Estimation")
    plt.show() 


def save_homography(H, filename="../homography/homography.npy"):
    """
    Save homography to .npy file.

    :param H: homography as numpy array
    :param filename: path to .npy file
    :return: None
    """

    np.save(filename, H)
    print("\nHomography saved as {}".format(filename))


if __name__ == "__main__":
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-p",
                        "--point_correspondences",
                        type=str,
                        help="path to .yaml file with point correspondences",
                        default="../homography/point_correspondences.yaml")
    parser.add_argument("-i",
                        "--image",
                        type=str,
                        help="path to image",
                        default="../homography/cuboid_10-2-284.png")
    parser.add_argument("-s",
                        "--save",
                        type=str,
                        help="save homography to .npy file",
                        default="../homography/homography.npy")
    args = parser.parse_args()

    # read point correspondences
    with open(args.point_correspondences, "r") as f:
        point_correspondences = yaml.load(f, Loader=yaml.FullLoader)
    world_points = np.array(point_correspondences["world_points"], dtype=np.float32)
    image_points = np.array(point_correspondences["image_points"], dtype=np.float32)

    # read image
    img = Image.open(args.image)
    img = np.array(img.convert("RGB"))

    # calculate homography
    H = estimate_homography(world_points, image_points)

    # visualize point correspondences
    visualize_homography(img, world_points, image_points, H)

    # save homography
    save_homography(H, filename=args.save)
