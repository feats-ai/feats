#!/usr/bin/env python3

import argparse
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt


def capture_image(cam, imgw=320, imgh=240):
    """
    Capture image from GelSight Mini and process it.

    :param cam: Video capture object
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


def calibrate_sensor(args):
    """
    Calibrate a new sensor by calculating the displacement of dots between two sensors.
    
    :param args: args containing the paths to the no contact samples of the train and test sensor
    :return: None
    """

    # load no contact samples of train and test sensor
    train_data = np.load(args.nc_train, allow_pickle=True).item()
    
    if args.nc_test is None:
        # capture image from camera of connected sensor
        cam = cv2.VideoCapture(0)
        gs_img = capture_image(cam)
        test_data = {"gs_img": gs_img}
    else:
        test_data = np.load(args.nc_test, allow_pickle=True).item()

    # convert images to grayscale
    train_image = cv2.cvtColor(train_data["gs_img"], cv2.COLOR_RGB2GRAY)
    test_image = cv2.cvtColor(test_data["gs_img"], cv2.COLOR_RGB2GRAY)

    # blur images
    blurred_train_image = cv2.GaussianBlur(train_image, (7, 7), 2)
    blurred_test_image = cv2.GaussianBlur(test_image, (7, 7), 2)

    # detect black points in image
    _, black_train = cv2.threshold(blurred_train_image, 55, 255, cv2.THRESH_BINARY_INV)
    _, black_test = cv2.threshold(blurred_test_image, 55, 255, cv2.THRESH_BINARY_INV)

    # detect clusters in image and draw bounding boxes
    _, labels_train = cv2.connectedComponents(black_train)
    _, labels_test = cv2.connectedComponents(black_test)
    centers_train = []; centers_test = []
    
    for label in range(1, labels_train.max() + 1):
        mask = np.array(labels_train == label, dtype=np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center, radius = cv2.minEnclosingCircle(contours[0])
        centers_train.append(center)
        cv2.circle(train_image, (int(center[0]), int(center[1])), int(radius), (255, 0, 0), 1)

    for label in range(1, labels_test.max() + 1):
        mask = np.array(labels_test == label, dtype=np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center, radius = cv2.minEnclosingCircle(contours[0])
        centers_test.append(center)
        cv2.circle(test_image, (int(center[0]), int(center[1])), int(radius), (255, 0, 0), 1)

    # print number of clusters
    print("Number of clusters in train sensor image:", len(centers_train))
    print("Number of clusters in test sensor image:", len(centers_test))

    # create figure with subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(train_image, cmap="gray")
    axs[0].set_title("Train Sensor")
    axs[1].imshow(test_image, cmap="gray")
    axs[1].set_title("Test Sensor")
    plt.show()

    # calculate displacement of dots
    vec = []
    for center in centers_test:
        distances = np.linalg.norm(np.array(centers_train) - center, axis=1)
        closest_center = centers_train[np.argmin(distances)]
        vec.append(np.array(closest_center) - np.array(center))
        # draw line between corresponding clusters
        cv2.line(test_image, (int(center[0]), int(center[1])), (int(closest_center[0]), int(closest_center[1])), (255, 0, 0), 1)

    # create figure with subplots
    plt.imshow(test_image, cmap="gray")
    plt.title("Displacement of Dots in Test Sensor")
    plt.show()

    # extract mean vector and save calibration data
    mean_vec = np.mean(vec, axis=0)
    print("Mean displacement vector:", mean_vec)
    np.save("../calibration/" + time.strftime("%Y%m%d-%H%M%S") + ".npy", mean_vec)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="Calibrate a new sensor by calculating the displacement of dots between two sensors.")
    parser.add_argument("--nc_train", type=str, default="../calibration/nocontact_samples/nocontact_train_sensor.npy", help="Path to the no contact sample of the train sensor.")
    parser.add_argument("--nc_test", type=str, help="Path to the no contact sample of the test sensor.")
    args = parser.parse_args()

    calibrate_sensor(args)