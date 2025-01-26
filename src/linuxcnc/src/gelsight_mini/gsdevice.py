#!/usr/bin/env python3

import cv2
import numpy as np


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


if __name__ == "__main__":
    
    # create video capture object
    cam = cv2.VideoCapture(0)

    while True:
        # capture image
        img = capture_image(cam)

        # convert rgb to bgr and scale up for better visualization
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_bgr = cv2.resize(img_bgr, (640, 480))

        # display image with red dot at the center
        img_bgr = cv2.circle(img_bgr, (320, 240), 5, (0, 0, 255), -1)
        cv2.imshow("GelSight Mini", img_bgr)
        cv2.waitKey(1)