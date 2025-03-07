#!usr/bin/env python3

import sys; sys.path.append("../")
from utils.RPC import Client

import argparse
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from resense_hex_21.hex21 import connect_to_sensor
from gelsight_mini.gsdevice import capture_image


def record_single_datapoint(serial_port:str="/dev/tty.usbmodem38A6337730371", indenter:str="None", folder:str="../raw_data/test/", x_0:float=0.0, y_0:float=0.0, z_0:float=0.0, d_x:float=0.0, d_y:float=0.0, d_z:float=0.0, deg:float=0.0) -> None:

    # connect to sensor and capture data point
    resense_hex21 = connect_to_sensor(serial_port)
    recording = resense_hex21.record_sample()

    # create video capture object
    cam = cv2.VideoCapture(0)

    # wait for camera to be ready
    while True:
        ret, _ = cam.read()
        if ret:
            break

    # capture image
    gs_img = capture_image(cam)

    # data to be saved
    data = {
        "gs_img": gs_img,
        "indenter": indenter,
        "x_0": x_0,
        "y_0": y_0,
        "z_0": z_0,
        "d_x": d_x,
        "d_y": d_y,
        "d_z": d_z,
        "deg": deg,
        "f_x": recording.force.x,
        "f_y": recording.force.y,
        "f_z": recording.force.z,
        "m_x": recording.torque.x,
        "m_y": recording.torque.y,
        "m_z": recording.torque.z
    }

    # filename based on the time
    filename = time.strftime("%Y%m%d-%H%M%S") + ".npy"

    # save data
    np.save(folder + filename, data)

    # print force and torque
    print()
    print("Force: ({}, {}, {}) N".format(recording.force.x, recording.force.y, recording.force.z))
    print("Torque: ({}, {}, {}) Nm".format(recording.torque.x, recording.torque.y, recording.torque.z))
    print()

    # display image but close it after 1 second
    plt.imshow(gs_img)
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="Record data with cnc machine.")
    parser.add_argument("--serial_port", type=str, default="/dev/tty.usbmodem38A6337730371", help="Serial port of the sensor")
    parser.add_argument("--indenter", type=str, default="None", help="Name of the indenter")
    parser.add_argument("--deg", type=float, default=0.0, help="Angle of the indenter")
    parser.add_argument("--folder", type=str, required=True, help="Folder to save data")
    parser.add_argument("--coordinates", type=str, required=True, help="Path to numpy file with coordinates")

    args = parser.parse_args()

    # read numpy file with coordinates
    coordinates = np.load(args.coordinates)
    print(coordinates)

    client = Client("192.168.0.123", 8080)
    client.connect()

    # number of points already collected
    j = 0

    # iterate over coordinates
    for i, (starting_position, displacement) in enumerate(coordinates):

        if i < j:
            continue

        # unpack starting position
        x_0, y_0, z_0 = starting_position

        # unpack displacement
        d_x, d_y, d_z = displacement

        # move to starting position but with z = +1
        retVal = client.move(x_0, y_0, z_0 + 1, 1000)

        # move to starting position
        retVal = client.move(x_0, y_0, z_0, 1000)

        # move displacement
        retVal = client.move(x_0 + d_x, y_0 + d_y, z_0 + d_z, 100)
        print("Current position: ({}, {}, {})".format(x_0 + d_x, y_0 + d_y, z_0 + d_z))

        # wait 0.2 seconds
        time.sleep(0.2)

        # record data point
        print("Sample no: ", i+1)
        record_single_datapoint(serial_port=args.serial_port, indenter=args.indenter, folder=args.folder, x_0=x_0, y_0=y_0, z_0=z_0, d_x=d_x, d_y=d_y, d_z=d_z, deg=args.deg)

        # undo displacement
        retVal = client.move(x_0, y_0, z_0, 100)

        # move to starting position but with z = +1
        retVal = client.move(x_0, y_0, z_0 + 1, 1000)

    # go home
    retVal = client.move(0.0, 0.0, 70.0, 1000)
    print(retVal)
