#!/usr/bin/env python3

import os
import argparse
import numpy as np

from resense_hex_21.resense import sensor
#from resense import sensor


def connect_to_sensor(serial_port):
    """
    Connect to the Resense Hex 21 sensor.

    :param serial_port: serial port to which the sensor is connected
    :return: sensor object
    """

    print("\nMaking sure that {} is readable...".format(serial_port))
    os.system("sudo chmod a+rw {}".format(serial_port))
    print("Connecting to sensor electronics at {}...".format(serial_port))

    resense_hex21 = sensor.HEXSensor(serial_port)
    resense_hex21.connect()

    return resense_hex21


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="Interface for the Resense Hex 21 sensor.")
    # Mac: ls -l  /dev/tty*    ->     /dev/tty.usbmodem38A6337730371
    parser.add_argument("--serial_port", nargs="+", default="/dev/tty.usbmodem38A6337730371", help="serial port to which the sensor is connected")
    args = parser.parse_args()

    resense_hex21 = connect_to_sensor(args.serial_port)

    while True:
        recording = resense_hex21.record_sample()

        print("Force: ({}, {}, {}) N".format(recording.force.x, recording.force.y, recording.force.z))
        print("Torque: ({}, {}, {}) Nm".format(recording.torque.x, recording.torque.y, recording.torque.z))
        print("\n")
