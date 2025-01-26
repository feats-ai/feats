#!/usr/bin/env python3

import argparse
import random
import numpy as np


def generate_initial_position(x_bound=[-3.0, +3.0], y_bound=[-5.0, +5.0], z_start=0.0):
    """
    Generate initial position for CNC machine.

    :param x_bound: x-axis boundary
    :param y_bound: y-axis boundary
    :param z_start: starting z-axis coordinate
    :return: initial position coordinates
    """

    x_0 = round(random.uniform(x_bound[0], x_bound[1]), 2)
    y_0 = round(random.uniform(y_bound[0], y_bound[1]), 2)
    z_0 = round(z_start, 2)

    return np.array([x_0, y_0, z_0])


def generate_displacement(dp_max=0.4, z_bound=[-2.0, 0.0]):
    """
    Generate random displacements in x, y and z directions.

    :param dp_max: maximum displacement in percentage
    :param z_bound: z-axis boundary
    :return: displacements in x and y directions
    """

    d_z = round(random.uniform(z_bound[0], z_bound[1]), 2)
    d_x = round(random.uniform(-dp_max*d_z, dp_max*d_z), 2)
    d_y = round(random.uniform(-dp_max*d_z, dp_max*d_z), 2)

    return np.array([d_x, d_y, d_z])


def generate_coordinates(num_points, x_bound, y_bound, z_bound, z_start, dp_max):
    """
    Generate CNC machine coordinates.

    :param num_points: number of points to generate
    :param x_bound: x-axis boundary
    :param y_bound: y-axis boundary
    :param z_bound: z-axis boundary
    :param z_start: starting z-axis coordinate
    :param dp_max: maximum displacement in percentage
    :return: CNC machine coordinates
    """

    coordinates = []

    for _ in range(num_points):
        starting_position = generate_initial_position(x_bound=x_bound, y_bound=y_bound, z_start=z_start)
        displacement = generate_displacement(dp_max=dp_max, z_bound=z_bound)

        coordinates.append([starting_position, displacement])

    return np.array(coordinates)


def save_coordinates(coordinates, folder, filename):
    """
    Save CNC machine coordinates to a file.

    :param coordinates: coordinates to save
    :param folder: folder to save the coordinates
    :param filename: filename to save the coordinates
    :return None
    """

    np.save(f"{folder}/{filename}.npy", coordinates)

    return None


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="Generate CNC machine coordinates.")
    parser.add_argument("--indenter", type=str, required=True, help="Indenter name")
    parser.add_argument("--folder", type=str, required=True, help="Folder to save the coordinates")
    parser.add_argument("--num_points", type=int, default=100, help="Number of points to generate")
    parser.add_argument("--x_bound", type=float, nargs=2, default=[-3.0, +3.0], help="X-axis boundary as two floats")
    parser.add_argument("--y_bound", type=float, nargs=2, default=[-5.0, +5.0], help="Y-axis boundary as two floats")
    parser.add_argument("--z_bound", type=float, nargs=2, default=[-2.0, 0.0], help="Z-axis boundary as two floats")
    parser.add_argument("--z_start", type=float, default=0.0, help="Starting Z coordinate")
    parser.add_argument("--dp_max", type=float, default=0.4, help="Maximum displacement percentage")

    args = parser.parse_args()

    coordinates = generate_coordinates(args.num_points, args.x_bound, args.y_bound, args.z_bound, args.z_start, args.dp_max)
    print(coordinates)
    save_coordinates(coordinates, args.folder, args.indenter)
