#!/usr/bin/env python3

import os
import argparse

from itertools import combinations
import multiprocessing

import math
import numpy as np
from functools import partial
import cv2

from shapely.geometry import Polygon, box

import matplotlib.pyplot as plt

import utils.ccx_results_reader as crr


def load_homography(filename):
    """
    Load homography from .npy file.

    :param filename: path to .npy file
    :return: homography as numpy array
    """

    H = np.load(filename, allow_pickle=True)

    return H


def get_surface_elements(elements, sur_nodes):
    """
    Filter for elements that have all nodes on the surface.

    :param elements: dictionary with element number as key and list of nodes as value
    :param sur_nodes: list of nodes that are on the surface
    :return: dictionary with element number as key and list of nodes as value
    """

    # remove nodes from elements that are not in sur_nodes
    sur_elements = {}

    # iterate over all elements
    for elem_num, elem_nodes in elements.items():

        # check if all nodes of the element are in sur_nodes
        elem_sur_nodes = []
        for node in elem_nodes:
            # if the node of the element is in sur_nodes, add it to the filtered list of element nodes
            if node in sur_nodes:
                elem_sur_nodes.append(node)

        # if the element has 6 nodes, add it to the list of surface elements
        if len(elem_sur_nodes) == 6:
            sur_elements[elem_num] = elem_sur_nodes

    return sur_elements


def getTriangleCorners(nodes):
    """
    Get triangle corners of nodes.

    :param nodes: list of nodes
    :return: list of triangle corners
    """

    def distance(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    # find max distances (cornerpoints)
    max_dist = 0
    triangle_points = None

    for combo in combinations(nodes, 3):
        dist = (distance(combo[0], combo[1]) +
                distance(combo[1], combo[2]) +
                distance(combo[2], combo[0]))
        if dist > max_dist:
            max_dist = dist
            triangle_points = combo

    return triangle_points


def calculateStructCellValue(structCell, unstructPolygons):
    """
    Calculate the value of a structured cell based on the unstructured polygons.

    :param structCell: structured cell
    :param unstructPolygons: list of unstructured polygons
    :return: value of the structured cell
    """

    structValue = 0

    for e in unstructPolygons:
        overlapArea = structCell.intersection(e[0]).area
        unstructArea = e[0].area
        weight = overlapArea / unstructArea if unstructArea > 0 else 0
        structValue += e[1] * weight

    return structValue


def calculateStructValues(structGrid, unstructPolygons):
    """
    Calculate the values of a structured grid based on the unstructured polygons.

    :param structGrid: structured grid
    :param unstructPolygons: list of unstructured polygons
    :return: structured values
    """

    rows, cols = len(structGrid), len(structGrid[0])
    structValues = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            structCell = structGrid[i][j]
            structValues[i][j] = calculateStructCellValue(structCell, unstructPolygons)

    return structValues


def calculateStructValues_parallel(structGrid, unstructPolygons):
    """
    Calculate the values of a structured grid based on the unstructured polygons in parallel.

    :param structGrid: structured grid
    :param unstructPolygons: list of unstructured polygons
    :return: structured values
    """

    rows, cols = len(structGrid), len(structGrid[0])
    flatArr = np.array(structGrid).flatten()
    pool_obj = multiprocessing.Pool()
    res = pool_obj.map(partial(calculateStructCellValue, unstructPolygons=unstructPolygons), flatArr)
    pool_obj.close()

    return np.array(res).reshape((rows, cols))


def createStructuredGrid(rows, cols, min, max):
    """
    Create a structured grid.

    :param rows: number of rows
    :param cols: number of columns
    :param min: minimum coordinates of the grid
    :param max: maximum coordinates of the grid
    :return: structured grid
    """

    structured_grid = []
    dx, dy = max[0] - min[0], max[1] - min[1]
    dx, dy = dx/cols, dy/rows

    for i in range(rows):
        row = []
        for j in range(cols):
            cell = box(min[0] + j*dx, min[1] + i*dy, min[0]+(j+1)*dx, min[1]+(i+1)*dy)
            row.append(cell)
        structured_grid.append(row)

    return structured_grid


def generate_label(coords, surfaceElements, contact_forces, H, parallel=False):
    """
    Generate structured grid label from surface elements and contact forces.

    :param coords: coordinates of the nodes
    :param surfaceElements: surface elements
    :param contact_forces: contact forces
    :param H: homography matrix
    :param parallel: flag to use parallel processing
    :return: structured grid label and total force
    """

    # store unstructured polygons in lists
    unstructPolygons_x = []
    unstructPolygons_y = []
    unstructPolygons_z = []

    # iterate over all elements
    for i in contact_forces.keys():
        pol = []
        nodes_x = []
        nodes_y = []
        # iterate for all nodes of the element
        for j in surfaceElements[i]:
            nodes_x.append(coords[j][0])
            nodes_y.append(coords[j][1])
        # project points to image using homography
        projectedNodes = cv2.perspectiveTransform(np.array([np.vstack((nodes_x, nodes_y)).T]), H)[0]
        nodes_x = projectedNodes[:,0]
        nodes_y = projectedNodes[:,1]

        # add x and y coordinates to pol in tuples
        for j in range(len(nodes_x)):
            pol.append((nodes_x[j], nodes_y[j]))

        # add polygon and force value to unstructPolygons
        unstructPolygons_x.append((Polygon(getTriangleCorners(pol)), contact_forces[i][1]))
        unstructPolygons_y.append((Polygon(getTriangleCorners(pol)), contact_forces[i][0]))
        unstructPolygons_z.append((Polygon(getTriangleCorners(pol)), contact_forces[i][2]))


    # define size of image and grid
    xmax = 320
    xmin = 0
    ymax = 240
    ymin = 0

    factor = (xmax - xmin)/(ymax - ymin)
    nx = 24
    ny = math.ceil(nx * factor)

    # create structured grid and calculate structured values
    structGrid = createStructuredGrid(nx,ny,(xmin, ymin), (xmax, ymax))
    if parallel == False:
        structVals_x = calculateStructValues(structGrid, unstructPolygons_x)
        structVals_y = calculateStructValues(structGrid, unstructPolygons_y)
        structVals_z = calculateStructValues(structGrid, unstructPolygons_z)
    else:
        structVals_x = calculateStructValues_parallel(structGrid, unstructPolygons_x)
        structVals_y = calculateStructValues_parallel(structGrid, unstructPolygons_y)
        structVals_z = calculateStructValues_parallel(structGrid, unstructPolygons_z)

    # calculate total force and create grid
    total_force = np.array([np.sum(structVals_x), np.sum(structVals_y), np.sum(structVals_z)])
    grid = np.stack((structVals_x, structVals_y, structVals_z), axis=2)

    return grid, total_force


def plot_label(grid, total_force, H, cnc_data):
    """
    Plot all 3 channels of grid.

    :param grid: grid to be plotted
    :param total_force: total force of grid for each channel
    :param H: homography as numpy array
    :param cnc_data: dictionary with cnc data
    :return: None
    """

    gs_img = cnc_data["gs_img"]

    # project indenter movement to image
    x_mv = np.array([[cnc_data["x_0"], cnc_data["x_0"] + cnc_data["d_x"]]], dtype=np.float32)
    y_mv = np.array([[cnc_data["y_0"], cnc_data["y_0"] + cnc_data["d_y"]]], dtype=np.float32)
    center_mv = cv2.perspectiveTransform(np.array([np.vstack((x_mv, y_mv)).T]), H)[0]

    # create figure for all 3 channels
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # plot each channel (extend to image size?)
    img1 = axs[0].imshow(grid[:, :, 0])
    img2 = axs[1].imshow(grid[:, :, 1])
    img3 = axs[2].imshow(grid[:, :, 2])
    img4 = axs[3].imshow(gs_img.astype(np.uint8))

    # plot indenter movement p_0 + d_p with an arrow
    axs[3].arrow(center_mv[:, 0][0], center_mv[:, 1][0], center_mv[:, 0][1] - center_mv[:, 0][0], center_mv[:, 1][1] - center_mv[:, 1][0], color="red", head_width=5)

    # search for max and min values in channel 0 and 1 and set limits for colorbar symmetrically
    max_val_S = np.max([np.max(grid[:, :, 0]), np.max(grid[:, :, 1])])
    min_val_S = np.min([np.min(grid[:, :, 0]), np.min(grid[:, :, 1])])

    if np.abs(max_val_S) > np.abs(min_val_S):
        min_val_S = -max_val_S
    else:
        max_val_S = -min_val_S

    min_val_N = np.min([np.min(grid[:, :, 2])])

    plt.colorbar(img1)
    img1.set_clim(min_val_S, max_val_S)
    plt.colorbar(img2)
    img2.set_clim(min_val_S, max_val_S)
    plt.colorbar(img3)
    img3.set_clim(min_val_N, 0)

    # use total force as title for each channel
    axs[0].set_title("x: {:.2f}N".format(total_force[0]))
    axs[1].set_title("y: {:.2f}N".format(total_force[1]))
    axs[2].set_title("z: {:.2f}N".format(total_force[2]))

    # show plot and close it after 1 second
    plt.show(block=False)
    plt.pause(1)
    plt.waitforbuttonpress()
    plt.close()


def save_label(label_filename, gs_img, grid, total_force):
    """
    Save grid and total force to .npy file.

    :param label_filename: path to .npy file
    :param gs_img: gelsight image
    :param grid: grid to be saved
    :param total_force: total force of grid for each channel
    :return: None
    """

    label = {
        "gs_img": gs_img,
        "grid_x": grid[:, :, 0],
        "grid_y": grid[:, :, 1],
        "grid_z": grid[:, :, 2],
        "f_x": total_force[0],
        "f_y": total_force[1],
        "f_z": total_force[2]
    }

    np.save(label_filename, label)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="Generate grid label from fea results.")
    parser.add_argument("--ccx_results", type=str, default="../data/ccx", help="folder with ccx results")
    args = parser.parse_args()

    # load homography
    homography_file = "../homography/homography.npy"
    H = load_homography(homography_file)

    ccx_results = args.ccx_results + "/"

    errors = 0

    for results_folder in os.listdir(ccx_results):

        # make sure file does not start with a dot
        if results_folder.startswith("."):
            continue

        # if folder does not have a .dat file, skip
        if not os.path.exists(ccx_results + results_folder + "/sim.dat") or os.stat(ccx_results + results_folder + "/sim.dat").st_size == 0:
            continue

        try:
            # define paths to files
            frd_file = ccx_results + results_folder + "/sim.frd"
            dat_file = ccx_results + results_folder + "/sim.dat"
            nam_file = ccx_results + results_folder + "/gelsight_miniSurfaceNodes.nam"
            npy_file = ccx_results + results_folder + "/" + [f for f in os.listdir(ccx_results + results_folder) if f.endswith(".npy")][0]

            # read frd file
            ccxReader = crr.read_frd(frd_file)
            coords = crr.get_coords(ccxReader.frd.node_block)
            elements = crr.get_elements(ccxReader.frd.elem_block)

            # read dat file
            contact_forces = crr.read_dat(dat_file)

            # read surface nodes
            sur_nodes = crr.read_nam(nam_file)

            # get surface elements
            sur_elements = get_surface_elements(elements, sur_nodes)

            # read gs_img from .npy file
            cnc_data = np.load(npy_file, allow_pickle=True).item()

            # generate label if contact_forces are not empty
            if contact_forces:
                label_filename = "../data/labels/train/" + results_folder + "_" + ccx_results.split("/")[-2] + ".npy"
                grid, total_force = generate_label(coords, sur_elements, contact_forces, H, parallel=False)

                #plot_label(grid, total_force, H, cnc_data)

                # save label
                save_label(label_filename, cnc_data["gs_img"], grid, total_force)

                print("Saved label to: {}".format(label_filename))

            else:
                print("No contact forces found in: {}".format(ccx_results + results_folder))
                # write filename in error file
                with open("../data/labels/errors.txt", "a") as f:
                    f.write("No contact forces found in: {}\n".format(ccx_results + results_folder))

        except:
            print("Error processing folder: {}".format(ccx_results + results_folder))
            with open("../data/labels/errors.txt", "a") as f:
                f.write("Error processing folder: {}\n".format(ccx_results + results_folder))
            errors += 1

    print("\n*** Finished processing with {} errors ***\n".format(errors))
