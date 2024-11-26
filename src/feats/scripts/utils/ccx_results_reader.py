#!/usr/bin/env python3

import os
import numpy as np

from utils.ccx2paraview import *


def read_frd(frd_filename):
    """
    Read .frd file and return ccx2paraview object.

    :param frd_filename: path to .frd file
    :return: ccx2paraview object
    """
    
    # check that file exists
    assert os.path.isfile(frd_filename), "File does not exist: {}".format(frd_filename)

    # create converter object and run it (this will create a .vtx file - feature of ccx2paraview)
    ccx2paraview = Converter(frd_filename, ["vtk"])
    ccx2paraview.run()

    return ccx2paraview


def get_coords(node_block):
    """
    Return coordinates of nodes in a dictionary with node id as key.

    :param node_block: node block of ccx2paraview object
    :return: dictionary with node id as key and coordinates as value
    """

    # get node ids
    node_ids = node_block.nodes.keys()

    # store coordinates in a dictionary with node id as key
    coords = {}
    for id in node_ids:
        coords[id] = np.array(node_block.nodes[id].coords)

    return coords


def get_values(result_blocks, result_type="S"):
    """
    Return values of selected result type in a dictionary with node id as key.
    Supported result types: "S" - stresses, "E" - strains, "U" - displacements.

    :param result_blocks: result blocks of ccx2paraview object
    :param result_type: type of result to return (default: "S" - stresses)
    :return: dictionary with node id as key and values as value
    """
    
    # set index of selected result block
    idx = 0
    for block in result_blocks:
        if block.name == result_type:
            break
        idx += 1

    # get selected result block
    result_block = result_blocks[idx]

    # get node ids
    node_ids = result_block.node_block.nodes.keys()

    # store values in a dictionary with node id as key
    values = {}
    for id in node_ids:
        values[id] = np.array(result_block.results[id])

    return values


def get_elements(elem_block):
    """
    Return elements in a dictionary with element number as key and nodes as value.

    :param elem_block: element block of ccx2paraview object
    :return: dictionary with element number as key and nodes as value
    """

    elements = {}

    # store nodes of element in a dictionary with element number as key
    for elem in elem_block.elements:
        elements[elem.num] = elem.nodes

    return elements


def read_nam(nam_file):
    """
    Read the .nam file and return the nodes.

    :param nam_file: path to the .nam file
    :return: list of nodes in the .nam file
    """

    # read the file
    with open(nam_file, "r") as f:
        lines = f.readlines()

    # just save the nodes starting from line 3
    nodes = []
    for line in lines[2:]:
        if line.startswith("*"):
            break
        nodes.append(int(line.split(",")[0]))

    return nodes


def read_dat(dat_filename):
    """
    Read .dat file and return the contact forces as a dictionary with element id as key.

    :param dat_filename: path to the .dat file
    :return: dictionary with element id as key and contact force as value
    """

    # store the contact forces in a dictionary
    contact_forces = {}

    # only save values from last time step
    save = False

    with open(dat_filename, "r") as f:
        for line in f:
            # skip lines until time step is reached
            if "contact force contribution (slave element+face,fx,fy,fz) for all contact elements and time 0.2000000E+01" in line:
                save = not save
                next(f)
                continue

            # make sure that values are saved only from last time step
            if save:
                # make sure that the line is not empty
                if line.strip():
                    elem_num = int(line.split()[0])
                    cf = np.array([np.float64(val) for val in line.split()[2:]], dtype=np.float64)
                    
                    # check if elem_num already exists in contact_forces
                    if elem_num in contact_forces:
                        contact_forces[elem_num] = np.add(contact_forces[elem_num], cf)
                    else:
                        contact_forces[elem_num] = cf
                
                elif not line.strip():
                    break

    return contact_forces