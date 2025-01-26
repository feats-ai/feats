from .recording import *
from .io_util import _validate_name_and_extension
import numpy as np
import pickle
import csv
import json
import struct


def _import_recording_from_csv(file_path: str) -> BufferedRecording:
    with open(file_path) as file_input:
        first_line = file_input.readline()
        csv_format = first_line.count(",") > first_line.count(";")
        cell_separator = ',' if csv_format else ';'
        dec_separator = '.' if csv_format else ','
        data_reader = csv.reader(file_input, delimiter=cell_separator)
        data_set_array = np.array([DataSet(int(data_row[0]),
                                           ForceValue(data_row[1], data_row[2], data_row[3], True, dec_separator),
                                           TorqueValue(data_row[4], data_row[5], data_row[6], True, dec_separator)
                                           ) for data_row in data_reader])
    return BufferedRecording(data_set_array)


def _import_recording_from_json(file_path: str) -> BufferedRecording:
    with open(file_path) as file_input:
        json_content = json.load(file_input)
        json_data = json_content['data']
    data_set_array = np.array([DataSet(data_row[0],
                                       ForceValue(data_row[1], data_row[2], data_row[3]),
                                       TorqueValue(data_row[4], data_row[5], data_row[6])
                                       ) for data_row in json_data])
    return BufferedRecording(data_set_array)


def _import_recording_from_pkl(file_path: str) -> BufferedRecording:
    with open(file_path, 'rb') as file_input:
        pickle_data = pickle.load(file_input)['data']
    data_set_array = np.array([DataSet(data_row[0],
                                       ForceValue(data_row[1], data_row[2], data_row[3]),
                                       TorqueValue(data_row[4], data_row[5], data_row[6])
                                       ) for data_row in pickle_data])
    return BufferedRecording(data_set_array)


def _import_recording_from_bin(file_path: str) -> BufferedRecording:
    with open(file_path, 'rb') as file_input:
        dataset_count = int.from_bytes(file_input.read(4), byteorder='big', signed=True)
        has_temperature = dataset_count < 0
        if dataset_count < 0:
            dataset_count = -dataset_count
        dataset_float_count = 7 if has_temperature else 6
        data_store_temp = []
        floats = [0] * 8
        for i in range(0, dataset_count):
            time_stamp = int.from_bytes(file_input.read(8), byteorder='big', signed=True)
            for j in range(0, dataset_float_count):
                floats[j] = struct.unpack('<f', file_input.read(4))[0]
            data_store_temp.append(DataSet(time_stamp,
                                           ForceValue(floats[0], floats[1], floats[2]),
                                           TorqueValue(floats[3], floats[4], floats[5])
                                           ))
    data_set_array = np.array(data_store_temp)
    return BufferedRecording(data_set_array)


def import_recording_from_file(file_path: str, file_extension: str = None) -> BufferedRecording:
    """
    Imports the recording from the specified file path. If file_extension is None, the file type will be detected
    from the file path. If file_extension is specified, it will determine the file format which will be attempted to
    read. If the file type is not supported, an Exception will be raised. Supported file types are bin, dat (binary),
    json, csv, pkl (pickle). This function is compatible with the recording files exported from FTE.
    :param file_path: The file to read from
    :param file_extension: The file type. Default is None
    :return: A BufferedRecording
    """
    file_extension = _validate_name_and_extension(file_path, file_extension)

    if file_extension == 'csv':
        return _import_recording_from_csv(file_path)
    if file_extension == 'json':
        return _import_recording_from_json(file_path)
    if file_extension == 'pkl':
        return _import_recording_from_pkl(file_path)
    if file_extension == 'bin' or file_extension == 'dat':
        return _import_recording_from_bin(file_path)

    raise Exception("extension not supported: ." + file_extension)
