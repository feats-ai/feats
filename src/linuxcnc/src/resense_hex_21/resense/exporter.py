from .recording import *
from .io_util import _validate_name_and_extension
import pickle
import json
import struct


def _export_recording_to_csv(recording, file_path):
    with open(file_path, 'w') as file_output:
        file_output.writelines('Timestamp,Fx,Fy,Fz,Mx,My,Mz\n')
        for data_set in recording.get_data_points():
            f = data_set.force
            t = data_set.torque
            file_output.writelines("{},{},{},{},{},{},{}\n".format(
                data_set.time_offset,
                f.x, f.y, f.z,
                t.x, t.y, t.z
            ))


def _export_recording_to_json(recording, file_path):
    raise Exception('export to json is not supported')


def _export_recording_to_pkl(recording, file_path):
    raise Exception('export to pkl is not supported')


def _export_recording_to_bin(recording, file_path):
    raise Exception('export to bin is not supported')


def export_recording_to_file(recording: BufferedRecording, file_path: str, file_extension: str = None):
    """
    Exports the recording to the specified file path. If file_extension is None, the file type will be detected
    from the file path. If file_extension is specified, it will determine the type of file written. If the
    file type is not supported, an Exception will be raised. Recordings exported from this function can be
    imported by FTE. Note that only export to CSV is supported in this version of Resense.py!
    :param recording: The recording to write
    :param file_path: The file to write to
    :param file_extension: The file type. Default is None
    """
    file_extension = _validate_name_and_extension(file_path, file_extension)

    if file_extension == 'csv':
        _export_recording_to_csv(recording, file_path)
    elif file_extension == 'json':
        _export_recording_to_json(recording, file_path)
    elif file_extension == 'pkl':
        _export_recording_to_pkl(recording, file_path)
    elif file_extension == 'bin' or file_extension == 'dat':
        _export_recording_to_bin(recording, file_path)
    else:
        raise Exception("extension not supported: ." + file_extension)
