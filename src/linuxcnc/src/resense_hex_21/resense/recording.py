import numpy as np
from enum import Enum
import pickle
import os
from pathlib import Path
from math import sqrt


class ForceValue:

    def __init__(self, x, y, z, parse: bool = False, decimal_separator: str = '.'):
        if parse:
            self.x = float(x.replace(decimal_separator, '.'))
            self.y = float(y.replace(decimal_separator, '.'))
            self.z = float(z.replace(decimal_separator, '.'))
        else:
            self.x = x
            self.y = y
            self.z = z

    def length_squared(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def length(self) -> float:
        return sqrt(self.length_squared())


class TorqueValue:

    def __init__(self, x, y, z, parse: bool = False, decimal_separator: str = '.'):
        if parse:
            self.x = float(x.replace(decimal_separator, '.'))
            self.y = float(y.replace(decimal_separator, '.'))
            self.z = float(z.replace(decimal_separator, '.'))
        else:
            self.x = x
            self.y = y
            self.z = z

    def length_squared(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def length(self) -> float:
        return sqrt(self.length_squared())


class Variable(Enum):
    FORCE = 1
    TORQUE = 2


class Direction(Enum):
    X = 1
    Y = 2
    Z = 3


class DataSet:

    def __init__(self, time_offset: int = 0, force: ForceValue = None, torque: TorqueValue = None):
        self.time_offset = time_offset
        self.force = force
        self.torque = torque

    def get_time_stamp(self, seconds: bool = True) -> float:
        """
        Returns the time stamp of this DataSet in seconds or microseconds depending on whether
        the parameter seconds is True or False.
        :param seconds: Whether to return time stamp as seconds or microseconds
        :return: The time stamp of this data set
        """
        return self.time_offset / 1000000.0 if seconds else self.time_offset

    def get_time_offset(self, first_time_offset: int, seconds: bool = True) -> float:
        """
        Returns the offset of this DataSet from the first DataSet in a recording. The time stamp
        of the first DataSet is specified by first_time_offset. The value is returned in seconds or
        microseconds depending on whether the parameter seconds is True or False.
        :param first_time_offset: The time stamp of the first/reference DataSet
        :param seconds: Whether to return time offset as seconds or microseconds
        :return: The time offset of this data set relative to the specified time stamp
        """
        delta = self.time_offset - first_time_offset
        return delta / 1000000.0 if seconds else delta


class BufferedRecording:

    def __init__(self, data_points: np.ndarray = None, file: str = None):
        """
        Creates a new BufferedRecording. If data_points is not None, the recording will contain the specified data
        points. If data_points is None and file is not None, the recording will be imported from a binary file format.
        Note that this file format IS NOT compatible with the importer, exporter or FTE! If the number of data points
        is 0 an Exception will be raised. If both data_points and file are None an Exception will be raised.
        :param data_points: The data points of this recording. Default None
        :param file: The file to read from. Default None
        """
        if data_points is None and file is None:
            raise Exception('specify either data_points or input_file')
        if data_points is not None:
            self.data_points = data_points
            self.name = ""
        else:
            with open(file, 'rb') as fin:
                obj = pickle.load(fin)
            self.data_points = obj.data_points
            self.name = os.path.basename(file)

        if len(self.data_points) == 0:
            raise Exception("no data points given")

        self.first_time_offset = self.data_points[0].time_offset
        self.length = len(self.data_points)

    def set_name(self, name: str):
        """
        Changes the name of this recording.
        :param name: The new name
        """
        self.name = name

    def get_name(self) -> str:
        """
        :return: The name of this recording
        """
        return self.name

    def get_time_duration(self, seconds: bool = True) -> float:
        """
        Returns the duration of this recording. If seconds is True, the return value will be in seconds,
        microseconds otherwise.
        :param seconds: Whether to return seconds or microseconds.
        :return: The duration of this recording
        """
        last_index = len(self.data_points) - 1
        duration_micro_seconds = self.data_points[last_index].time_offset - self.first_time_offset
        return duration_micro_seconds / 1000000.0 if seconds else duration_micro_seconds

    def get_average_frequency(self) -> float:
        """
        Returns the average frequency of this recording. This is calculated by taking the total number of samples
        and dividing it by the duration of this recording.
        :return: The average frequency
        """
        average_distance = self.get_time_duration() / self.get_data_point_count()
        return 1.0 / average_distance

    def get_data_point_count(self) -> int:
        """
        :return: The number of data points in this recording
        """
        return self.length

    def get_data_point(self, index: int):
        """
        Returns the nth data point in this recording.
        :param index: The index of the data point
        :return: The data point at the specified index
        """
        if index < 0 or index >= self.length:
            return None
        return self.data_points[index]

    def _get_data_point_offset_seconds(self, data_point: DataSet, seconds: bool = True):
        return data_point.get_time_offset(self.first_time_offset, seconds)

    def get_array_of_timestamps(self, relative: bool = True, seconds: bool = True):
        """
        Returns a numpy array containing the time stamps of all data sets in this recording. If relative is False,
        all time stamps will be absolute values as they were imported or received from the system clock. If relative
        is True, all time values will be a time offset relative to the first data point in this recording. If
        seconds is True, the resulting time values will be in seconds, microseconds otherwise.
        :param relative: Whether to return relative values
        :param seconds: Whether to return seconds or microseconds
        :return: A numpy array of time values
        """
        return np.array([
            (self._get_data_point_offset_seconds(point, seconds) if relative else point.get_time_stamp(seconds))
            for point in self.data_points])

    def get_array_of_values(self, variable: Variable, direction: Direction) -> np.ndarray:
        """
        Returns a 1D numpy array containing floating point values from all data points. In this process only one
        of the measurements (force/torque) and one dimension (x/y/z) is selected from each data point.
        :param variable: Force/Torque
        :param direction: X/Y/Z
        :return: 1D numpy array
        """
        if variable is Variable.FORCE:
            if direction is Direction.X:
                return np.array([point.force.x for point in self.data_points])
            elif direction is Direction.Y:
                return np.array([point.force.y for point in self.data_points])
            elif direction is Direction.Z:
                return np.array([point.force.z for point in self.data_points])
        elif variable is Variable.TORQUE:
            if direction is Direction.X:
                return np.array([point.torque.x for point in self.data_points])
            elif direction is Direction.Y:
                return np.array([point.torque.y for point in self.data_points])
            elif direction is Direction.Z:
                return np.array([point.torque.z for point in self.data_points])

    def get_array_of_vectors(self, variable: Variable) -> np.ndarray:
        """
        Returns a 2D numpy array. Each row represents the force/torque (specified by the variable parameter) value
        of one data set. Each column contains the corresponding X/Y/Z value.
        :param variable: Force/Torque
        :return: 2D numpy array
        """
        if variable is Variable.FORCE:
            return np.array([[point.torque.data_point, point.torque.y, point.torque.z] for point in self.data_points])
        elif variable is Variable.FORCE:
            return np.array([[point.force.data_point, point.force.y, point.force.z] for point in self.data_points])

    def get_data_points(self, start=0, end=None) -> np.ndarray:
        """
        Returns a 1D numpy array of all data points.
        :param start: Start offset
        :param end: End offset
        :return: 1D numpy array
        """
        if end is None:
            end = self.get_data_point_count()
        return np.array(self.data_points[start:end])

    def get_data_point_indices_for_time_frame(self, start_time: float, end_time: float) -> tuple:
        """
        Returns a tuple of two indices. The first index represents the first data point in this recording whose
        timestamp is at least start_time after the timestamp of the first data point. The second index represents
        the last data point in this recording whose timestamp is at most end_time after the timestamp of the first
        data point. Both time intervals are specified in seconds. If any index or time offset is out of this
        recordings bounds, (-1,-1) is returned.
        :param start_time: Start time offset
        :param end_time: End time offset
        :return: The time frame converted to indices
        """
        if self.get_data_point_count() == 0:
            return -1, -1

        recording_start_time = self.data_points[0].time_offset
        recording_frequency = self.get_average_frequency()
        data_point_count = self.get_data_point_count()

        # convert values to microseconds for easy comparison
        start_time_us = 1000000 * start_time + recording_start_time
        end_time_us = 1000000 * end_time + recording_start_time

        # guess the start index and then iterate over data points until the correct data point is found
        start_index = int(start_time * recording_frequency)
        if start_index < 0 or start_index >= data_point_count:
            return -1, -1
        while self.data_points[start_index].time_offset > start_time_us:
            start_index -= 1
            if start_index < 0:
                return -1, -1
        while self.data_points[start_index].time_offset < start_time_us:
            start_index += 1
            if start_index >= data_point_count:
                return -1, -1

        # guess the end index and then iterate over data points until the correct data point is found
        end_index = int(end_time * recording_frequency)
        if end_index < 0 or end_index >= data_point_count:
            return -1, -1
        while self.data_points[end_index].time_offset < end_time_us:
            end_index += 1
            if end_index >= data_point_count:
                return -1, -1
        while self.data_points[end_index].time_offset > end_time_us:
            end_index -= 1
            if end_index < 0:
                return -1, -1

        return start_index, end_index

    def save(self, file: str):
        with open(file, 'wb') as file_output:
            pickle.dump(self, file_output)


class BufferedRecordingSet:

    def __init__(self, recordings: list = None):
        """
        Creates a set of recordings. If the parameter recordings is specified, those recordings will be added
        to the created recording set.
        :param recordings: Initial recordings
        """
        self.recordings = [] if recordings is None else recordings

    def add_recording(self, recording: BufferedRecording):
        """
        Adds a new recording to this set.
        :param recording: A recording
        """
        self.recordings.append(recording)

    def set_recording_names(self, name: str):
        """
        Sets the names of all recordings. Each name will be appended with a hyphen and a numerical index
        starting from 0.
        :param name: The name of the recordings
        """
        index = 0
        for recording in self.recordings:
            recording.set_name(name + "-" + str(index))
            index += 1

    def add_recordings(self, other):
        """
        Add all recordings from the other recording set to this one.
        :param other: Another BufferedRecordingSet
        """
        for recording in other.recordings:
            self.recordings.append(recording)

    def get_recordings(self) -> list:
        """
        :return: A list of all recordings
        """
        return self.recordings

    def get_recording_count(self) -> int:
        """
        :return: The number of recordings
        """
        return len(self.recordings)

    def get_recording(self, index) -> BufferedRecording:
        """
        Returns the nth recording in this set.
        :param index: The index
        :return: The recording at the index
        """
        return self.recordings[index]

    def save(self, folder: str):
        """
        Saves all recordings into the specified folder. If recordings have names, the name will be used as the
        file name. Otherwise, the files will be named 'unnamed-' appended with a numerical index.
        :param folder: The folder to save to
        """
        unnamed_index = 0
        Path(folder).mkdir(parents=True, exist_ok=True)
        for recording in self.recordings:
            recording_name = recording.get_name()
            if len(recording_name) == 0:
                recording_name = "unnamed-" + str(unnamed_index)
                unnamed_index += 1
            file_path = os.path.join(folder, recording_name + '.bin')
            recording.save(file_path)

    def load(self, folder: str):
        """
        Loads all recordings from the specified folder. The recordings will be unnamed and added to this
        recording set.
        :param folder: The folder to read from
        """
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            recording = BufferedRecording(file=file_path)
            self.add_recording(recording)


def concatenate_recordings(first: BufferedRecording, seconds: BufferedRecording) -> BufferedRecording:
    """
    Returns a new BufferedRecording containing all data sets from the specified two recordings.
    :param first: First recording
    :param seconds: Another recording
    :return: A concatenated recording
    """
    data_points = first.data_points + seconds.data_points
    return BufferedRecording(data_points=data_points)
