from .recording import *
import serial
import time
import struct

_BAUD_RATE = 2000000


class CalibrationMatrix:

    def __init__(self, file_path: str = None):
        """
        Creates a new calibration matrix. If file_path is specified, the constructor will load from that file.
        Otherwise, an identity matrix will be created.
        :param file_path: File to read from, default None
        """
        self.matrix = np.zeros((6, 6))
        if file_path is None:
            for column in range(0, 6):
                self.matrix[column][column] = 1
        else:
            with open(file_path, 'r') as file_input:
                for row in range(0, 6):
                    line = file_input.readline()
                    values = line.split(',')
                    for column in range(0, 6):
                        self.matrix[row][column] = float(values[column])

    def process(self, raw_values):
        """
        Processes a 6D vector of six raw values using the calibration matrix. This implements a standard
        matrix multiplication. This returns an array containing three force and three torque values.
        :param raw_values: The raw values
        :return: The calculated F/T values
        """
        dest = np.empty((6,))
        for row in range(0, 6):
            result = 0.0
            for column in range(0, 6):
                result += self.matrix[row][column] * raw_values[column]
            dest[row] = result
        return dest


class HEXSensor:

    def __init__(self, com_port):
        """
        Creates a new HEX sensor object to connect to a HEX F/T sensor
        :param com_port: The com port to use
        """
        self._serial_interface = None
        self._com_port = com_port
        self._calibration_matrix = CalibrationMatrix()

    def is_connected(self) -> bool:
        """
        Checks if the sensor was connected and the connection is still open.
        :return: Connection status (True/False)
        """
        if self._serial_interface is None:
            return False
        return self._serial_interface.isOpen()

    def connect(self) -> bool:
        """
        Attempts to connect to the sensor electronics interface. Status can be checked by calling is_connected()
        :return: True if connection successful, False otherwise
        """
        if self.is_connected():
            return True
        self._serial_interface = serial.Serial(port=self._com_port, baudrate=_BAUD_RATE, timeout=None)
        return self.is_connected()

    def disconnect(self):
        """
        Disconnects the sensor electronics interface. Calling disconnect does nothing if the sensor was
        never connected before or the serial port was closed to another reason.
        """
        if self.is_connected():
            self._serial_interface.close()

    def set_calibration_matrix(self, matrix: CalibrationMatrix):
        """
        Sets the calibration matrix to use for recording samples. The default value is an identity matrix which
        requires the sensor electronics to output F/T values instead of raw values. This can be configures by setting
        DIP switch 6 to ON. Please consider the manual for further information.
        :param matrix: The new configuration matrix
        """
        self._calibration_matrix = matrix

    def record_sample(self):
        """
        Records a single sample from the sensor electronics interface. Returns None if no sensor is connected
        or an instance of a DataSet containing the microsecond time stamp and the force and torque values.
        The DateSet is already processed using the calibration matrix specified by set_calibration_matrix or
        the default identity matrix. Please note that calling this function will block until at least one
        data set was sent by the electronics interface.
        :return: DataSet if read was successful, None otherwise
        """
        if not self.is_connected():
            return None
        serial_line = self._serial_interface.read(28)
        raw_values = [struct.unpack('f', serial_line[i * 4:(i + 1) * 4])[0] for i in range(0, 6)]
        processed_values = self._calibration_matrix.process(raw_values)
        return DataSet(time.time_ns() // 1000,
                       ForceValue(processed_values[0], processed_values[1], processed_values[2]),
                       TorqueValue(processed_values[3], processed_values[4], processed_values[5]))

    def record_duration(self, duration: float, sample_rate: int):
        """
        The duration and sample rate are multiplied to calculate the number of samples in the time frame. Then,
        that amount of samples are read by record_samples(). This method blocks until all samples are read or
        an I/O error occurs. The specified sample rate has to be equal to the sample rate configured on the
        electronics interface using the DIP switches. Please consider the manual for further information.
        :param duration: The duration in seconds
        :param sample_rate: The sample rate set on the electronics interface
        :return: BufferedRecording if read was successful, None otherwise
        """
        num_samples = int(sample_rate * duration)
        return self.record_samples(num_samples)

    def record_samples(self, num_samples: int):
        """
        Reads the specified number of samples from the serial port. This method blocks until all samples are read or
        an I/O error occurs.
        :param num_samples: The number of samples to read
        :return: BufferedRecording if read was successful, None otherwise
        """
        dest_array = np.empty((num_samples,), dtype=DataSet)
        read = 0
        for i in range(0, num_samples):
            sample = self.record_sample()
            if sample is None:
                break
            dest_array[i] = sample
            read += 1
        if read == 0:
            return None
        return BufferedRecording(data_points=dest_array[0:read])
