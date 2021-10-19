import numpy as np
from preprocessing import TrackClusters
import cv2
import datetime
# import vlc
import time
import serial
from settings import *
import matplotlib.pyplot as plt
import binascii
from collections import deque
import settings
import pyvisa as visa
import pymmcore
import tqdm
import pandas as pd


class VideoStream:

    def __init__(self):

        # Initialiye core object
        self.core = pymmcore.CMMCore()

        # Find camera config --> get camera
        self.core.setDeviceAdapterSearchPaths(["C:/Program Files/Micro-Manager-2.0gamma"])
        self.core.loadSystemConfiguration('C:/Program Files/Micro-Manager-2.0gamma/mmHamamatsu.cfg')
        self.label = self.core.getCameraDevice()

        # Set exposure time
        self.core.setExposure(EXPOSURE_TIME)

        # Prepare acquisition
        self.core.prepareSequenceAcquisition(self.label)
        self.core.startContinuousSequenceAcquisition(0.1)
        self.core.initializeCircularBuffer()
        time.sleep(1)

    def snap(self, f_name, size=(IMG_SIZE, IMG_SIZE)):

        # Get image
        img = self.core.getLastImage()

        # Resize image
        img = cv2.resize(img, size)

        # Save image
        cv2.imwrite(f_name, img)

        # Return image
        return img

    # def __init__(self, url=STREAM_URL):
    #
    #     # Define VLC instance
    #     instance = vlc.Instance()
    #
    #     # Define VLC player
    #     self.player = instance.media_player_new()
    #
    #     # Define VLC media
    #     self.media = instance.media_new(url)
    #
    #     # Set player media
    #     self.player.set_media(self.media)
    #     self.player.play()
    #     time.sleep(2)
    #
    # def snap(self, f_name):
    #
    #     # Snap an image of size (IMG_SIZE, IMG_SIZE)
    #     self.player.video_take_snapshot(0, f_name, IMG_SIZE, IMG_SIZE)


class Actuator:

    def __init__(self):

        # Initiate contact with arduino
        self.arduino = serial.Serial(port=SERIAL_PORT_ARDUINO, baudrate=BAUDRATE)
        print(f"Arduino: {self.arduino.readline().decode()}")
        time.sleep(1)  # give serial communication time to establish

        # Initiate status of 4 piëzo transducers
        self.arduino.write(b"9")  # Turn all outputs to LOW

    def move(self, action: int):

        self.arduino.write(b"9")  # Turn old piezo off
        self.arduino.write(f"{action}".encode())  # Turn new piezo on

    def close(self):

        self.arduino.write(b"9")


class Observer:

    def __init__(self, port=SERIAL_PORT_LEICA):

        self.observer = serial.Serial(port=port, baudrate=9600, timeout=2)
        print(f"Opened port {port}: {self.observer.isOpen()}")
        self.pos = (0, 0)

    def reset(self):
        self.observer.write(bytearray([255, 82]))  # Reset device
        time.sleep(2)  # Give the device time to reset before issuing new commands

    def close(self):
        self.observer.close()
        print(f"Closed serial connection")

    def get_status(self, motor: int):  # TODO --> use check_status to make sure the motor does not get command while still busy
        assert motor in [0, 1, 2]  # Check if device number is valid
        self.observer.write(bytearray([motor, 63, 58]))  # Ask for device status
        received = self.observer.read(1)  # Read response (1 byte)
        if received:
            print(f"Received: {received.decode()}")  # Print received message byte

    def write_target_pos(self, motor: int, target_pos: int):

        # Translate coordinate to a 3-byte message
        msg = self.coord_to_msg(int(target_pos))
        # print(f"Writing target pos bytes: {msg} to motor {motor}")

        # [device number, command, 3, message, stop signal]
        self.observer.write(bytearray([motor, 84, 3, msg[0], msg[1], msg[2], 58]))
        time.sleep(SLEEP_TIME)

    def get_motor_pos(self, motor):

        # Ask for position
        self.observer.write(bytearray([motor, 97, 58]))

        # Initialise received variable
        received = b''

        # For timeout functionality
        t0 = time.time()

        # Check for received message until timeout
        while not received:
            received = self.observer.read(3)  # Read response (3 bytes)
            if received == b'\x00\x00\x00':
                return 0
            if time.time() - t0 >= SLEEP_TIME:  # Check if it takes longer than a second
                print(f"No bytes received: {received}")
                return 0

        # Return translated message
        translated = self.msg_to_coord(received)
        # print(f"Motor pos: {translated}")  # Print received message as coordinate
        time.sleep(SLEEP_TIME)

        return translated

    def get_target_pos(self, motor: int):

        # Ask for position
        self.observer.write(bytearray([motor, 116, 58]))

        # Initialise received variable
        received = b''

        # For timeout functionality
        t0 = time.time()

        # Check for received message until timeout
        while not received:
            received = self.observer.read(3)  # Read response (3 bytes)
            if received == b'\x00\x00\x00':
                return 0
            if time.time() - t0 >= SLEEP_TIME:  # Check if it takes longer than a second
                print(f"No bytes received: {received}")
                return 0

        # Return translated message
        translated = self.msg_to_coord(received)
        # print(f"Target pos: {translated}")  # Print received message as coordinate
        time.sleep(SLEEP_TIME)

        return translated

    def move_to_target(self, motor: int, target_pos: int):  # TODO --> Coordinate boundaries so we don't overshoot the table and get the motor stuck, as we need to reset then

        """
        100k steps is 1 cm in real life
        """

        # Write target position
        self.write_target_pos(motor=motor, target_pos=target_pos)

        # Move motor to target coordinate
        self.observer.write(bytearray([motor, 71, 58]))

        # Give motor time to move
        time.sleep(SLEEP_TIME)

    def coord_to_msg(self, coord: int):

        # Convert from two's complement
        if np.sign(coord) == -1:
            coord += 16777216

        # Convert to hexadecimal and pad coordinate with zeros of necessary (len should be 6 if data length is 3)
        hexa = hex(coord).split("x")[-1].zfill(6)

        # Get the four digit binary code for each hex value
        four_digit_binaries = [bin(int(n, 16))[2:].zfill(4) for n in hexa]

        # Convert to 3-byte message
        eight_digit_binaries = [f"{four_digit_binaries[n] + four_digit_binaries[n + 1]}".encode() for n in range(0, 6, 2)][::-1]

        return [int(m, 2) for m in eight_digit_binaries]

    def msg_to_coord(self, msg: str):

        # Read LSB first
        msg = msg[::-1]

        # Convert incoming hex to readable hex
        hexa = binascii.hexlify(bytearray(msg))

        # Convert hex to decimal
        coord = int(hexa, 16)

        # Convert to negative coordinates if applicable
        if bin(coord).zfill(24)[2] == '1':
            coord -= 16777216
        return int(coord)

    def pixels_to_increment(self, pixels: np.array):
        return np.array([pixels[0]*PIXEL_MULTIPLIER_LEFT_RIGHT, pixels[1]*PIXEL_MULTIPLIER_UP_DOWN])

    def move_increment(self, offset_pixels: np.array):

        # Get increments and add to current position
        self.pos += self.pixels_to_increment(pixels=offset_pixels)

        # Move motors x and y
        print("Moving...")
        self.move_to_target(motor=1, target_pos=self.pos[0])  # Move left/right
        self.move_to_target(motor=2, target_pos=self.pos[1])  # Move up/down
        time.sleep(3)  # TODO --> check optimal sleep time


class FunctionGenerator:

    def __init__(self, instrument_descriptor=INSTR_DESCRIPTOR):
        rm = visa.ResourceManager()
        print(rm.list_resources())
        if not instrument_descriptor:
            instrument_descriptor = rm.list_resources()[0]
        self.AFG3000 = rm.open_resource(instrument_descriptor)
        self.AFG3000.write('*RST')  # reset AFG

    def reset(self):
        self.set_vpp(vpp=MIN_VPP)
        self.set_frequency(frequency=MIN_FREQUENCY)

    def set_vpp(self, vpp: float):
        self.AFG3000.write(f'source1:voltage:amplitude {vpp}')  # Set vpp

    def set_frequency(self, frequency: float):
        self.AFG3000.write(f'source1:Frequency {frequency*1000}')  # Set frequency


class SwarmEnvTrackBiggestCluster:

    def __init__(self,
                 source=VideoStream(),
                 actuator=Actuator(),
                 observer=Observer(),
                 function_generator=FunctionGenerator(),
                 target_points=TARGET_POINTS,
                 metadata=METADATA):

        # Initialize devices
        self.source = source  # Camera
        self.actuator = actuator  # Piezo's
        self.observer = observer  # Leica xy-platform
        self.function_generator = function_generator  # Function generator

        # Keep track of target point (idx in target_points)
        self.target_points = target_points
        self.target_idx = 0

        # Metadatastructure
        self.metadata = metadata

        # Initialize Vpp and frequency
        self.function_generator.reset()
        self.vpp = MIN_VPP
        self.frequency = MIN_FREQUENCY  # kHz

        # Initialize memory
        self.memory = deque(maxlen=5)

    def reset(self, bbox):

        # Set env steps to 0
        self.step = 0

        # Initialize tracking algorithm
        self.tracker = TrackClusters(bbox=bbox)

        # Initialize function generator
        self.function_generator.reset()

        # Get time
        self.now = round(time.time(), 3)

        # Define file name
        filename = SAVE_DIR + f"reset.png"

        # Snap a frame from the video stream and save
        img = self.source.snap(f_name=filename)
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # Still need to optimize this

        # Get the centroid of (biggest) swarm
        self.state, self.size = self.tracker.reset(img=img)

        # Exception handling
        if not self.state:
            self.state = (0, 0)

        # Add metadata to dataframe
        self.metadata.append(
          {"Filename": filename,
           "Time": self.now,
           "Delta_time": None,
           "Vpp": self.vpp,
           "Frequency": self.frequency,
           "Size": self.size,
           "Action": None,
           "Pos(t)": self.state,
           "Pos(t-dt)": None,
           "Target": self.target_points[self.target_idx],
           "Step": 0}
        )

        # Return centroids of n amount of swarms
        return self.state

    def map(self):
        return

    # TODO --> Incorporate PID function from model.py
    def set_vpp_and_frequency(self, dist_from_target):

        # Set Vpp based on distance from target
        self.vpp = (dist_from_target / IMG_SIZE) * (MAX_VPP - MIN_VPP) + MIN_VPP
        self.function_generator.set_vpp(vpp=self.vpp)

        # Set frequency if we don't move at a certain speed
        if self.memory > 1:
            if np.average(self.memory) < THRESHOLD_SPEED:
                self.frequency += 1
                if self.frequency > MAX_FREQUENCY:
                    self.frequency = MIN_FREQUENCY
                self.function_generator.set_frequency(frequency=self.frequency)
                time.sleep(0.1)  # TODO --> Optimize

    def env_step(self, action: int):

        # Calculate vpp and frequency
        dist_from_target = np.linalg.norm(np.array(self.state) - np.array(self.target_points[self.target_idx]))
        self.set_vpp_and_frequency(dist_from_target=dist_from_target)

        # Actuate piezos
        self.actuator.move(action)

        # Get time
        self.now = round(time.time(), 3)

        # Define file name
        filename = SAVE_DIR + f"{self.now}.png"

        # Snap a frame from the video stream
        img = self.source.snap(f_name=filename)
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # Still need to optimize this

        # Get the new state
        old_state = self.state  # Copy old state
        self.state, self.size = self.tracker.update(img=img,  # Read image
                                                    target=self.target_points[self.target_idx],  # For verbose purposes
                                                    verbose=True)
        # Exception handling
        if not self.state:
            self.state = (0, 0)

        # Add distance traveled to memory
        self.memory.append(np.linalg.norm(np.array(old_state) - np.array(self.state)))

        # Add metadata to dataframe
        self.metadata.append(
          {"Filename": filename,
           "Time": self.now,
           "Vpp": self.vpp,
           "Frequency": self.frequency,
           "Size": self.size,
           "Action": action,
           "State": self.state,
           "Target": self.target_points[self.target_idx],
           "Step": self.step}
        )

        # # Move microscope to next point if offset goes into bounds
        # if dist_from_target < OFFSET_BOUNDS:
        #
        #     # Stop piezos
        #     self.actuator.close()
        #
        #     old_target = self.target_points[self.target_idx]
        #     self.target_idx = (self.target_idx + 1) % (len(self.target_points))
        #     new_target = self.target_points[self.target_idx]
        #     offset_pixels = ( (new_target[0] - old_target[0]), (new_target[1] - old_target[1]) )
        #     self.observer.move_increment(offset_pixels=np.array(offset_pixels))
        #
        #     # Update tracker position according to out movement
        #     # self.tracker.bbox[0], self.tracker.bbox[1] = new_target[0], new_target[1]
        #
        #     # Get time
        #     self.now = round(time.time(), 3)
        #
        #     # Define file name
        #     filename = SAVE_DIR + f"{self.now}.png"
        #
        #     # Snap a frame from the video stream
        #     img = self.source.snap(f_name=filename)
        #     img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # Still need to optimize this
        #
        #     self.tracker.reset(img=img,
        #                        bbox=(int(new_target[0]-0.5*self.tracker.bbox[2]),
        #                              int(new_target[1]-0.5*self.tracker.bbox[3]),
        #                              self.tracker.bbox[2],
        #                              self.tracker.bbox[3]))

        # Add step
        self.step += 1

        # Return centroids of n amount of swarms
        return self.state

    def close(self):
        print(f'Final bbox: {self.tracker.bbox}')
        self.actuator.close()
        self.observer.close()


if __name__ == '__main__':

    fg = FunctionGenerator()
    fg.set_frequency(frequency=240e3)
    # fg.set_vpp(vpp=2.1)


