import numpy as np
from preprocessing import TrackClusters
import cv2
import datetime
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
from model import get_action
import tektronix_func_gen as tfg
import atexit
from model import calc_action


class VideoStreamHammamatsu:

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

        # Error handling (sometimes the video buffer is empty if we take super fast images)
        img = None
        while not np.any(img):
            try:
                # Get image
                img = self.core.getLastImage()
            except:
                pass

        # Resize image
        img = cv2.resize(img, size)
        # img = np.array(img / np.max(img))

        # Save image
        cv2.imwrite(f_name, img)

        # Return image
        return img


class VideoStreamKronos:

    def __init__(self, url=STREAM_URL):
        import vlc

        # Define VLC instance
        instance = vlc.Instance()

        # Define VLC player
        self.player = instance.media_player_new()

        # Define VLC media
        self.media = instance.media_new(url)

        # Set player media
        self.player.set_media(self.media)
        self.player.play()
        time.sleep(2)

    def snap(self, f_name):

        # Snap an image of size (IMG_SIZE, IMG_SIZE)
        self.player.video_take_snapshot(0, f_name, IMG_SIZE, IMG_SIZE)


class ActuatorPiezos:

    def __init__(self):

        # Initiate contact with arduino
        self.arduino = serial.Serial(port=SERIAL_PORT_ARDUINO, baudrate=BAUDRATE_ARDUINO)
        print(f"Arduino: {self.arduino.readline().decode()}")
        time.sleep(1)  # give serial communication time to establish

        # Initiate status of 4 piëzo transducers
        self.arduino.write(b"9")  # Turn all outputs to LOW

    def move(self, action: int):

        if action == -1:
            return

        self.arduino.write(b"9")  # Turn old piezo off
        self.arduino.write(f"{action}".encode())  # Turn new piezo on

    def close(self):

        self.arduino.write(b"9")


class TranslatorLeica:

    def __init__(self, port=SERIAL_PORT_LEICA):

        # Open Leica
        self.observer = serial.Serial(port=port,
                                      baudrate=BAUDRATE_LEICA,  # Baudrate has to be 9600
                                      timeout=2)  # 2 seconds timeout recommended by the manual
        print(f"Opened port {port}: {self.observer.isOpen()}")
        self.pos = (0, 0)  # Reset position to (0, 0)

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
        time.sleep(SLEEP_TIME)

    def write_target_pos(self, motor: int, target_pos: int):

        # Translate coordinate to a 3-byte message
        msg = self.coord_to_msg(int(target_pos))
        self.observer.write(bytearray([motor, 84, 3, msg[0], msg[1], msg[2], 58]))  # [device number, command, 3, message, stop signal]
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
        # self.observer.write(bytearray([71, 58]))
        # self.reset()
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
        self.AFG3000 = tfg.FuncGen(instrument_descriptor).ch1

    def reset(self):

        self.set_vpp(vpp=MIN_VPP)
        self.set_frequency(frequency=MIN_FREQUENCY)
        self.set_waveform('SQUARE')
        self.turn_on()

        print(f'FG settings: {self.AFG3000.get_settings()}')

    def set_vpp(self, vpp: float):
        self.AFG3000.set_amplitude(vpp)


    def get_vpp(self):
        return self.AFG3000.get_amplitude()

    def set_frequency(self, frequency: float):
        self.AFG3000.set_frequency(frequency * 1e3)

    def get_frequency(self):
        return self.AFG3000.get_frequency()

    def set_waveform(self, waveform: str):
        assert waveform in ['SIN', 'SQUARE', 'RAMP'], f'Invalid waveform: {waveform}'
        self.AFG3000.set_function(waveform)

    def get_waveform(self):
        return self.AFG3000.get_function()

    def turn_on(self):
        self.AFG3000.set_output("ON")

    def turn_off(self):
        self.AFG3000.set_output("OFF")


class SwarmEnv:

    def __init__(self,
                 target_points=TARGET_POINTS,
                 metadata=METADATA):

        # Initialize devices
        self.source = VideoStreamHammamatsu()  # Camera
        self.actuator = ActuatorPiezos()  # Piezo's
        self.translator = TranslatorLeica()  # Leica xy-platform
        self.function_generator = FunctionGenerator()  # Function generator

        # Metadatastructure
        self.metadata = metadata
        self.model = calc_action

        # Initialize Vpp and frequency to their minima
        self.function_generator.reset()
        self.vpp = 20
        self.frequency = 2000  # kHz

        # Keep track of target point (idx in target_points)
        self.target_points = target_points
        self.target_idx = 0

        # Initialize memory
        self.memory = deque(maxlen=MEMORY_LENGTH)
        self.magic_vpp = 0

        # Set exit condition
        atexit.register(self.close)

    def draw_bbox(self, img):

        refPt = []

        def click_and_crop(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                refPt.append((x, y))
            elif event == cv2.EVENT_LBUTTONUP:
                refPt.append((x, y))
                cv2.rectangle(img, refPt[0], refPt[1], (0, 255, 0), 1)
                cv2.imshow("image", img)

        clone = img.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", click_and_crop)

        while True:
            # display the image and wait for a keypress
            cv2.imshow("image", img)
            key = cv2.waitKey(0)
            # If backspace, reset
            if key == ord("\x08"):
                img = clone.copy()
            # If enter, break loop
            elif key == ord("\r"):
                break

        cv2.destroyAllWindows()

        return [refPt[0][0], refPt[0][1], refPt[1][0] - refPt[0][0], refPt[1][1] - refPt[0][1]]

    def draw_targets(self, img):

        refPt = []

        def click_and_crop(event, x, y, flags, param):

            if event == cv2.EVENT_LBUTTONDOWN:
                refPt.append((x, y))
                cv2.circle(img, (x, y), 0, (255, 120, 0), 5)
                cv2.imshow("image", img)


        clone = img.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", click_and_crop)

        while True:
            # display the image and wait for a keypress
            cv2.imshow("image", img)
            key = cv2.waitKey(0)
            # if the 'r' key is pressed, reset the cropping region
            if key == ord("\x08"):
                img = clone.copy()
            # If enter, break loop
            elif key == ord("\r"):
                break

        cv2.destroyAllWindows()

        return refPt

    def reset(self):

        # Set env steps to 0
        self.step = 0

        # Initialize function generator
        self.function_generator.reset()

        # Get time
        self.now = round(time.time(), 3)

        # Define file name
        filename = SAVE_DIR + f"{self.now}-reset.png"

        # Snap a frame from the video stream and save
        img = self.source.snap(f_name=filename)
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # TODO --> check if this works

        # Draw bbox around swarm to track
        bbox = np.array(np.array(self.draw_bbox(img=img)), dtype=int).tolist()

        # Manualy add target points
        targets = np.array(np.array(self.draw_targets(img=img)), dtype=int).tolist()
        if targets:
            self.target_points = targets
            self.target_idx = 0

        # Initialize tracking algorithm
        self.tracker = TrackClusters(bbox=bbox)

        # Get the centroid of (biggest) swarm
        self.state, self.size = self.tracker.reset(img=img)

        # Add state to memory
        self.memory.append(np.array(self.state))

        # Add metadata to dataframe
        self.metadata = self.metadata.append(
            {"Filename": filename,
             "Time": self.now,
             "Vpp": self.vpp,
             "Frequency": self.frequency,
             "Size": self.size,
             "Action": None,
             "State": self.state,
             "Target": self.target_points[self.target_idx],
             "Step": self.step,
             "OFFSET_BOUNDS": OFFSET_BOUNDS,
             "MEMORY_LENGTH": MEMORY_LENGTH,
             "THRESHOLD_SPEED": THRESHOLD_SPEED,
             "MIN_VPP": MIN_VPP,
             "MAX_VPP": MAX_VPP,
             "MIN_FREQUENCY": MIN_FREQUENCY},
             ignore_index=True
        )

        # Return centroids of n amount of swarms
        return self.state

    def env_step(self):

        # Save metadata every X amount of steps
        if not self.step % 50:
            self.metadata.to_csv(metadata_filename)

        # Get time
        self.now = round(time.time(), 3)

        # Define file name
        filename = SAVE_DIR + f"{self.now}.png"

        # Snap a frame from the video stream
        img = self.source.snap(f_name=filename)
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # TODO --> check if this works

        # Get the new state
        self.state, self.size = self.tracker.update(img=img,  # Read image
                                                    target=self.target_points[self.target_idx],  # For verbose purposes
                                                    verbose=True)  # Show live tracking

        # Exception handling
        if not self.state:
            self.state = (0, 0)

        # Only update function generator and arduino every UPDATE_ENV_EVERY steps
        if not self.step % UPDATE_ENV_EVERY:

            if self.step != 0:
                print(f'FPS: {1 / ((time.time() - self.t0) / UPDATE_ENV_EVERY)}')

            # self.actuator.arduino.write(b"9")
            self.action = self.model(pos0=self.state,
                                     offset=np.array(self.state) - np.array(self.target_points[self.target_idx]),
                                     mode='avg')

            self.function_generator.set_frequency(frequency=PIEZO_RESONANCES[self.action])

            # Actuate piezos
            self.actuator.move(self.action)

            self.t0 = time.time()

        # Add state to memory
        self.memory.append(np.array(self.state))

        # Add metadata to dataframe
        self.metadata = self.metadata.append(
            {"Filename": filename,
             "Time": self.now,
             "Vpp": self.vpp,
             "Frequency": self.frequency,
             "Size": self.size,
             "Action": self.action,
             "State": self.state,
             "Target": self.target_points[self.target_idx],
             "Step": self.step,
             "OFFSET_BOUNDS": OFFSET_BOUNDS,
             "MEMORY_LENGTH": MEMORY_LENGTH,
             "THRESHOLD_SPEED": THRESHOLD_SPEED,
             "THRESHOLD_DIRECTION": THRESHOLD_DIRECTION,
             "MIN_VPP": MIN_VPP,
             "MAX_VPP": MAX_VPP,
             "MIN_FREQUENCY": MIN_FREQUENCY,
             "MAX_FREQUENCY": MAX_FREQUENCY},
             ignore_index=True
        )

        # # Move microscope to next point if offset goes into bounds
        if np.linalg.norm(np.array(self.state) - np.array(self.target_points[self.target_idx])) < OFFSET_BOUNDS:
            self.target_idx = (self.target_idx + 1) % (len(self.target_points))

        # Add step
        self.step += 1

        # Return centroids of n amount of swarms
        return self.state

    # TODO --> There is a lot of double euclidean distance calculation in this function now. Maybe make this smarter???
    def set_vpp_and_frequency(self):

        # Set Vpp based on distance from target
        self.vpp = np.sqrt(np.linalg.norm(self.memory[-1] -
                           np.array(self.target_points[self.target_idx])) /
                           (np.sqrt(2)*0.5*IMG_SIZE)) * \
                           (MAX_VPP - MIN_VPP) + \
                           MIN_VPP
        self.vpp += self.magic_vpp
        self.vpp = min(self.vpp, MAX_VPP)

        self.function_generator.set_vpp(vpp=self.vpp)

        # Calculate average direction of target position
        # target_offsets = np.array(self.target_points[self.target_idx]) - np.array(self.memory)
        # avg_direction_target = np.average(a=target_offsets / np.linalg.norm(target_offsets,
        #                                   axis=1).reshape((len(self.memory), 1)), axis=0)
        #
        # Calculate average direction of swarm movement
        # movement_offsets = np.array(self.memory)[1:] - np.array(self.memory)[:-1]
        # movement_speeds = np.linalg.norm(x=movement_offsets,
        #                                  axis=1)
        # avg_direction_movement = np.mean(a=np.nan_to_num(x=movement_offsets / movement_speeds.reshape((len(self.memory) - 1, 1))),
        #                                  axis=0)
        # avg_direction_movement = np.nan_to_num(x=avg_direction_movement / np.linalg.norm(avg_direction_movement))
        #
        # # Set frequency if we don't move at a certain speed
        # if len(self.memory) > 1:
        #
        #     # If movement is slow
        #     if np.average(movement_speeds) < THRESHOLD_SPEED:
        #         print('We are stationary')
        #         self.frequency += 4
        #         print('Magic is happening')
        #         self.magic_vpp += 1
        #
        #     # If movement is not in direction of target
        #     elif np.any(np.abs(avg_direction_target - avg_direction_movement) > THRESHOLD_DIRECTION):
        #         print('We are crooked')
        #         self.frequency += 2
        #     else:
        #         self.magic_vpp = 0
        #
        #     # Make sure the frequency does not go out of bounds
        #     if self.frequency > MAX_FREQUENCY:
        #         self.frequency = MIN_FREQUENCY
        #
        #     # Set frequency to function generator
        #     self.function_generator.set_frequency(frequency=self.frequency)

    def close(self):
        print(f'Final bbox: {self.tracker.bbox}')  # Print final bounding box, for if we want to continue tracking the same swarm
        self.metadata.to_csv(metadata_filename)  # Save metadata
        self.actuator.close()  # Close communication
        self.translator.close()  # Close communication
        self.function_generator.turn_off()
        cv2.destroyAllWindows()


class DataGatherEnv:

    def __init__(self,
                 source=VideoStreamHammamatsu(),
                 actuator=ActuatorPiezos(),
                 translator=TranslatorLeica(),
                 function_generator=FunctionGenerator(),
                 metadata=METADATA):

        # Initialize devices
        self.source = source  # Camera
        self.actuator = actuator  # Piezo's
        self.translator = translator  # Leica xy-platform
        self.function_generator = function_generator  # Function generator

        self.function_generator.set_waveform('SQUARE')
        self.function_generator.turn_on()

        # Metadata structure
        # self.metadata = metadata
        self.metadata = pd.DataFrame(
            {"Filename": "Initial",
             "Time": (None, None),
             "Vpp": None,
             "Frequency": None,
             "Action": None}

        )
        # self.metadata = self.metadata.dropna(axis=1, how='all')

        # Set exit condition
        atexit.register(self.close)

    def env_step(self, action, vpp, frequency):

        ### Alter freq, vpp, action ###

        # Get time
        self.now = round(time.time(), 3)

        # Define file name
        filename = SAVE_DIR + f"{self.now}.png"

        # Snap a frame from the video stream
        self.source.snap(f_name=filename)
        img = cv2.imread(filename)
        cv2.imshow('Image', img)
        cv2.waitKey(1)

        # Add metadata to dataframe
        self.metadata = self.metadata.append(
            {"Filename": filename,
             "Time": self.now,
             "Vpp": vpp,
             "Frequency": frequency,
             "Action": action},
             ignore_index=True
        )

    def close(self):
        self.metadata.to_csv(metadata_filename)  # Save metadata
        self.actuator.close()  # Close communication
        self.translator.close()  # Close communication
        self.function_generator.turn_off()
        cv2.destroyAllWindows()
