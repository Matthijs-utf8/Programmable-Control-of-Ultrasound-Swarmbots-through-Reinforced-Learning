# Arduino settings
SERIAL_PORT_ARDUINO = "COM5"  # Communication port with Arduino
BAUDRATE_ARDUINO = 115200  # Baudrate Arduino

# Hammamatsu settings
INSTR_DESCRIPTOR = 'USB0::0x0699::0x034F::C020081::INSTR'  # Name of Tektronix function generator
EXPOSURE_TIME = 25  # Exposure time Hammamatsu

# Kronos settings
STREAM_URL = "rtsp://10.4.51.109"  # RTSP stream url of Kronos

# Leica settings
SERIAL_PORT_LEICA = "COM4"  # Communication port Leica
BAUDRATE_LEICA = 9600 # Baudrate Leica
SLEEP_TIME = 0.03  # Time Leica takes to wait for next command
PIXEL_MULTIPLIER_LEFT_RIGHT = 75  # Pixel --> step size correction in x
PIXEL_MULTIPLIER_UP_DOWN = 85  # Pixel --> step size correction in y

# General nvironment settings
SAVE_DIR = "C:\\Users\\ARSL\\PycharmProjects\\Project_Matt\\snapshots\\"  # Location for images
IMG_SIZE = 300  # Size of image (IMG_SIZE, IMG_SIZE)
OFFSET_BOUNDS = 5  # Minimum Euclidean distance for reaching checkpoint
TARGET_POINTS = [(150, 100), (100, 150), (150, 200), (200, 150)]  # Checkpoints
UPDATE_ENV_EVERY = 16  # Update Vpp and frequency every UPDATE_ENV_EVERY steps

# Data settings
import pandas
METADATA = pandas.read_csv("metadata.csv")  # Make this file if you don't have it yet
del METADATA['Unnamed: 0']  # Remove unwanted column

# Model settings
from tensorflow.keras.models import load_model
MODEL_NAME = 'predict_state'  # Name of model
MODEL = load_model(f"C:\\Users\\ARSL\\PycharmProjects\\Project_Matt\\models\\" + MODEL_NAME)  # Load model
SIZE_NORMALIZING_FACTOR = 400  # TODO --> Optimize
VPP_STEP_SIZE = 0.2  # Step size of vpp
FREQUENCY_STEP_SIZE = 1  # Step size of frequency

# PID
THRESHOLD_SPEED = 0.75  # Minimum equilibrium speed
MIN_VPP = 3  # Minimum Vpp
MAX_VPP = 10  # Maximum Vpp
MIN_FREQUENCY = 230  # Minimum frequency
MAX_FREQUENCY = 270 # Maximum frequency

# Run settings
MAX_STEPS = 1000  # Number of consecutive steps in an episode
EPISODES = 1  # Number of episodes