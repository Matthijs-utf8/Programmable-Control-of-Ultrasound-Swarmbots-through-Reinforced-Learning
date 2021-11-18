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
BAUDRATE_LEICA = 9600  # Baudrate Leica
SLEEP_TIME = 0.03  # Time Leica takes to wait for next command
PIXEL_MULTIPLIER_LEFT_RIGHT = 75  # Pixel --> step size correction in x
PIXEL_MULTIPLIER_UP_DOWN = 85  # Pixel --> step size correction in y

# General nvironment settings
SAVE_DIR = "C:\\Users\\ARSL\\PycharmProjects\\Project_Matt\\data_standardized_4\\"  # Location for images
IMG_SIZE = 300  # Size of environment (IMG_SIZE, IMG_SIZE)
OFFSET_BOUNDS = 10  # Minimum Euclidean distance for reaching checkpoint
TARGET_POINTS = [(90, 275), (25, 210)]  # Checkpoints
UPDATE_ENV_EVERY = 100  # Update Vpp and frequency every UPDATE_ENV_EVERY steps

# Model settings
# from tensorflow.keras.models import load_model
import pickle
MODEL_NAME = 'rx_reg_with_forces2.pkl'  # Name of model
MODEL = pickle.load(open(f"C:\\Users\\ARSL\\PycharmProjects\\Project_Matt\\venv\\Include\\AI_Actuated_Micrswarm_4\\" + MODEL_NAME, 'rb'))  # Load model
print('Model loaded :)')
# SIZE_NORMALIZING_FACTOR = 400  # TODO --> Optimize
VPP_STEP_SIZE = 0.5  # Step size of vpp
FREQUENCY_STEP_SIZE = 2  # Step size of frequency

# PID
MEMORY_LENGTH = UPDATE_ENV_EVERY  # Length of the amount of steps used in the running average state of swarm
THRESHOLD_SPEED = 0.5  # Minimum equilibrium speed
THRESHOLD_DIRECTION = 0.75  # Minimum movement direction
MIN_VPP = 2  # Minimum Vpp
MAX_VPP = 8  # Maximum Vpp
MIN_FREQUENCY = 240  # Minimum frequency
MAX_FREQUENCY = 260  # Maximum frequency

# Run settings
MAX_STEPS = 10000  # Number of consecutive steps in an episode
EPISODES = 1  # Number of episodes

# Data settings
import pandas
metadata_filename = "new_model_tests_16_11_2021.csv"
try:
    METADATA = pandas.read_csv(metadata_filename)  # Make this file if you don't have it yet
    del METADATA['Unnamed: 0']  # Remove unwanted column
except:
    METADATA = pandas.DataFrame(
        {"Filename": "",
         "Time": None,
         "Vpp": None,
         "Frequency": None,
         "Size": None,
         "Action": None,
         "State": (None, None),
         "Target": (None, None),
         "Step": None,
         "OFFSET_BOUNDS": None,
         "MEMORY_LENGTH": None,
         "THRESHOLD_SPEED": None,
         "THRESHOLD_DIRECTION": None,
         "MIN_VPP": None,
         "MAX_VPP": None,
         "MIN_FREQUENCY": None,
         "MAX_FREQUENCY": None
         }
    )