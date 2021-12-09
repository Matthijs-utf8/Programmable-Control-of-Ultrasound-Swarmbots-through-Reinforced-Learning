# Arduino settings
SERIAL_PORT_ARDUINO = "COM5"  # Communication port with Arduino
BAUDRATE_ARDUINO = 115200  # Baudrate Arduino

# Hammamatsu settings
EXPOSURE_TIME = 25  # Exposure time Hammamatsu

# Tektronix settings
INSTR_DESCRIPTOR = 'USB0::0x0699::0x034F::C020081::INSTR'  # Name of Tektronix function generator

# Kronos settings
STREAM_URL = "rtsp://10.4.51.109"  # RTSP stream url of Kronos

# Leica settings
SERIAL_PORT_LEICA = "COM4"  # Communication port Leica
BAUDRATE_LEICA = 9600  # Baudrate Leica
SLEEP_TIME = 0.03  # Time Leica takes to wait for next command
PIXEL_MULTIPLIER_LEFT_RIGHT = 75  # Pixel --> step size correction in x
PIXEL_MULTIPLIER_UP_DOWN = 85  # Pixel --> step size correction in y

# General environment settings
MAX_STEPS = 10000  # Number of consecutive steps in an episode
IMG_SIZE = 300  # Size of environment/image (IMG_SIZE, IMG_SIZE)
OFFSET_BOUNDS = 10  # Minimum Euclidean distance for reaching checkpoint
TARGET_POINTS = [(90, 275), (25, 210)]  # Checkpoints
UPDATE_ENV_EVERY = 1  # Update Vpp and frequency every UPDATE_ENV_EVERY steps

# Model settings
import pickle
import numpy as np
models_folder = 'C:\\Users\\ARSL\\PycharmProjects\\Project_Matt\\venv\\Include\\AI_Actuated_Micrswarm_4\\models'
model_name = 'Vector_fields.npy'
VECT_FIELDS = np.load(f"{models_folder}\\{model_name}")
# MODEL_NAME = 'DecisionTreeRegressor.pkl'  # Name of model
# MODEL = pickle.load(open(f"C:\\Users\\ARSL\\PycharmProjects\\Project_Matt\\venv\\Include\\AI_Actuated_Micrswarm_4\\" + MODEL_NAME, 'rb'))  # Load model
PIEZO_RESONANCES = {0: 2350, 1: 1500, 2: 2050, 3: 1900}

# PID
# MEMORY_LENGTH = UPDATE_ENV_EVERY  # Length of the amount of steps used in the running average state of swarm

# Data location settings
import datetime
import os
PROJECT_NAME = 'Project_Matt' # Project name (use only one project name per person, this makes it easy to keep track)
DATE = datetime.date.today() # Todays date, for keeping track of the experiments
EXPERIMENT_RUN_NAME = 'TEST' # Use a descriptive name here so you know what you did during the experiment
SAVE_DIR = f"C:\\Users\\ARSL\\PycharmProjects\\{PROJECT_NAME}\\{DATE}"  # Location for images all the images and metadata
SNAPSHOTS_SAVE_DIR = f'{SAVE_DIR}\\{EXPERIMENT_RUN_NAME}\\' # For saving metadata from experimental run
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)  # For saving all data from TODAY
if not os.path.isdir(SNAPSHOTS_SAVE_DIR):
    os.mkdir(SNAPSHOTS_SAVE_DIR)  # For saving snapshots from one experimental run

# Data settings
import pandas
metadata_filename = f"{SAVE_DIR}{EXPERIMENT_RUN_NAME}.csv"
try:
    METADATA = pandas.read_csv(metadata_filename)  # Make this file if you don't have it yet
    del METADATA['Unnamed: 0']  # Remove unwanted column
except:
    METADATA = pandas.DataFrame()