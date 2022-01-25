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
MAX_STEPS = 20000  # Number of consecutive steps in an episode
IMG_SIZE = 300  # Size of environment/image (IMG_SIZE, IMG_SIZE)
OFFSET_BOUNDS = 5  # Minimum Euclidean distance to satisfy checkpoint condition
TARGET_POINTS = []  # Checkpoints
UPDATE_RATE_ENV = 5  # Update rate environment (frames)
SAVE_RATE_METADATA = 50  # Update rate metadata csv (frames)
PIEZO_RESONANCES = {0: 2350, 1: 1500, 2: 2000, 3: 1900}  # kHz

# Model settings
import numpy as np
# MODELS_FOLDER = 'C:\\Users\\ARSL\\PycharmProjects\\Project_Matt\\venv\\Include\\AI_Actuated_Micrswarm_4\\models'
MODELS_FOLDER = "C:\\Users\\Matthijs\\PycharmProjects\\ARSL_Autonomous_Navigation\\models"
MODEL_NAME = 'Circles_final_week.npy'
# Q_VALUES_INITIAL = np.load(f"{MODELS_FOLDER}\\{MODEL_NAME}")
Q_VALUES_INITIAL = np.zeros((4, IMG_SIZE, IMG_SIZE, 2))
Q_VALUES_INITIAL[0, :, :, 0] -= 1
Q_VALUES_INITIAL[1, :, :, 1] += 1
Q_VALUES_INITIAL[2, :, :, 0] += 1
Q_VALUES_INITIAL[3, :, :, 1] -= 1
MAX_VELO = 10
Q_VALUES_UPDATE_KERNEL_FUNC = lambda x, y: np.log(x**2 + y**2 + 1)
xx = np.linspace(-IMG_SIZE, IMG_SIZE-1, int(IMG_SIZE*2))
yy = np.linspace(-IMG_SIZE, IMG_SIZE-1, int(IMG_SIZE*2))
xx, yy = np.meshgrid(xx, yy)
Q_VALUES_UPDATE_KERNEL = np.array(list(map(Q_VALUES_UPDATE_KERNEL_FUNC, xx, yy)))
Q_VALUES_UPDATE_KERNEL = np.repeat(np.abs((Q_VALUES_UPDATE_KERNEL / np.max(Q_VALUES_UPDATE_KERNEL)) - 1)[:, :, np.newaxis], 2, axis=2)
UPDATE_RATE_Q_VALUES = UPDATE_RATE_ENV  # Update rate Q values (frames)
MAX_MEM_LEN = UPDATE_RATE_ENV  # Max length of memory (datapoints)
GAMMA = 0.9  # Discount factor
EPSILON = 0.01  # Exploration coefficient

# Data location settings
import datetime
import os
PROJECT_NAME = 'Project_Matt'  # Project name (use only one project name per person, this makes it easy to keep track)
DATE = datetime.date.today() # Todays date, for keeping track of the experiments
EXPERIMENT_RUN_NAME = 'Circles_final_week'  # Use a descriptive name here so you know what you did during the experiment
# SAVE_DIR = f"C:\\Users\\ARSL\\PycharmProjects\\{PROJECT_NAME}\\{DATE}"  # Location for images all the images and metadata
SAVE_DIR = f"E:\\{DATE}"  # Location for images all the images and metadata
SNAPSHOTS_SAVE_DIR = f'{SAVE_DIR}\\{EXPERIMENT_RUN_NAME}\\'  # For saving metadata from experimental run
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)  # For saving all data from TODAY
if not os.path.isdir(SNAPSHOTS_SAVE_DIR):
    os.mkdir(SNAPSHOTS_SAVE_DIR)  # For saving snapshots from one experimental run

# Data settings
import pandas
METADATA_FILENAME = f"{SAVE_DIR}\\{EXPERIMENT_RUN_NAME}.csv"
try:
    METADATA = pandas.read_csv(METADATA_FILENAME)  # Make this file if you don't have it yet
    del METADATA['Unnamed: 0']  # Remove unwanted column
except:
    METADATA = pandas.DataFrame()