# Communication settings
SAVE_DIR = "C:\\Users\\ARSL\\PycharmProjects\\Project_Matt\\snapshots\\"
SERIAL_PORT_ARDUINO = "COM5"
INSTR_DESCRIPTOR = 'USB0::0x0699::0x034F::C020081::INSTR'
SERIAL_PORT_LEICA = "COM4"
BAUDRATE = 115200
# STREAM_URL = "rtsp://10.4.51.109"

# Environment settings
IMG_SIZE = 300
OFFSET_BOUNDS = 5
SLEEP_TIME = 0.03
PIXEL_MULTIPLIER_LEFT_RIGHT = 75
PIXEL_MULTIPLIER_UP_DOWN = 85
TARGET_POINTS = [(150, 100), (100, 150), (150, 200), (200, 150)]
EXPOSURE_TIME = 25
UPDATE_ENV_EVERY = 16

# Data settings
import pandas
METADATA = pandas.read_csv("metadata.csv")  # Make this file if you don't have it yet
del METADATA['Unnamed: 0']

# Model settings
from tensorflow.keras.models import load_model
MODEL_NAME = 'predict_state'
MODEL = load_model(f"C:\\Users\\ARSL\\PycharmProjects\\Project_Matt\\models\\" + MODEL_NAME)
SIZE_NORMALIZING_FACTOR = 400  # TODO --> Optimize
VPP_STEP_SIZE = 0.2
FREQUENCY_STEP_SIZE = 1

# PID
THRESHOLD_SPEED = 0.75
MIN_VPP = 3
MAX_VPP = 10
MIN_FREQUENCY = 230
MAX_FREQUENCY = 270

# Run settings
MAX_STEPS = 10000
EPISODES = 1
UPDATE_VPP_EVERY = 10

