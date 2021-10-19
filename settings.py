# Communication settings
SAVE_DIR = "C:\\Users\\ARSL\\PycharmProjects\\Project_Matt\\snapshots\\"
SERIAL_PORT_ARDUINO = "COM5"
# INSTR_DESCRIPTOR = 'USB0::0x0699::0x034F::C020081::INSTR'
INSTR_DESCRIPTOR = None
SERIAL_PORT_LEICA = "COM4"
BAUDRATE = 115200
# STREAM_URL = "rtsp://10.4.51.109"

# Environment settings
IMG_SIZE = 300
OFFSET_BOUNDS = 10
SLEEP_TIME = 0.03
PIXEL_MULTIPLIER_LEFT_RIGHT = 75
PIXEL_MULTIPLIER_UP_DOWN = 85
TARGET_POINTS = [(150, 150), (150, 150)]
EXPOSURE_TIME = 25

# Data settings
import pandas
METADATA = pandas.read_csv("Metadata.csv")  # Make this file if you don't have it yet

# PID
THRESHOLD_SPEED = 1
MIN_VPP = 2
MAX_VPP = 4
MIN_FREQUENCY = 240
MAX_FREQUENCY = 260

# Run settings
MAX_STEPS = 1
EPISODES = 1
UPDATE_VPP_EVERY = 10

