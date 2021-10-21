import numpy as np
from settings import *
import matplotlib.pyplot as plt

def random_action(nr_actions=4):
    return np.random.randint(low=0, high=nr_actions)

# TODO --> Test if absolute or relative state works best, or if euclidean distance is better
def predict_state(Vpp: float, frequency: int, size: int, model) -> tuple:
    """
    Calculate and return predicted state some time in the future, depending on the model.
    :param Vpp: Voltage of function generator
    :param frequency: Frequency of function generator
    :param size: Size of swarm (width or height of the bounding box)
    :return: Predicted magnitude of motion between 0 and 1 (1 representing the maximum of 300 pixels of movement)
    """
    return model.predict(np.array([[Vpp/MAX_VPP, frequency/MAX_FREQUENCY, size/SIZE_NORMALIZING_FACTOR]][0]))

def get_action(size: int, offset_to_target: tuple):
    """
    Calculate best vpp and frequency for a given size of swarm and distance to target
    + calculate best piezo to actuate
    :param size: Size of swarm
    :param offset_to_target: (x, y) offset to target
    :return: Vpp, frequency and choice of piezo
    """

    # Generate potential combinations of inputs to the model
    steps = [[Vpp, frequency, size] for Vpp in np.linspace(MIN_VPP, MAX_VPP, int((MAX_VPP - MIN_VPP) / VPP_STEP_SIZE + 1)) for frequency in np.linspace(MIN_FREQUENCY, MAX_FREQUENCY, int((MAX_FREQUENCY - MIN_FREQUENCY) / FREQUENCY_STEP_SIZE + 1))]

    # Map all combinations to model to get the predicted motion
    results = np.array(list(map(predict_state, steps)))  # TODO --> Optimize

    # Find the Vpp and frequency that belong to the best inputs
    # 'Best' is defined as the input that brings the robot closest to the target in either the x or y direction
    Vpp, frequency, size = steps[np.abs(results - np.max(np.abs(offset_to_target))).argmin()]

    # Choose one of four piezos
    # IMPORTANT --> 0: dx > 0 (move left), 1: dy > 0 (move up), 2: dx < 0 (move right), 3: dy < 0 (move down)
    action = np.argmax(np.abs(offset_to_target))
    if np.sign(offset_to_target[action]) == -1:
        action += 2

    return action, Vpp, frequency


def walk_to_pixel(blob_pos, target_pos):

    if not blob_pos:
        return random_action()

    offsets = - np.subtract(target_pos[0], blob_pos[0]), - np.subtract(target_pos[1], blob_pos[1])

    # Get the largest offset and compute action
    action = np.argmax(np.abs(offsets))
    if np.sign(offsets[action]) == -1:
        action += 2

    return action