import numpy as np
from settings import *
import matplotlib.pyplot as plt

def random_action(nr_actions=4):
    """
    Choose random action from 0 to NR_OF_PIEZOS
    :param nr_actions:  NR_OF_PIEZOS
    :return:            integer from 0 to NR_OF_PIEZOS
    """
    return np.random.randint(low=0, high=nr_actions)

def calc_action(pos0, offset, mode='naive'):
    """
    Calculate optimal piezo to actuate
    :param pos0:    Swarm position
    :param offset:  Offset to target
    :param mode:    Selection mode
    :return:        integer from 0 to NR_OF_PIEZOS
    """
    # Same as walk_to_pixel function
    if mode == 'naive':
        action = np.argmax(np.abs(offset))
        if not np.sign(offset[action]) == -1:
            action += 2
        return action

    # Choose action from single vector in pos0
    elif mode == 'single_choice':
        single_point_ROI = VECT_FIELDS[:, pos0[0], pos0[1], :]
        action = np.argmin(np.linalg.norm(single_point_ROI - offset, axis=1))
        return (action + 2) % 4

    # Choose action from ROI of the vector field based on the minimum of ROI - offset
    elif mode == 'max':
        ROI = VECT_FIELDS[:, slice(max(pos0[0] - max_velo, 0), max(pos0[0] + max_velo, 0)),
                             slice(max(pos0[1] - max_velo, 0), max(pos0[1] + max_velo, 0)),
                             slice(0, 2)]
        action = np.unravel_index(np.argmin(np.linalg.norm(ROI - offset, axis=-1)), ROI.shape)[0]
        return (action + 2) % 4

    # Choose action from ROI of the vector field based on the average of ROI - offset
    elif mode == 'avg':
        ROI = VECT_FIELDS[:, slice(max(pos0[0] - max_velo, 0), max(pos0[0] + max_velo, 0)),
                             slice(max(pos0[1] - max_velo, 0), max(pos0[1] + max_velo, 0)),
                             slice(0, 2)]
        action = np.argmin(np.linalg.norm(np.average(ROI, axis=(1, 2)) - offset, axis=-1))
        return (action + 2) % 4

    else:
        raise ValueError(f'Mode {mode} is unvalid')

def walk_to_pixel(blob_pos, target_pos):

    offsets = - np.subtract(target_pos[0], blob_pos[0]), - np.subtract(target_pos[1], blob_pos[1])

    # Get the largest offset and compute action
    action = np.argmax(np.abs(offsets))
    if np.sign(offsets[action]) == -1:
        action += 2

    return action