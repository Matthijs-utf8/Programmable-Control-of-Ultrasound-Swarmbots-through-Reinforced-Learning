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

def update_q_values(action, memory, q_values):

    # Filter for action
    if action in [0, 1, 2, 3]:

        # Get mean position of memory
        mean_pos = np.mean(memory, 0, dtype=int)

        # Calculate average direction of swarm movement
        avg_speed = np.mean(np.array(memory)[1:] - np.array(memory)[:-1], axis=0)

        print(mean_pos)

        if mean_pos[0] < MAX_VELO:
            x_slice = slice(0, mean_pos[0] + MAX_VELO)
            x_slice_kernel = slice(MAX_VELO - (mean_pos[0] % MAX_VELO), -1)
        elif mean_pos[0] > IMG_SIZE - MAX_VELO:
            x_slice = slice(mean_pos[0]-MAX_VELO, IMG_SIZE)
            x_slice_kernel = slice(0, 2*MAX_VELO - (mean_pos[0] % MAX_VELO))
        else:
            x_slice = slice(mean_pos[0]-MAX_VELO, mean_pos[0]+MAX_VELO)
            x_slice_kernel = slice(0, 100)

        if mean_pos[1] < MAX_VELO:
            y_slice = slice(0, mean_pos[1] + MAX_VELO)
            y_slice_kernel = slice(MAX_VELO - (mean_pos[1] % MAX_VELO), -1)
        elif mean_pos[1] > IMG_SIZE - MAX_VELO:
            y_slice = slice(mean_pos[1]-MAX_VELO, IMG_SIZE)
            y_slice_kernel = slice(0, 2*MAX_VELO - (mean_pos[1] % MAX_VELO))
        else:
            y_slice = slice(mean_pos[1]-MAX_VELO, mean_pos[1]+MAX_VELO)
            y_slice_kernel = slice(0, 100)

        # Define Region of Interest
        ROI = [action,
               slice(max(mean_pos[0] - MAX_VELO, 0), min(mean_pos[0] + MAX_VELO, q_values.shape[1])),
               slice(max(mean_pos[1] - MAX_VELO, 0), min(mean_pos[1] + MAX_VELO, q_values.shape[2])),
               slice(0, 2)]

        # Define kernel for updating function of the q values
        kernel = Q_VALUES_UPDATE_KERNEL.copy()[x_slice_kernel, y_slice_kernel, slice(0, 2)] * avg_speed

        # Update q values
        q_values[ROI] += 0.1 * kernel * avg_speed

        return q_values

    else:
        return q_values

# if __name__ == '__main__':
#     update_q_values(0, [(26, 27), (28, 29)], q_values=Q_VALUES_INITIAL)

def calc_action(pos0, offset, q_values=None, mode='naive'):
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
        return (action + 2) % 4

    if mode == 'straight_line':
        action = np.random.choice((0, 1), p=(np.abs(offset)/np.sum(np.abs(offset))))
        if not np.sign(offset[action]) == -1:
            action += 2
        return (action + 2) % 4

    # Choose action from single vector in pos0
    elif mode == 'single_choice':
        single_point_ROI = q_values[:, pos0[0], pos0[1], :]
        action = np.argmin(np.linalg.norm(single_point_ROI - offset, axis=1))
        return (action + 2) % 4

    # Choose action from ROI of the vector field based on the minimum of ROI - offset
    elif mode == 'max':
        ROI = q_values[:, slice(max(pos0[0] - MAX_VELO, 0), max(pos0[0] + MAX_VELO, 0)),
                             slice(max(pos0[1] - MAX_VELO, 0), max(pos0[1] + MAX_VELO, 0)),
                             slice(0, 2)]
        action = np.unravel_index(np.argmin(np.linalg.norm(ROI - offset, axis=-1)), ROI.shape)[0]
        return (action + 2) % 4

    # Choose action from ROI of the vector field based on the average of ROI - offset
    elif mode == 'avg':
        ROI = q_values[:, slice(max(pos0[0] - MAX_VELO, 0), max(pos0[0] + MAX_VELO, 0)),
                             slice(max(pos0[1] - MAX_VELO, 0), max(pos0[1] + MAX_VELO, 0)),
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