from postprocessing.cluster_detection_and_tracking import find_clusters, TrackClusters
import numpy as np
import pandas as pd
import os
import cv2
import tqdm
import os
from collections import deque
from manipulation.settings import *
from ast import literal_eval as make_tuple
import matplotlib.pyplot as plt
import time
#
# def moving_average(a, n=3) :
#     ret = np.cumsum(a, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n
#
# memory = deque(maxlen=500)
#
# running_speeds = []
# running_positions = []
# running_angle = []
# running_target_angle = []
# running_target = []
# running_position = []
#
# for n, datapoint in tqdm.tqdm(METADATA.iterrows()):
#
#     if "reset" in datapoint["Filename"]:
#         print("Reset")
#         memory = deque(maxlen=50)
#         running_speeds = []
#         running_positions = []
#         running_angle = []
#         running_target_angle = []
#         running_target = []
#         running_position = []
#
#     # Add to memory
#     memory.append(make_tuple(datapoint["State"]))
#     mean_position = np.mean(memory, axis=0)
#     running_positions.append(mean_position.tolist())
#
#     # Calculate speed and direction of swarm
#     movement_offsets = np.array(memory)[1:] - np.array(memory)[:-1]
#     if not len(movement_offsets):
#         continue
#     mean_offset = np.mean(movement_offsets, axis=0)
#     avg_direction_movement = np.degrees(np.arctan(mean_offset[1] / (mean_offset[0] + 1e-10)))
#     mean_speed = np.linalg.norm(mean_offset)
#
#     running_angle.append(avg_direction_movement)
#     running_speeds.append(mean_speed)
#
#     # Calculate direction to target and add to memory
#     target_offsets = np.array(make_tuple(datapoint["Target"])) - mean_position
#     avg_direction_target = target_offsets / np.linalg.norm(target_offsets)
#     running_target_angle.append(np.degrees(np.arctan(avg_direction_target[1] / (avg_direction_target[0]+1e-8))))
#
#     running_target.append(make_tuple(datapoint["Target"]))
#     running_position.append(make_tuple(datapoint["State"]))
#
# plt.figure(1)
# plt.plot(running_speeds)
#
# plt.figure(2)
# plt.plot(np.array(running_target_angle) - np.array(running_angle))
#
# # plt.figure(2)
# # plt.plot(running_target_angle)
#
#
#
# plt.figure(3)
# plt.plot(np.array(running_target)[:, 0])
#
# plt.figure(3)
# plt.plot(np.array(running_position)[:, 0])
#
# plt.figure(4)
# plt.plot(np.array(running_target)[:, 1])
#
# plt.figure(4)
# plt.plot(np.array(running_position)[:, 1])
#
# plt.figure(5)
# plt.plot(np.array(running_target)[:, 1] - np.array(running_position)[:, 1])
#
# plt.figure(6)
# plt.plot(np.array(running_target)[:, 0] - np.array(running_position)[:, 0])
#
# plt.figure(7)
# plt.plot(moving_average(np.linalg.norm(np.array(running_target) - np.array(running_position), axis=1), n=200))
#
#
# plt.show()



import numpy as np
import matplotlib.pyplot as plt
# from manipulation.settings import *

# env_size = 300
# step_size = 12
# piezo = 0
#
# state = tuple((15, 278))
# kernel_slice_x = slice(300-state[1], 600-state[1]-1)
# kernel_slice_y = slice(300-state[0], 600-state[0]-1)
#
#
#
#
#
# xx = np.linspace(-300, 299, 600)
# yy = np.linspace(-300, 299, 600)
# xx, yy = np.meshgrid(xx, yy)
# # A,B = np.meshgrid(range(150),range(150))
#
# print(Q_VALUES_UPDATE_KERNEL.shape)
#
# update_func = lambda x, y: np.log(x**2 + y**2 + 1)
#
# arr = np.array(list(map(update_func, xx, yy)))
# arr = np.abs((arr / np.max(arr)) - 1)
# # plt.figure(0, figsize=(20, 20))
# # plt.quiver(xx, yy, Q_VALUES_UPDATE_KERNEL[:, :, 0], Q_VALUES_UPDATE_KERNEL[:, :, 1])
# # plt.tight_layout()
# # plt.show()
# print(arr.shape)
# plt.imshow(arr[kernel_slice_x, kernel_slice_y])
# plt.colorbar()
# plt.show()

MODELS_FOLDER = "C:\\Users\\Matthijs\\PycharmProjects\\ARSL_Autonomous_Navigation\\models"


def quiver(q_values, piezo, step_size, env_size=300, axis=None, color="black"):

    slices = slice(0, env_size - 1, step_size)

    xx = np.linspace(0, env_size - 1, int(env_size / step_size))
    yy = np.linspace(0, env_size - 1, int(env_size / step_size))
    xx, yy = np.meshgrid(xx, yy)

    # plt.figure(0, figsize=(20, 20))
    plt.title(f"{piezo}")
    axis.quiver(xx, yy, q_values[piezo, slices, slices, 0], q_values[piezo, slices, slices, 1], color=color)
    plt.tight_layout()
    # plt.show()

# q_values = np.load(f"{MODELS_FOLDER}\\1639400509.184_Vector_fields.npy")

# quiver(q_values=q_values, piezo=0, step_size=15)

def update_q_values(action, memory, q_values):

    # Filter for action
    if action in [0, 1, 2, 3]:

        # Get mean position of memory
        mean_pos = np.mean(memory, 0, dtype=int)

        # Calculate average direction of swarm movement
        avg_speed = np.mean(np.array(memory)[1:] - np.array(memory)[:-1], axis=0)

        kernel_slice_x = slice(300 - mean_pos[1], 600 - mean_pos[1])
        kernel_slice_y = slice(300 - mean_pos[0], 600 - mean_pos[0])

        # Update q values
        q_values[action] = GAMMA * q_values[action] + (1-GAMMA) * Q_VALUES_UPDATE_KERNEL[kernel_slice_x, kernel_slice_y] * avg_speed


        return q_values

    else:
        return q_values

def calc_action(pos0, offset, q_values=None, mode='naive'):
    """
    Calculate optimal piezo to actuate
    :param pos0:    Swarm position
    :param offset:  Offset to target
    :param mode:    Selection mode
    :return:        integer from 0 to NR_OF_PIEZOS
    """

    # if np.random.rand() < EPSILON:
    #     return random_action()

    # Same as walk_to_pixel function
    if mode == 'naive':
        action = np.argmax(np.abs(offset))
        if not np.sign(offset[action]) == -1:
            action += 2
        return (action + 2) % 4

    elif mode == 'straight_line':
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


if __name__ == "__main__":

    q_values = np.zeros((4, 300, 300, 2))
    q_values[0, :, :, 0] -= 1
    q_values[1, :, :, 1] += 1
    q_values[2, :, :, 0] += 1
    q_values[3, :, :, 1] -= 1

    mem_len = 5
    memory = deque(maxlen=(mem_len))

    action = -1

    # plt.imshow(Q_VALUES_UPDATE_KERNEL[:, :, 0])
    # plt.show()

    # Loop through datapoints
    for n, datapoint in tqdm.tqdm(METADATA.iterrows()):

        if "reset" in datapoint["Filename"]:
            memory = deque(maxlen=(mem_len))
            state = make_tuple(datapoint["State"])
            target = make_tuple(datapoint["Target"])
            memory.append(state)
            q_values = np.zeros((4, 300, 300, 2))
            q_values[0, :, :, 0] -= 1
            q_values[1, :, :, 1] -= 1
            q_values[2, :, :, 0] += 1
            q_values[3, :, :, 1] += 1
        else:
            state = make_tuple(datapoint["State"])
            memory.append(state)
            if not n % mem_len:

                q_values = update_q_values(q_values=q_values, memory=memory, action=int(action))

                target = make_tuple(datapoint["Target"])
                action = datapoint["Action"]
                offset = np.array(state) - np.array(target)

                # action = calc_action(pos0=state, offset=offset, q_values=q_values, mode="single_choice")



        if not n % mem_len and n > 0:
            fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
            m = 0
            for i in range(2):
                for j in range(2):
                    ax[i, j].set_title(f"{int(m)}")
                    ax[i, j].scatter(state[0], state[1], c="r")
                    ax[i, j].scatter(target[0], target[1], c="g")
                    if m == action:
                        quiver(q_values=q_values, piezo=m, step_size=15, axis=ax[i, j], color="green")
                    else:
                        quiver(q_values=q_values, piezo=m, step_size=15, axis=ax[i, j], color="blue")
                    m += 1
            plt.tight_layout()

            # redraw the canvas
            fig.canvas.draw()

            # convert canvas to image
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                                sep='')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            cv2.imshow("Test", img)
            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

