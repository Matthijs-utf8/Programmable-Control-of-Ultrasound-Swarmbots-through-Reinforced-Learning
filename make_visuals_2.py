import cv2
import tqdm
import os
import time
import pandas
from ast import literal_eval as make_tuple
from collections import deque
import numpy as np
from settings import *

if __name__ == "__main__":

    metadata = pandas.read_csv("E:\\metadata.csv")  # Make this file if you don't have it yet
    del metadata['Unnamed: 0']

    metadata.fillna(-1, inplace=True)

    folder = f"E:\\snapshots_21_10_21\\"

    action_map = {0: "Move left", 1: "Move up", 2: "Move right", 3: "Move down", None: None, -1: 'Reset'}

    for i, datapoint in tqdm.tqdm(metadata.iterrows()):

        # if make_tuple(datapoint['Target']) != (150, 150):
        #     continue

        # Reset
        if '-reset.png' in datapoint['Filename']:
            centers = []
            colors = []
            memory = deque(maxlen=256)
            last_freq = 240
            img = cv2.imread(f"{folder}{datapoint['Time']}-reset.png", cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.imread(f"{folder}{datapoint['Time']}.png", cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Define state
        state = make_tuple(datapoint['State'])
        target = make_tuple(datapoint["Target"])
        size = datapoint['Size']
        action = datapoint['Action']
        bbox = [int(state[0]-0.5*size), int(state[1]-0.5*size), size, size]

        memory.append(state)
        # print(memory)

        if len(memory) > 1:

            # Calculate average direction of target position
            target_offsets = np.array(target) - np.array(memory)
            avg_direction_target = np.average(target_offsets / np.linalg.norm(target_offsets, axis=1).reshape((len(memory), 1)), axis=0)

            # Calculate average direction of swarm movement
            movement_offsets = np.array(memory)[1:] - np.array(memory)[:-1]
            movement_speeds = np.linalg.norm(movement_offsets, axis=1)
            avg_direction_movement = np.mean(np.nan_to_num(movement_offsets / movement_speeds.reshape((len(memory)-1, 1))), axis=0)
            avg_direction_movement = np.nan_to_num(avg_direction_movement / np.linalg.norm(avg_direction_movement))

            # print(target_offsets)
            # print(avg_direction_target)
            # print(movement_offsets)
            # print(movement_speeds)
            # print(np.nan_to_num(movement_offsets / movement_speeds.reshape((len(memory)-1, 1))))
            # print(avg_direction_movement)
            # print("________________")

            # Set frequency if we don't move at a certain speed

            # If movement is slow
            if np.average(movement_speeds) < THRESHOLD_SPEED:
                color = (0, 0, 255)

            # If movement is not in direction of target
            elif np.any(np.abs(avg_direction_target - avg_direction_movement) > THRESHOLD_DIRECTION):
                color = (0, 0, 255)

            else:
                color = (0, 255, 0)
                # print("Got'm bois")
            # print(tuple(np.array(np.array(state) + avg_direction_movement * 10, dtype=np.int)))
            # print(np.abs(avg_direction_target - avg_direction_movement))
            cv2.arrowedLine(img, state, tuple(np.array(np.array(state) + avg_direction_movement * 30, dtype=np.int)), color, 1)
            cv2.arrowedLine(img, state, tuple(np.array(np.array(state) + avg_direction_target * 30, dtype=np.int)), color, 1)
        else:
            color = (0, 255, 0)

        colors.append(color)

        # if datapoint["Frequency"] == last_freq:
        #     color = (0, 255, 0)
        #     colors.append(color)
        # else:
        #     last_freq = datapoint["Frequency"]
        #     color = (0, 0, 255)
        #     colors.append(color)

        # Tracking center, bounding box, action
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(img, p1, p2, color)
        cv2.circle(img, make_tuple(datapoint['Target']), 0, (178, 255, 102), 5)
        cv2.putText(img, f"{action_map[action]}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"Vpp: {round(datapoint['Vpp'], 2)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"Freq: {round(datapoint['Frequency'], 2)}kHz", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        centers.append(state)
        for i, center in enumerate(centers):
            cv2.circle(img, center, 0, (255, 120, 0), 2)

        # for pred in self.predictions:
        #     cv2.circle(img, (round(pred[0]), round(pred[1])), 0, (102, 255, 255), 2)
        # print(datapoint["State"])
        # cv2.imwrite(f'E:\\snapshots_21_10_21_processed_2\\{datapoint["Time"]}.png', img)

        cv2.imshow("Tracking", img)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

