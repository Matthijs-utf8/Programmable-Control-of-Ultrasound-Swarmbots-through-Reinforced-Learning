import cv2
import tqdm
import os
import time
import pandas
from ast import literal_eval as make_tuple
from collections import deque
import numpy as np
from settings import *
import shutil

if __name__ == "__main__":

    # Read metadata and specify folder for the images
    filename = 'experiments_square_channel_29_11_2021_L.csv'
    metadata = pandas.read_csv(f"{filename}")  # Make this file if you don't have it yet
    del metadata['Unnamed: 0']
    metadata = metadata[metadata['Action'] != -1]
    folder = "C:\\Users\\ARSL\\PycharmProjects\\Project_Matt\\experiments_square_channel_29_11_2021\\"
    imgs = os.listdir(folder)

    a = 0
    centers = []
    targets = []

    for i, datapoint in tqdm.tqdm(metadata.iterrows()):

        if f'{datapoint["Time"]}.png' not in imgs:
            print(f'{datapoint["Filename"]} not found...')
            continue

        # Reset
        if '-reset.png' in datapoint['Filename']:
            img = cv2.imread(f"{folder}{datapoint['Time']}-reset.png", cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.imread(f"{folder}{datapoint['Time']}.png", cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Parameters
        state = make_tuple(datapoint['State'])
        action = datapoint['Action']
        size = datapoint['Size']
        bbox = [int(state[0] - 0.5 * size), int(state[1] - 0.5 * size), size, size]

        # Draw green line on the side which the piezo was actuated
        if action == 0:
            cv2.line(img, (IMG_SIZE-2, IMG_SIZE), (298, 0), (0, 255, 0), 4)
        elif action == 2:
            cv2.line(img, (2, IMG_SIZE), (2, 0), (0, 255, 0), 4)
        elif action == 1:
            cv2.line(img, (0, IMG_SIZE-2), (IMG_SIZE, 298), (0, 255, 0), 4)
        elif action == 3:
            cv2.line(img, (0, 2), (IMG_SIZE, 2), (0, 255, 0), 4)

        centers.append(state)
        targets.append(make_tuple(datapoint["Target"]))

        # Tracking center, bounding box, action
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(img, p1, p2, (255, 255, 255))
        cv2.circle(img, make_tuple(datapoint['Target']), 0, (178, 255, 102), 5)
        cv2.putText(img, f"Vpp: {round(datapoint['Vpp'], 2)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"Freq: {round(datapoint['Frequency'], 2)}kHz", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        centers.append(state)
        for i, center in enumerate(centers):
            cv2.circle(img, center, 0, (255, 120, 0), 2)
        for i, target in enumerate(targets):
            cv2.circle(img, target, 0, (255, 0, 0), 2)
        cv2.imwrite(f"C:\\Users\\ARSL\\PycharmProjects\\Project_Matt\\{filename}_processed\\{datapoint['Time']}.png", img)

        # Display image
        cv2.imshow("Tracking", img)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

