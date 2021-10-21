import cv2
import matplotlib.pyplot as plt
import tqdm
import numpy
import os
import time
import pandas
from ast import literal_eval as make_tuple

if __name__ == "__main__":

    metadata = pandas.read_csv("metadata.csv")  # Make this file if you don't have it yet
    del metadata['Unnamed: 0']

    metadata.fillna(-1, inplace=True)

    folder = f"C:\\Users\\ARSL\\PycharmProjects\\Project_Matt\\snapshots\\"

    action_map = {0: "Move left", 1: "Move up", 2: "Move right", 3: "Move down", None: None, -1: 'Reset'}



    for i, datapoint in tqdm.tqdm(metadata.iterrows()):
        # print(i)

        # Read image
        img = cv2.imread(datapoint['Filename'], cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Reset
        if '-reset.png' in datapoint['Filename']:
            centers = []

        # Define state
        state = make_tuple(datapoint['State'])
        size = datapoint['Size']
        action = datapoint['Action']
        bbox = [int(state[0]-0.5*size), int(state[1]-0.5*size), size, size]

        # Tracking center, bounding box, action
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(img, p1, p2, (255, 128, 0))
        cv2.circle(img, make_tuple(datapoint['Target']), 0, (178, 255, 102), 5)
        cv2.putText(img, f"{action_map[action]}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"Vpp: {round(datapoint['Vpp'], 2)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"Freq: {round(datapoint['Frequency'], 2)}kHz", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        centers.append(state)
        for center in centers:
            cv2.circle(img, center, 0, (255, 255, 102), 2)
        # for pred in self.predictions:
        #     cv2.circle(img, (round(pred[0]), round(pred[1])), 0, (102, 255, 255), 2)

        cv2.imshow("Tracking", img)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break



        # break
