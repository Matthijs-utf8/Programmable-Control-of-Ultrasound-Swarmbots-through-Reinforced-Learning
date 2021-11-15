from preprocessing2 import TrackClusters, find_clusters
import numpy as np
import pandas as pd
import os
import cv2
import tqdm
import multiprocessing

# SAVE_DIR = "data_standerdized_4"
NUM_CLUSTER = 50
METADATA = "metadata_for_new_model4.csv"
PROCESSED_METADATA = "metadata_for_new_model4_tracked2.csv"


class TrackNClusters:

    def __init__(self):

        pass

    def reset(self, img):

        # Initialize tracking algorithm
        centroids, areas, bboxes = find_clusters(image=img, amount_of_clusters=NUM_CLUSTER, verbose=True)

        self.trackers = [TrackClusters(bbox=bboxes[i]) for i in range(len(areas))]

        for tracker in self.trackers:
            center, bbox = tracker.reset(img=img)

        return centroids, areas

    def env_step(self, img):

        centers = []

        for tracker in self.trackers:

            center, bbox = tracker.update(img=img, target=None, verbose=True)
            if not center:
                centers.append((None, None))
            else:
                centers.append(tuple(center))

        return centers

env = TrackNClusters()
metadata = pd.read_csv(METADATA)  # Make this file if you don't have it yet
del metadata['Unnamed: 0']
metadata.fillna(-1, inplace=True)

# try:
processed_metadata = pd.read_csv(PROCESSED_METADATA)  # Make this file if you don't have it yet
del processed_metadata['Unnamed: 0']  # Remove unwanted column

vpp = 0
freq = 0
action = 0

if __name__ == "__main__":


    # except:

    # print(processed_metadata['Time'].tolist())

    # processed_metadata = pd.DataFrame(
    #     [{"Time": 0,
    #      "Vpp": vpp,
    #      "Frequency": freq,
    #      "Action": action}]
    # )
    metadata = metadata[metadata.values != -1]

    for n, datapoint in tqdm.tqdm(metadata.iterrows()):

        # img = cv2.imread(f"{SAVE_DIR}{datapoint['Time']}.png", cv2.IMREAD_GRAYSCALE)
        # try:
        if datapoint['Time'] in processed_metadata['Time'].tolist():
            print(f'Skip: {n}')
            continue

        print(datapoint['Filename'])
        img = cv2.imread(datapoint['Filename'], cv2.IMREAD_GRAYSCALE)
        new_vpp = datapoint['Vpp']
        new_freq = datapoint['Frequency']
        new_action = datapoint['Action']

        data = {"Time": datapoint['Time'],
                "Vpp": new_vpp,
                "Frequency": new_freq,
                "Action": new_action}

        if new_vpp != vpp or new_freq != freq or new_action != action:
            # processed_metadata.to_csv(PROCESSED_METADATA)
            # centers, areas = env.reset(img=img)
            centers = [0, 0, 0]
            areas = [0, 0, 0]
            print(f'Reset: {n}')
        else:
            # centers = env.env_step(img=img)
            centers=[0, 0, 0]
            print(print(f'Step: {n}'))

        # print(vpp, freq, action)

        for i in range(len(centers)):
            data.__setitem__(f"Cluster{i}", [centers[i], areas[i]])

        processed_metadata = processed_metadata.append(data, ignore_index=True)


        vpp = new_vpp
        freq = new_freq
        action = new_action

        # Display result
        cv2.imshow("Tracking", img)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        # except:
        #     continue



