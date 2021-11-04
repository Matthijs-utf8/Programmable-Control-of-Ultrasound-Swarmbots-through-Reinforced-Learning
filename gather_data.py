from preprocessing2 import TrackClusters, find_clusters
import numpy as np
import pandas as pd
import os
import cv2
import tqdm

SAVE_DIR = "E:\\snapshots_21_10_21\\"
NUM_CLUSTER = 50
METADATA = ""
PROCESSED_METADATA = ""


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

if __name__ == "__main__":

    env = TrackNClusters()
    metadata = pd.read_csv(METADATA)  # Make this file if you don't have it yet
    del metadata['Unnamed: 0']
    metadata.fillna(-1, inplace=True)
    processed_metadata = pd.read_csv(PROCESSED_METADATA)  # Make this file if you don't have it yet
    del processed_metadata['Unnamed: 0']  # Remove unwanted column

    try:
        processed_metadata = pd.read_csv(PROCESSED_METADATA)  # Make this file if you don't have it yet
        del METADATA['Unnamed: 0']  # Remove unwanted column
    except:
        processed_metadata = pd.DataFrame(
            {"Time": -1}
        )

    for n, datapoint in tqdm.tqdm(metadata.iterrows()):

        data = {"Time": datapoint['Time'],
                "Vpp": datapoint['Vpp'],
                "Frequency": datapoint['Frequency'],
                "Action": datapoint['Action']}

        if '-reset.png' in datapoint['Filename']:
            img = cv2.imread(f"{SAVE_DIR}{datapoint['Time']}-reset.png", cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(f"{SAVE_DIR}{datapoint['Time']}.png", cv2.IMREAD_GRAYSCALE)

        if not n % 60:
            processed_metadata.to_csv(PROCESSED_METADATA)
            centers, areas = env.reset(img=img)
        else:
            centers = env.env_step(img=img)

        for i in range(len(centers)):
            data.__setitem__(f"Cluster{i}", [centers[i], areas[i]])

        processed_metadata = processed_metadata.append(data, ignore_index=True)

        # Display result
        cv2.imshow("Tracking", img)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

