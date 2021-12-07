"""
Save format for file: time.time()
Metadata is saved in different csv files in one folder
"""

from preprocessing2 import TrackClusters, find_clusters
import numpy as np
import pandas as pd
import os
import cv2
import tqdm
import os


class TrackNClusters:

    def __init__(self):

        pass

    def reset(self, img):

        # Initialize tracking algorithm
        centroids, areas, bboxes = find_clusters(image=img, amount_of_clusters=None, verbose=True)

        self.trackers = [TrackClusters(bbox=bboxes[i]) for i in range(len(areas))]

        for tracker in self.trackers:
            _, _ = tracker.reset(img=img)

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

    vpp = 0
    freq = 0
    action = 0

    metadata_centroids_extracted = pd.DataFrame()

    metadata_tracked_folder = "E:\\square_env_recordings_metadata_tracked\\"
    if not os.path.exists(metadata_tracked_folder):
        os.mkdir(metadata_tracked_folder)

    print('Loading data...')
    for file in os.listdir('E:\\square_env_recordings_metadata\\'):

        csv = pd.read_csv(f'E:\\square_env_recordings_metadata\\{file}')
        del csv['Unnamed: 0']
        csv = csv.dropna()

        metadata = csv

        for n, datapoint in tqdm.tqdm(metadata.iterrows()):

            filename = f"E:\\square_env_recordings\\{datapoint['Time']}.png"

            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            new_vpp = datapoint['Vpp']
            new_freq = datapoint['Frequency']
            new_action = datapoint['Action']

            data = {"Time": datapoint['Time'],
                    "Vpp": new_vpp,
                    "Frequency": new_freq,
                    "Action": new_action}

            if new_vpp != vpp or new_freq != freq or new_action != action:
                centers, areas = env.reset(img=img)
                metadata_centroids_extracted.to_csv(f"{metadata_tracked_folder}{file}")
            else:
                centers = env.env_step(img=img)

            data.__setitem__(f"num_clusters", len(centers))
            for i in range(len(centers)):
                data.__setitem__(f"Cluster{i}", [centers[i], areas[i]])

            metadata_centroids_extracted = metadata_centroids_extracted.append(data, ignore_index=True)

            vpp = new_vpp
            freq = new_freq
            action = new_action

            # Display result
            cv2.imshow("Tracking", img)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break



