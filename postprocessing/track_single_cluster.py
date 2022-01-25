from postprocessing.cluster_detection_and_tracking import find_clusters, TrackClusters
import numpy as np
import pandas as pd
import os
import cv2
import tqdm
import os
from manipulation.settings import *


class TrackNClusters:

    def __init__(self):
        pass

    def reset(self, img, cutoff=None):
        """
        Initialize trackers for all clusters on screen
        :param img: Working image
        :return:    Cluster centroids and areas
        """

        # Find all clusters
        centroid, area, bbox = find_clusters(image=img, amount_of_clusters=1, verbose=False, cutoff=cutoff)

        # Initialize tracking algorithm for each cluster
        self.tracker = TrackClusters(bbox=bbox[0])
        self.tracker.reset(img=img)

        return centroid, area, bbox[0]

    def env_step(self, img):
        """
        Perfrom tracking step
        :param img: Working image
        :return:    Cluster centroids
        """

        center, bbox = self.tracker.update(img=img, target=None, verbose=True)

        return center, bbox

if __name__ == "__main__":

    # Initialize environment
    env = TrackNClusters()

    # Initialize variables
    vpp = 0
    freq = 0
    action = 0

    # Initialize metadata
    print('Loading data...')
    METADATA_CENTROID_EXTRACTED = pd.DataFrame()  # Empty dataframe for extracted data

    # Loop through datapoints
    for n, datapoint in tqdm.tqdm(METADATA.iterrows()):

        if "reset" in datapoint["Filename"]:
            filename = f"{datapoint['Time']}-reset.png"
            img = cv2.imread(f"{SNAPSHOTS_SAVE_DIR}{filename}", cv2.IMREAD_GRAYSCALE)
            cutoff = int(np.percentile(img, 3))
            center, area, bbox = env.reset(img=img, cutoff=cutoff)
            # METADATA_CENTROID_EXTRACTED.to_csv(f"{SAVE_DIR}\\{EXPERIMENT_RUN_NAME}_processed.csv")
        else:
            filename = f"{datapoint['Time']}.png"
            img = cv2.imread(f"{SNAPSHOTS_SAVE_DIR}{filename}", cv2.IMREAD_GRAYSCALE)
            cutoff = int(np.percentile(img, 3))
            center, bbox = env.env_step(img=img)

        data = {"Filename": f"{SNAPSHOTS_SAVE_DIR}{filename}"}
        data.__setitem__(f"Center", center)
        data.__setitem__(f"Area", area)
        metadata_centroids_extracted = METADATA_CENTROID_EXTRACTED.append(data, ignore_index=True)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(img, p1, p2, (255, 255, 0))

        # Display result
        cv2.imshow("Tracking", img)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break


