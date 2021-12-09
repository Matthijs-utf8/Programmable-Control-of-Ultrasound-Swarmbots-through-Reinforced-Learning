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
        """
        Initialize trackers for all clusters on screen
        :param img: Working image
        :return:    Cluster centroids and areas
        """

        # Find all clusters
        centroids, areas, bboxes = find_clusters(image=img, amount_of_clusters=None, verbose=True)

        # Initialize tracking algorithm for each cluster
        self.trackers = [TrackClusters(bbox=bboxes[i]) for i in range(len(areas))]
        for tracker in self.trackers:
            _, _ = tracker.reset(img=img)

        return centroids, areas

    def env_step(self, img):
        """
        Perfrom tracking step
        :param img: Working image
        :return:    Cluster centroids
        """

        # Track clusters in the next frame
        centers = []
        for tracker in self.trackers:
            center, bbox = tracker.update(img=img, target=None, verbose=True)
            if not center:
                centers.append((None, None))
            else:
                centers.append(tuple(center))

        return centers

if __name__ == "__main__":

    # Initialize environment
    env = TrackNClusters()

    # Initialize variables
    vpp = 0
    freq = 0
    action = 0

    # Initialize metadata
    print('Loading data...')
    PROJECT_NAME = 'Project_Matt'  # Project name
    DATE = ''  # Date of the experiment
    EXPERIMENT_RUN_NAME = 'TEST'  # Name of the experiment run
    SAVE_DIR = f"C:\\Users\\ARSL\\PycharmProjects\\{PROJECT_NAME}\\{DATE}"  # Location for images all the images and metadata
    METADATA = pd.read_csv(f"{SAVE_DIR}{EXPERIMENT_RUN_NAME}.csv")
    del csv['Unnamed: 0'] # Delete unwanted column to save memory
    csv = csv.dropna() # Drop NaN rows
    METADATA_CENTROIDS_EXTRACTED = pd.DataFrame() # Empty dataframe for extracted data

    # Loop through datapoints
    for n, datapoint in tqdm.tqdm(METADATA.iterrows()):

        # Load image and metadata from image
        img = cv2.imread(f"{SAVE_DIR}\\{EXPERIMENT_RUN_NAME}\\{datapoint['Time']}.png", cv2.IMREAD_GRAYSCALE)
        new_vpp = datapoint['Vpp']
        new_freq = datapoint['Frequency']
        new_action = datapoint['Action']

        # Initialize processed metadata
        data = {"Time": datapoint['Time'],
                "Vpp": new_vpp,
                "Frequency": new_freq,
                "Action": new_action}

        # Reset if we have a change in frequency, vpp or piezo
        if new_vpp != vpp or new_freq != freq or new_action != action:
            centers, areas = env.reset(img=img)
            metadata_centroids_extracted.to_csv(f"{SAVE_DIR}{EXPERIMENT_RUN_NAME}_processed.csv")
        else:
            centers = env.env_step(img=img)

        # Save data to dictionary and append to processed csv
        data.__setitem__(f"num_clusters", len(centers))
        for i in range(len(centers)):
            data.__setitem__(f"Cluster{i}", [centers[i], areas[i]])
        metadata_centroids_extracted = metadata_centroids_extracted.append(data, ignore_index=True)

        # Replace old variables
        vpp = new_vpp
        freq = new_freq
        action = new_action

        # Display result
        cv2.imshow("Tracking", img)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break



