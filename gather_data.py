from preprocessing2 import TrackClusters, find_clusters
import numpy as np
import pandas as pd
import os
import cv2
import tqdm

SAVE_DIR = "E:\\snapshots_21_10_21\\"
NUM_CLUSTER = 50


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
    metadata = pd.read_csv("E:\\metadata.csv")  # Make this file if you don't have it yet
    del metadata['Unnamed: 0']
    new_metadata = pd.read_csv("Metadata.csv")  # Make this file if you don't have it yet
    # del new_metadata['Unnamed: 2']  # Remove unwanted column
    del new_metadata['Unnamed: 0']  # Remove unwanted column

    metadata.fillna(-1, inplace=True)

    for n, datapoint in tqdm.tqdm(metadata.iterrows()):

        data = {"Time": datapoint['Time'],
                "Vpp": datapoint['Vpp'],
                "Frequency": datapoint['Frequency'],
                "Action": datapoint['Action']}

        # if datapoint["Time"] in new_metadata["Time"]:
        #     continue


        if '-reset.png' in datapoint['Filename']:
            # img = cv2.imread(f"{SAVE_DIR}{datapoint['Time']}-reset.png", cv2.IMREAD_GRAYSCALE)
            # if not np.any(img):
            #     print("No image found1!!!")
            #     continue
            # centers, areas = env.reset(img=img)
            # print(n)
            if n % 60 == 0:
                img = cv2.imread(f"{SAVE_DIR}{datapoint['Time']}-reset.png", cv2.IMREAD_GRAYSCALE)
                if not np.any(img):
                    print("No image found2!!!")
                    continue

                new_metadata.to_csv("Metadata.csv")
                centers, areas = env.reset(img=img)

            else:
                continue

        else:
            img = cv2.imread(f"{SAVE_DIR}{datapoint['Time']}.png", cv2.IMREAD_GRAYSCALE)
            if not np.any(img):
                print("No image found2!!!")
                continue

            if n % 60 == 0:
                new_metadata.to_csv("Metadata.csv")
                centers, areas = env.reset(img=img)
            else:
                centers = env.env_step(img=img)





        for i in range(len(centers)):
            data.__setitem__(f"Cluster{i}", [centers[i], areas[i]])

        new_metadata = new_metadata.append(data, ignore_index=True)
        # print(new_metadata.tail())


        # Display result
        cv2.imshow("Tracking", img)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

