import cv2
from operator import itemgetter
import numpy as np
from settings import *
import tqdm
import os
import model
import tensorflow as tf

nnet_path = "C:\\Users\\Matthijs\\PycharmProjects\\AI_Actuated_Microswarm_4\\model4"
nnet = tf.keras.models.load_model(nnet_path)

action_map = {0: "Move left", 1: "Move up", 2: "Move right", 3: "Move down", None: None}
RUN_DIR = "good_run_1"

# Find the indices of the top n values from a list or array quickly
def find_top_n_indices(data, top):
    indexed = enumerate(data)  # create pairs [(0, v1), (1, v2)...]
    sorted_data = sorted(indexed,
                         key=itemgetter(1),   # sort pairs by value
                         reverse=True)       # in reversed order
    return [d[0] for d in sorted_data[:top]]  # take first N indices


# Find n largest clusters using thresholding, canny edge detection and contour finding from OpenCV
def find_clusters(image, amount_of_clusters, verbose=False):

    if not image.any():
        return None

    # Check if image is grayscale
    assert len(image.shape) == 2, "Image must be grayscale"

    # hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    # plt.plot(hist)
    # plt.show()

    # Using cv2.blur() method
    cleared_image = cv2.blur(cv2.threshold(image, 130, 255, cv2.THRESH_BINARY)[1], (2, 2)) # TODO --> Automatic threshold settings

    # Separate clusters from background and convert background to black
    canny = cv2.Canny(cleared_image, threshold1=0, threshold2=0)

    # Find contours
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return [], [], []

    # Locate n biggest contours
    biggest_contours = find_top_n_indices([cv2.contourArea(con) for con in contours],
                                          top=amount_of_clusters)

    # Locate the centroid of each contour
    centroids = []
    areas = []
    bboxes = []

    # Find the features of contours
    for n in biggest_contours:

        # Calculate centroid moment and area
        M = cv2.moments(contours[n])
        cX = int(M["m10"] / (M["m00"] + 1e-8))
        cY = int(M["m01"] / (M["m00"] + 1e-8))
        area = cv2.contourArea(contours[n])

        if area <= 1:
            continue

        # Add features to respective lists
        centroids.append((cX, cY))
        areas.append(area)
        squared_area = np.sqrt(area)
        # bboxes.append(cv2.boundingRect(contours[n]))
        bboxes.append((int(cX-squared_area),
                       int(cY-squared_area),
                       int(2*squared_area),
                       int(2*squared_area)))


    if verbose:

        img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        for n in range(len(centroids)):
            cv2.drawContours(img, contours, n, (0, 0, 255), 1)
            cv2.circle(img, centroids[n], 0, (255, 0, 0), 5)
            cv2.rectangle(img,
                          pt1=(int(bboxes[n][0]), int(bboxes[n][1])),
                          pt2=(int(bboxes[n][0] + bboxes[n][2]), int(bboxes[n][1] + bboxes[n][3])),
                          color=(255, 255, 0))

        # Display result
        cv2.imshow("Tracking", img)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            return

    return centroids, areas, bboxes


# TODO --> Tracking function for multiple clusters
#  (go to cv2 version 4.4.X, https://github.com/opencv/opencv-python/issues/441)
class TrackClusters:

    def __init__(self):
        pass

    def reset(self, img):

        # Find largest cluster and define bounding box (see find_clusters for formula)
        _, _, self.bbox = find_clusters(image=img, amount_of_clusters=1, verbose=False)

        if not self.bbox:
            self.bbox = (0, 0, IMG_SIZE, IMG_SIZE)
        else:
            self.bbox = self.bbox[0]


        # Define tracker and initialise
        self.tracker = cv2.TrackerCSRT_create()  # Very accurate, dynamic sizing, not the fastest, still okay
        # self.tracker = cv2.legacy_TrackerMedianFlow.create()  # Very fast, dynamic sizing, medium accuracy

        self.ok = self.tracker.init(img, self.bbox)

        # Calculate center of bounding box
        self.center = (int(self.bbox[0] + 0.5 * self.bbox[2]), int(self.bbox[1] + 0.5 * self.bbox[3]))
        self.centers = [self.center]
        self.predictions = [self.center]

        return self.center


    def update(self, img, verbose=False, action=None):

        # Perform tracker update and calculate new center
        self.ok, self.bbox = self.tracker.update(img)
        self.predictions.append(tuple(np.array(nnet.predict(np.array(np.array((action/3, self.centers[-1][0]/300, self.centers[-1][1]/300))[np.newaxis, :])) * 300)[0].tolist()))
        # print(self.predictions)
        self.center = (int(self.bbox[0] + 0.5 * self.bbox[2]), int(self.bbox[1] + 0.5 * self.bbox[3]))
        self.centers.append(self.center)


        image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if verbose:

            # Draw bounding box
            if self.ok:

                # Tracking success
                p1 = (int(self.bbox[0]), int(self.bbox[1]))
                p2 = (int(self.bbox[0] + self.bbox[2]), int(self.bbox[1] + self.bbox[3]))
                cv2.rectangle(image, p1, p2, (255, 128, 0))
                cv2.circle(image, (150, 150), 0, (178, 255, 102), 5)
                cv2.putText(image, f"{action_map[action]}", (80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                for center in self.centers:
                    cv2.circle(image, center, 0, (255, 255, 102), 2)
                for pred in self.predictions:
                    cv2.circle(image, (round(pred[0]), round(pred[1])), 0, (102, 255, 255), 2)

            else:

                # Tracking failure, reset
                cv2.putText(img, "Tracking failure detected", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
                self.reset(img)

            # Display result
            # cv2.imwrite(f"C:\\Users\\Matthijs\\PycharmProjects\\AI_actuated_microswarm_2\\Include\\Visuals\\{RUN_DIR}\\{len(self.centers)}.png", image)
            cv2.imshow("Tracking", image)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                return

        return self.center


# Testing pipeline for the cluster finding algorithm
if __name__ == "__main__":

    import os
    import time
    t0 = time.time()

    # pid = model.PID()
    # pid.setPoint(set_point=TARGET_COORD[0])

    t0 = time.time()
    cluster_tracker = TrackClusters()
    bounds = slice(0, 300), slice(0, 300)

    folder = f"C:\\Users\\Matthijs\\PycharmProjects\\AI_Actuated_Microswarm_2\\Include\\snapshots\\{RUN_DIR}\\"

    for file in tqdm.tqdm(os.listdir(folder)):

        if "0.png" in file:
            action = 0
        elif "1.png" in file:
            action = 1
        elif "2.png" in file:
            action = 2
        elif "3.png" in file:
            action = 3

        if "-reset" in file:
            img = cv2.imread(f"{folder}/{file}", cv2.IMREAD_GRAYSCALE)[bounds]
            # find_clusters(img, amount_of_clusters=20, verbose=True)
            state = cluster_tracker.reset(img)
        else:
            img = cv2.imread(f"{folder}/{file}", cv2.IMREAD_GRAYSCALE)[bounds]
            # # find_clusters(img, amount_of_clusters=20, verbose=True)
            state = cluster_tracker.update(img, verbose=True, action=action)
            # print(np.array(TARGET_COORD) - np.array(state))
            # print(pid.update(np.max(state)))
            # print("___________")



    print(time.time() - t0)


