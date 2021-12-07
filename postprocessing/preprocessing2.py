import cv2
from operator import itemgetter
import numpy as np
from settings import *
import matplotlib.pyplot as plt

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

    # plt.hist(image)
    # plt.show()

    # Using cv2.blur() method
    cleared_image = cv2.blur(cv2.threshold(image, 110, 255, cv2.THRESH_BINARY)[1], (2, 2))  # TODO --> Automatic threshold settings

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
        bboxes.append([int(cX-squared_area),
                       int(cY-squared_area),
                       int(2*squared_area),
                       int(2*squared_area)])


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

        # cv2.imshow("Tracking", img)
        # cv2.circle(img, TARGET_COORD, 0, (255, 0, 0), 10)
        # Exit if ESC pressed
        # k = cv2.waitKey(1) & 0xff
        # if k == 27:
        #     return

    while len(areas) < 50:
        centroids.append((None, None))
        areas.append(None)
        bboxes.append([None, None, None, None])

    return centroids, areas, bboxes


class TrackClusters:

    def __init__(self, bbox=None):
        self.bbox = bbox
        # print(self.bbox)

    def reset(self, img):

        # Check if we specified a bounding box to start with, otherwise select largest cluster
        # if not self.bbox:
        #     _, _, self.bbox = find_clusters(image=img, amount_of_clusters=1, verbose=False)
        # if not self.bbox:
        #     self.bbox = (0, 0, IMG_SIZE, IMG_SIZE)
        if not self.bbox:
            return [None, None]

        # Define tracker and initialise
        self.tracker = cv2.TrackerCSRT_create()  # Very accurate, dynamic sizing, not the fastest, still okay
        # self.tracker = cv2.legacy_TrackerMedianFlow.create()  # Very fast, dynamic sizing, medium accuracy

        self.ok = self.tracker.init(img, self.bbox)

        # Calculate center of bounding box
        self.center = [int(self.bbox[0] + 0.5 * self.bbox[2]), int(self.bbox[1] + 0.5 * self.bbox[3])]

        return self.center, self.bbox

    def update(self, img, target: tuple, verbose: bool=False):

        # Perform tracker update and calculate new center
        try:
            self.ok, self.bbox = self.tracker.update(img)
        except:
            # print("Problem with tracking")
            return [None, None]
        self.bbox = list(self.bbox)
        self.center = [int(self.bbox[0] + 0.5 * self.bbox[2]), int(self.bbox[1] + 0.5 * self.bbox[3])]

        if verbose:

            # Draw bounding box
            if self.ok:

                # Tracking success
                p1 = (int(self.bbox[0]), int(self.bbox[1]))
                p2 = (int(self.bbox[0] + self.bbox[2]), int(self.bbox[1] + self.bbox[3]))
                cv2.rectangle(img, p1, p2, (255, 0, 0))
                cv2.circle(img, target, 0, (255, 0, 0), 5)

            # else:
            #
            #     # Tracking failure, reset
            #     cv2.putText(img, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            #     self.reset(img)



        return self.center, self.bbox