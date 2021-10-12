import numpy as np
from settings import *
import matplotlib.pyplot as plt

class PID:

    def __init__(self, P=2.0, I=0.0, D=1.0, Derivator=0.0, Integrator=0.0, Integrator_max=500, Integrator_min=-500):

        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.Derivator = Derivator
        self.Integrator = Integrator
        self.Integrator_max = Integrator_max
        self.Integrator_min = Integrator_min

        self.set_point = 0.0
        self.error = 0.0

    def update(self, current_value):
        """
        Calculate PID output value for given reference input and feedback
        """
         # Calculate offset
        self.error = self.set_point - current_value

        # Set PD values
        self.P_value = self.Kp * self.error
        self.D_value = self.Kd * (self.error - self.Derivator)
        self.Derivator = self.error

        # Set I value
        self.Integrator = self.Integrator + self.error
        if self.Integrator > self.Integrator_max:
            self.Integrator = self.Integrator_max
        elif self.Integrator < self.Integrator_min:
            self.Integrator = self.Integrator_min
        self.I_value = self.Integrator * self.Ki

        # Add PID value
        PID = self.P_value + self.I_value + self.D_value

        return PID

    def setPoint(self, set_point):
        """
        Initilize the setpoint of PID
        """
        self.set_point = set_point
        self.Integrator = 0
        self.Derivator = 0

    def setIntegrator(self, Integrator):
        self.Integrator = Integrator

    def setDerivator(self, Derivator):
        self.Derivator = Derivator

    def setKp(self,P):
        self.Kp=P

    def setKi(self,I):
        self.Ki=I

    def setKd(self,D):
        self.Kd=D

    def getPoint(self):
        return self.set_point

    def getError(self):
        return self.error

    def getIntegrator(self):
        return self.Integrator

    def getDerivator(self):
        return self.Derivator

def random_action(nr_actions=4):
    return np.random.randint(low=0, high=nr_actions)

def initial_actions(action_length=40):
    action_list = np.array([[0, 2]]).repeat(int(action_length/2), 0)
    action_list = np.append(action_list, np.array([[1, 3]]).repeat(int(action_length/2), 0)).tolist()
    return action_list

def walk_to_pixel(blob_pos, target_pos):

    if not blob_pos:
        return random_action()

    offsets = - np.subtract(target_pos[0], blob_pos[0]), - np.subtract(target_pos[1], blob_pos[1])

    # Get the largest offset and compute action
    action = np.argmax(np.abs(offsets))
    if np.sign(offsets[action]) == -1:
        action += 2

    return action


def average_centroid_correction(centroids, areas, target_pos=None):

    """
    Correct location of swarm centroids based on their average location and outliers
    :param centroids: Location of n number of centroids
    :return: action --> range(0, 4)
    Setup: Incoming images are not mirrored!
    Action: 0 --> Relay_channel: 1 --> Out: Green --> Piezo: Bottom --> Move: Up
    Action: 1 --> Relay_channel: 2 --> Out: Purple --> Piezo: Right --> Move: Left
    Action: 2 --> Relay_channel: 3 --> Out: Blue --> Piezo: Top --> Move: Down
    Action: 3 --> Relay_channel: 4 --> Out: White --> Piezo: Left --> Move: Right
    """

    if np.any(centroids):

        # If we want to move to a specific coordinate
        if target_pos:
            # print(target_pos)
            medians = target_pos

        # Else, cluster towards the median coordinate
        else:
            # Calculate median to find the reference coordinates
            medians = np.min(centroids, axis=0) + (np.max(centroids, axis=0) - np.min(centroids, axis=0)) / 2

        # print(medians)
        # Calculate the weighted average of the centroids
        averages = np.average(centroids, axis=0, weights=areas)  # Make this a weighted average
        # print(averages)
        # Calculate offsets from the reference
        offsets = - np.subtract(medians[0], averages[0]), - np.subtract(medians[1], averages[1])
        # print(offsets)
        # Get the largest offset and compute action
        action = np.argmax(np.abs(offsets))
        if np.sign(offsets[action]) == -1:
            action += 2

    # If centroids are all located at [0, 0], no centroids were found
    else:
        print("No centroids found. Chose random action")
        action = random_action()

    return action

if __name__ == "__main__":

    np.random.seed(0)
    nr_of_blobs = 5
    img_size = 15
    target_pos = [0, 8]
    centroids = np.random.randint(0, img_size, (nr_of_blobs, 2))
    # print(centroids)
    # areas = np.random.randint(10, 50, (nr_of_blobs,))
    areas = np.ones(nr_of_blobs,)
    blank = np.zeros((img_size, img_size, 3))

    for blob_pos in centroids:
        # print(blob_pos)
        blank[blob_pos[0], blob_pos[1]] = 255, 255, 255

    plt.imshow(blank)
    plt.show()



    test = average_centroid_correction(centroids=centroids,
                                       areas=areas,
                                       target_pos=target_pos)

    print(test)
