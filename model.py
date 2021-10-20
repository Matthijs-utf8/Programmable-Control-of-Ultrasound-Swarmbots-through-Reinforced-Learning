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

# TODO --> Test if absolute or relative state works best, or if euclidean distance is better
def predict_state(Vpp: float, frequency: int, size: int, model) -> tuple:
    """
    Calculate and return predicted state some time in the future, depending on the model.
    :param Vpp: Voltage of function generator
    :param frequency: Frequency of function generator
    :param size: Size of swarm (width or height of the bounding box)
    :return: Predicted magnitude of motion between 0 and 1 (1 representing the maximum of 300 pixels of movement)
    """
    return model.predict(np.array([[Vpp/MAX_VPP, frequency/MAX_FREQUENCY, size/SIZE_NORMALIZING_FACTOR]][0]))

def get_action(size: int, offset_to_target: tuple):
    """
    Calculate best vpp and frequency for a given size of swarm and distance to target
    Calculate best piezo to actuate
    :param size: Size of swarm
    :param offset_to_target: (x, y) offset to target
    :return: Vpp, frequency and choice of piezo
    """

    # Generate potential combinations of inputs to the model
    steps = [[Vpp, frequency, size] for Vpp in np.linspace(MIN_VPP, MAX_VPP, int((MAX_VPP - MIN_VPP) / VPP_STEP_SIZE + 1)) for frequency in np.linspace(MIN_FREQUENCY, MAX_FREQUENCY, int((MAX_FREQUENCY - MIN_FREQUENCY) / FREQUENCY_STEP_SIZE + 1))]

    # Map all combinations to model to get the predicted motion
    results = np.array(list(map(predict_state, steps)))  # TODO --> Optimize

    # Find the Vpp and frequency that belong to the best inputs
    # 'Best' is defined as the input that brings the robot closest to the target in either the x or y direction
    Vpp, frequency, size = steps[np.abs(results - np.max(np.abs(offset_to_target))).argmin()]

    # Choose one of four piezos
    # IMPORTANT --> 0: dx > 0 (move left), 1: dy > 0 (move up), 2: dx < 0 (move right), 3: dy < 0 (move down)
    action = np.argmax(np.abs(offset_to_target))
    if np.sign(offset_to_target[action]) == -1:
        action += 2

    return action, Vpp, frequency


def walk_to_pixel(blob_pos, target_pos):

    if not blob_pos:
        return random_action()

    offsets = - np.subtract(target_pos[0], blob_pos[0]), - np.subtract(target_pos[1], blob_pos[1])

    # Get the largest offset and compute action
    action = np.argmax(np.abs(offsets))
    if np.sign(offsets[action]) == -1:
        action += 2

    return action

if __name__ == "__main__":

    get_action(100, (-50, -20))
