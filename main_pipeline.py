import numpy as np
from environment_pipeline import SwarmEnvTrackBiggestCluster
# from environment_pipeline import SwarmEnvDetectNClusters
from model import average_centroid_correction, random_action, walk_to_pixel, initial_actions, PID
import time
import preprocessing
from tqdm import tqdm
from settings import *

# Action: 0 --> Relay_channel: 1 --> Out: Green --> Piezo: Right --> Move: Left
# Action: 1 --> Relay_channel: 2 --> Out: Purple --> Piezo: Bottom --> Move: Up
# Action: 2 --> Relay_channel: 3 --> Out: Blue --> Piezo: Left --> Move: Right
# Action: 3 --> Relay_channel: 4 --> Out: White --> Piezo: Top --> Move: Down


def main():

    TARGET_COORD = TARGET_POINTS[0]
    action_map = {2: 0, 3: 1, 0: 2, 1: 3}

    # Initiate environment and action model
    env = SwarmEnvTrackBiggestCluster()
    # pid = PID()
    model = walk_to_pixel

    # Loop through episodes
    for episode in range(EPISODES):

        state = env.reset()
        print(f"Initial state: {state}")

        # Loop through steps
        for i in tqdm(range(MAX_STEPS)):

            print(env.tracker.bbox)

            action = action_map[model(state, target_pos=TARGET_COORD)]
            state = env.env_step(action, vpp=0.1, frequency=240e3)

            print(env.tracker.bbox)
            print('_________')

    env.close()

def setup():
    pass


if __name__ == "__main__":

    main()
