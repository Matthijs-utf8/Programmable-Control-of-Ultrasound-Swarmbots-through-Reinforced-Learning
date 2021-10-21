import numpy as np
from environment_pipeline import SwarmEnvTrackBiggestCluster
# from environment_pipeline import SwarmEnvDetectNClusters
from model import random_action, walk_to_pixel
import time
import preprocessing
from tqdm import tqdm
from settings import *


# Action: 0 --> Relay_channel: 1 --> Out: Green --> Piezo: Right --> Move: Left
# Action: 1 --> Relay_channel: 2 --> Out: Purple --> Piezo: Bottom --> Move: Up
# Action: 2 --> Relay_channel: 3 --> Out: Blue --> Piezo: Left --> Move: Right
# Action: 3 --> Relay_channel: 4 --> Out: White --> Piezo: Top --> Move: Down


def main():

    # Initiate environment and action model
    env = SwarmEnvTrackBiggestCluster()
    model = walk_to_pixel

    # Loop through episodes
    for episode in range(EPISODES):

        state = env.reset(bbox=None)  # Fill in bbox from last step to continue tracking same swarm
        print(f"Initial state: {state}")
        t0 = time.time()

        # Loop through steps
        for _ in tqdm(range(MAX_STEPS)):

            action = model(state, target_pos=TARGET_POINTS[env.target_idx])
            state = env.env_step(action)

        print(f'Time per step: {(time.time() - t0)/MAX_STEPS}')

    env.close()


if __name__ == "__main__":

    main()
