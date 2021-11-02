import numpy as np
from environment_pipeline import SwarmEnv
from environment_pipeline import DataGatherEnv
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
    # env = SwarmEnv()
    # model = walk_to_pixel

    env = DataGatherEnv()

    # Loop through episodes
    for episode in range(EPISODES):

        # state = env.reset()  # Fill in bbox from last step to continue tracking same swarm
        # print(f"Initial state: {state}")
        # t0 = time.time()

        t0 = time.time()

        # Loop through steps
        # for _ in tqdm(range(MAX_STEPS)):
        #
            # action = model(state, target_pos=TARGET_POINTS[env.target_idx])
            # state = env.env_step(action)

        vpp_steps = 11
        freq_steps = 21
        action_steps = 4
        env_steps = 60
        for vpp in np.linspace(MIN_VPP, MAX_VPP, num=vpp_steps):
            for frequency in np.linspace(MIN_FREQUENCY, MAX_FREQUENCY, num=freq_steps):
                for action in range(action_steps-1):
                    for step in range(env_steps):
                        env.env_step(action=action,
                                     vpp=vpp,
                                     frequency=frequency)

        print(f'Time per step: {(time.time() - t0)/(vpp_steps * freq_steps * action_steps * env_steps)}')

    env.close()


if __name__ == "__main__":

    main()
