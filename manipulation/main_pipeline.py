from environment_pipeline import SwarmEnv
from model import calc_action
from tqdm import tqdm
from settings import *

'''
Action: 0 --> Relay_channel: 1 --> Out: Green --> Piezo: Right --> Move: Left
Action: 1 --> Relay_channel: 2 --> Out: Purple --> Piezo: Bottom --> Move: Up
Action: 2 --> Relay_channel: 3 --> Out: Blue --> Piezo: Left --> Move: Right
Action: 3 --> Relay_channel: 4 --> Out: White --> Piezo: Top --> Move: Down
'''


def main():

    # Initiate environment and action model
    env = SwarmEnv()
    model = calc_action

    state = env.reset()  # Fill in bbox from last step to continue tracking same swarm
    print(f"Initial state: {state}")

    # Loop through steps
    for _ in tqdm(range(MAX_STEPS)):

        action = model(state, offset)
        state = env.env_step()

    env.close()


if __name__ == "__main__":

    main()
