import numpy as np
from environment_pipeline import DataGatherEnv
import time
from tqdm import tqdm
from settings import *

'''
Action: 0 --> Relay_channel: 1 --> Out: Green --> Piezo: Right --> Move: Left
Action: 1 --> Relay_channel: 2 --> Out: Purple --> Piezo: Bottom --> Move: Up
Action: 2 --> Relay_channel: 3 --> Out: Blue --> Piezo: Left --> Move: Right
Action: 3 --> Relay_channel: 4 --> Out: White --> Piezo: Top --> Move: Down
'''

def main():

    # Import environment form pipeline
    env = DataGatherEnv()

    vpp_steps = 6
    freq_steps = 19
    action_steps = 4
    env_steps = 100
    total_steps = vpp_steps * freq_steps * action_steps * env_steps
    folder = "square_env_recordings"
    print(f"Total steps: {total_steps}")

    for frequency in tqdm(np.linspace(1500, 2500, num=freq_steps)):

        env.function_generator.set_frequency(frequency=frequency)

        # TODO: --> debug code and make sure all hyperparams are in setting.py
        metadata_filename = f"{folder}\\{frequency}.csv"
        env.metadata = pandas.DataFrame(
            {"Filename": "Initial",
             "Time": (None, None),
             "Vpp": None,
             "Frequency": None,
             "Action": None
             }
        )

        for vpp in np.linspace(10, 20, num=vpp_steps):

            t0 = time.time()

            env.function_generator.set_vpp(vpp=vpp)

            for action in range(action_steps):

                env.actuator.move(action=action)

                for step in range(env_steps):

                    env.env_step(action=action,
                                 vpp=vpp,
                                 frequency=frequency)

            print(f'FPS: {1 / ((time.time() - t0) / (action_steps * env_steps))}')

            env.metadata.to_csv(metadata_filename)

    env.close()


if __name__ == "__main__":

    main()
