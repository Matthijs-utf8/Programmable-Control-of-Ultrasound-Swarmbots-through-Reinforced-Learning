import numpy as np
import pandas as pd
import tqdm
from ast import literal_eval as make_tuple
import os
import cv2
import matplotlib.pyplot as plt

# Initialize metadata
print('Loading data...')
PROJECT_NAME = 'Project_Matt'  # Project name
DATE = ''  # Date of the experiment
EXPERIMENT_RUN_NAME = 'TEST'  # Name of the experiment run
SAVE_DIR = f"C:\\Users\\ARSL\\PycharmProjects\\{PROJECT_NAME}\\{DATE}"  # Location for images all the images and metadata
METADATA = pd.read_csv(f"{SAVE_DIR}{EXPERIMENT_RUN_NAME}_processed.csv")
del csv['Unnamed: 0'] # Delete unwanted column to save memory

# Initialize dynamics csv
PRED_SECONDS = 1.5 # Length of movement timestep
DYNAMICS_CSV = f"{SAVE_DIR}\\{EXPERIMENT_RUN_NAME}_dynamics_{PRED_SECONDS}s.csv"

# Define the three input variables and make sets of the same input variables
df = processed_csv[["Frequency", "Action", "Vpp"]]
uniques = sorted(df.groupby(list(METADATA)).apply(lambda x: tuple(x.index)).tolist())
starting_points = {}
for i, unique in enumerate(uniques):
    starting_points.__setitem__(unique[0], len(unique))

if __name__ == "__main__":

    # Specify column names that you would like to save in metadata
    columns = ['Time', 'Vpp', 'Frequency', 'Action', 'Cluster', 'Size', 'X0', 'Y0', 'X1', 'Y1', 'Magnitude', 'dX', 'dY']

    # Initielize empty array for dynamics data
    dynamics_array = np.zeros((METADATA.shape[0] * METADATA.shape[1], len(columns)))
    j = 0

    for n, datapoint in tqdm.tqdm(metadata.iterrows()):

        # Loop through datapoints and record the change in position over PRED_SECONDS seconds
        if n in starting_points.keys():
            a = 0
            num_entries = starting_points[n]
            closest = np.argmin(np.array(np.abs(metadata[n:n + int(num_entries - 1 - a)]['Time'] - datapoint["Time"] - PRED_SECONDS)))
        else:
            if a + closest >= num_entries - 2:
                continue
            else:
                a += 1
                closest = np.argmin(np.array(np.abs(metadata[n:n + int(num_entries - 1 - a)]['Time'] - datapoint['Time'] - PRED_SECONDS)))

        # Datapoint closest to PRED_SECONDS seconds
        datapoint1 = metadata.iloc[n+closest]

        # Exception handling
        if datapoint["Action"] != datapoint1["Action"]:
            raise ValueError("Something went wrong!")

        # Initalize metadata
        initial_data = {"Time": datapoint['Time'],
                        "Vpp": datapoint['Vpp'],
                        "Frequency": datapoint['Frequency'],
                        "Action": datapoint['Action']}

        # Loop trhough clusters and add their dynamics to the dynamics csv
        for i in range(int(datapoint["num_clusters"])):

            data = initial_data.copy()

            pos0, size = make_tuple(datapoint[f"Cluster{i}"])
            pos1, _ = make_tuple(datapoint1[f"Cluster{i}"])

            data.__setitem__("Cluster", i)
            data.__setitem__("Size", size)
            data.__setitem__("X0", pos0[0])
            data.__setitem__("Y0", pos0[1])
            data.__setitem__("X1", pos1[0])
            data.__setitem__("Y1", pos1[1])

            # Exception handling
            if None in data.values():
                raise ValueError("Something went wrong!")

            offset = np.array(pos1) - np.array(pos0)
            magnitude = np.linalg.norm(offset)

            if not magnitude:
                data.__setitem__("Magnitude", 0)
                data.__setitem__("dX", 0)
                data.__setitem__("dY", 0)
            else:
                vector = tuple(offset / magnitude)
                data.__setitem__("Magnitude", magnitude)
                data.__setitem__("dX", vector[0])
                data.__setitem__("dY", vector[1])

            # Add to dynamics array
            dynamics_array[j] = list(data.values())

            j += 1

    # Save data from array to csv (really fast this way)
    dynamics_array = np.delete(dynamics_array, slice(j, len(dynamics_array), 1), axis=0)
    dynamics_csv = pd.DataFrame(dynamics_array, columns=['Time', 'Vpp', 'Frequency', 'Action', 'Cluster', 'Size', 'X0', 'Y0', 'X1', 'Y1', 'Magnitude', 'dX', 'dY'])
    dynamics_csv.to_csv(f"{training_data_folder}{filename_dynamics_csv}.csv")

