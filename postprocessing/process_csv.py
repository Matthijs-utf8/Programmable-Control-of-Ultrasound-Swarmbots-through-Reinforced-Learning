import numpy as np
import pandas as pd
import tqdm
from ast import literal_eval as make_tuple
import os
import cv2
import matplotlib.pyplot as plt

# Load data from tracked csv
print('Loading data...')
# print(os.listdir('E:\\square_env_recordings_metadata_tracked')[-1])
processed_csv = pd.read_csv(f'E:\\square_env_recordings_metadata_tracked\\2500.0.csv')

# Define the three input variables and make sets of the same input variables
df = processed_csv[["Frequency", "Action", "Vpp"]]
uniques = sorted(df.groupby(list(df)).apply(lambda x: tuple(x.index)).tolist())
starting_points = {}
for i, unique in enumerate(uniques):
    starting_points.__setitem__(unique[0], len(unique))

# Create new data folder and filename the dynamics csv
PRED_SECONDS = 1.5
training_data_folder = f"E:\\square_env_recordings_metadata_training\\"
filename_dynamics_csv = f"dynamics_{PRED_SECONDS}s"
if not os.path.exists(training_data_folder):
    os.mkdir(training_data_folder)

if __name__ == "__main__":

    # Get metadata csv
    metadata = processed_csv

    dynamics_dict = {}
    columns = ['Time', 'Vpp', 'Frequency', 'Action', 'Cluster', 'Size', 'X0', 'Y0', 'X1', 'Y1', 'Magnitude', 'dX', 'dY']
    dynamics_array = np.zeros((10_000_000, len(columns)))
    j = 0

    for n, datapoint in tqdm.tqdm(metadata.iterrows()):

        # Check if reset
        if n in starting_points.keys():
            a = 0
            num_entries = starting_points[n]
            closest = np.argmin(np.array(np.abs(metadata[n:n + int(num_entries - 1 - a)]['Time'] - datapoint["Time"] - PRED_SECONDS)))
            # print("Reset", n, a, closest)
        else:
            # print(a + closest)
            if a + closest >= num_entries - 2:
                #
                # print("Skip:", n)
                continue
            else:
                a += 1
                # print(a)
                closest = np.argmin(np.array(np.abs(metadata[n:n + int(num_entries - 1 - a)]['Time'] - datapoint['Time'] - PRED_SECONDS)))
                # print("Step", n, a, closest)

        datapoint1 = metadata.iloc[n+closest]

        if datapoint["Action"] != datapoint1["Action"]:
            raise ValueError("Something went wrong!")

        initial_data = {"Time": datapoint['Time'],
                        "Vpp": datapoint['Vpp'],
                        "Frequency": datapoint['Frequency'],
                        "Action": datapoint['Action']}

        for i in range(int(datapoint["num_clusters"])):

            data = initial_data.copy()

            pos0, size = make_tuple(datapoint[f"Cluster{i}"])
            pos1, _ = make_tuple(datapoint1[f"Cluster{i}"])

            # data.__setitem__("Index", j)
            data.__setitem__("Cluster", i)
            data.__setitem__("Size", size)
            data.__setitem__("X0", pos0[0])
            data.__setitem__("Y0", pos0[1])
            data.__setitem__("X1", pos1[0])
            data.__setitem__("Y1", pos1[1])

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

            dynamics_array[j] = list(data.values())

            j += 1

    # Save data
    dynamics_array = np.delete(dynamics_array, slice(j, len(dynamics_array), 1), axis=0)
    dynamics_csv = pd.DataFrame(dynamics_array, columns=['Time', 'Vpp', 'Frequency', 'Action', 'Cluster', 'Size', 'X0', 'Y0', 'X1', 'Y1', 'Magnitude', 'dX', 'dY'])
    dynamics_csv.to_csv(f"{training_data_folder}{filename_dynamics_csv}.csv")

