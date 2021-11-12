import numpy as np
import pandas as pd
import tqdm
from ast import literal_eval as make_tuple

# METADATA = "C:\\Users\\Matthijs\\PycharmProjects\\AI_Actuated_Microswarm_4\\Include\\AI_Actuated_Micrswarm_4\\develop_model\\Metadata.csv"
METADATA = "metadata_for_new_model4_tracked2.csv"
PROCESSED_CSV = "training_data\\metadata_for_new_model4_tracked2_processed"
NUM_CLUSTER = 50

if __name__ == "__main__":

    metadata = pd.read_csv(METADATA)
    del metadata['Unnamed: 0']

    vpp = -1
    freq = -1
    action = -1
    a = -1

    csv_index = 0

    try:
        processed_csv = pd.read_csv(f"{PROCESSED_CSV}_{csv_index}.csv")  # Make this file if you don't have it yet
        del processed_csv['Unnamed: 0']  # Remove unwanted column
    except:
        processed_csv = pd.DataFrame(
            {'Index': 0,
             "Time": -1,
             "Vpp": -1,
             "Frequency": -1,
             "Action": -1}, index=[0]
        )


    for n, datapoint in tqdm.tqdm(metadata.iterrows()):


        if int(n / 1000) > csv_index:
            processed_csv.to_csv(f"{PROCESSED_CSV}_{csv_index}.csv")
            csv_index += 1
            try:
                processed_csv = pd.read_csv(f"{PROCESSED_CSV}_{csv_index}.csv")  # Make this file if you don't have it yet
                del processed_csv['Unnamed: 0']  # Remove unwanted column
            except:
                processed_csv = pd.DataFrame(
                    {'Index': 0,
                     "Time": -1,
                     "Vpp": -1,
                     "Frequency": -1,
                     "Action": -1}, index=[0]
                )

        # Check if already in processed
        if datapoint['Time'] in processed_csv['Time'].tolist():
            continue

        # Define meta variables
        new_vpp = datapoint['Vpp']
        new_freq = datapoint['Frequency']
        new_action = datapoint['Action']
        time = datapoint['Time']

        # Check if reset
        if new_vpp != vpp or new_freq != freq or new_action != action:
            a = 0
            closest = np.argmin(np.array(np.abs(metadata[n:n + int(99 - a)]['Time'] - time - 1)))
        else:
            a += 1
            if a >= 50:
                continue
            closest = np.argmin(np.array(np.abs(metadata[n:n + int(99 - a)]['Time'] - datapoint['Time'] - 1)))

        datapoint1 = metadata.iloc[n+closest]

        for i in range(NUM_CLUSTER):

            data = {"Time": time,
                    "Vpp": new_vpp,
                    "Frequency": new_freq,
                    "Action": new_action}

            pos, size = make_tuple(datapoint[f"Cluster{i}"])
            pos1, _ = make_tuple(datapoint1[f"Cluster{i}"])

            data.__setitem__("Cluster", i)
            data.__setitem__("Size", size)
            data.__setitem__("X0", pos[0])
            data.__setitem__("Y0", pos[1])
            data.__setitem__("X1", pos1[0])
            data.__setitem__("Y1", pos1[1])

            if None in data.values():
                continue

            offset = np.array(pos1) - np.array(pos)
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

            processed_csv = processed_csv.append(data, ignore_index=True)

        if not n % 100:
            processed_csv.to_csv(f"{PROCESSED_CSV}_{csv_index}.csv")