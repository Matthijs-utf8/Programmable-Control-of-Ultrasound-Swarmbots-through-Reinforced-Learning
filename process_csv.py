import numpy as np
import pandas as pd
import tqdm
from ast import literal_eval as make_tuple

METADATA = "metadata_for_new_model4_tracked.csv"
PROCESSED_CSV = "metadata_for_new_model4_tracked_processed.csv"
NUM_CLUSTER = 50

if __name__ == "__main__":

    metadata = pd.read_csv(METADATA)
    del metadata['Unnamed: 0']

    try:
        processed_csv = pd.read_csv(PROCESSED_CSV)  # Make this file if you don't have it yet
        del processed_csv['Unnamed: 0']  # Remove unwanted column
    except:
        processed_csv = pd.DataFrame(
            {'Index': 0,
             "Time": -1,
             "Vpp": -1,
             "Frequency": -1,
             "Action": -1}, index=[0]
        )

    vpp = 0
    freq = 0
    action = 0

    for n, datapoint in tqdm.tqdm(metadata.iterrows()):

        try:

            if datapoint['Time'] in processed_metadata['Time'].tolist():
                continue

            if new_vpp != vpp or new_freq != freq and new_action != action:
                closest = np.argmin(np.array(np.abs(metadata[n:n + 100]['Time'])) - datapoint['Time'] - 1)
                # print(closest)

            for i in range(NUM_CLUSTER):

                data = {"Time": datapoint['Time'],
                        "Vpp": datapoint['Vpp'],
                        "Frequency": datapoint['Frequency'],
                        "Action": datapoint['Action']}

                pos, size = make_tuple(datapoint[f"Cluster{i}"])

        except:
            print('Could not complete process.')
            continue

    for n, datapoint in tqdm.tqdm(metadata.iterrows()):

        if n % 60 < 30 and n < len(metadata)-1:

            for i in range(NUM_CLUSTER):

                data = {"Time": datapoint['Time'],
                        "Vpp": datapoint['Vpp'],
                        "Frequency": datapoint['Frequency'],
                        "Action": datapoint['Action']}

                pos, size = make_tuple(datapoint[f"Cluster{i}"])
                pos_plus_30 = make_tuple(metadata[f"Cluster{i}"][n + 30])[0]

                data.__setitem__("Cluster", i)
                data.__setitem__("Size", size)
                data.__setitem__("X0", pos[0])
                data.__setitem__("Y0", pos[1])
                data.__setitem__("X1", pos_plus_30[0])
                data.__setitem__("Y1", pos_plus_30[1])

                if None in data.values():
                    continue

                offset = np.array(pos_plus_30) - np.array(pos)
                magnitude = np.linalg.norm(offset)

                if not magnitude:
                    continue

                vector = tuple(offset / magnitude)

                data.__setitem__("Magnitude", magnitude)
                data.__setitem__("dX", vector[0])
                data.__setitem__("dY", vector[1])

                processed_csv = processed_csv.append(data, ignore_index=True)

        if not n % 60:
            processed_csv.to_csv(PROCESSED_CSV)


