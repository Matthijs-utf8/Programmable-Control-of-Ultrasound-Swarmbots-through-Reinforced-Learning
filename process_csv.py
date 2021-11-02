import numpy as np
import pandas as pd
import tqdm
from ast import literal_eval as make_tuple

METADATA = "E:\\metadata_0.csv"
PROCESSED_CSV = "processed_csv.csv"

if __name__ == "__main__":

    metadata = pd.read_csv(METADATA)
    del metadata['Unnamed: 0']
    processed_csv = pd.read_csv(PROCESSED_CSV)
    del processed_csv['Unnamed: 0']

    for n, datapoint in tqdm.tqdm(metadata.iterrows()):

        if n % 60 < 30 and n < len(metadata)-1:

            for i in range(50):

                data = {"Time": datapoint['Time'],
                        "Vpp": datapoint['Vpp'],
                        "Frequency": datapoint['Frequency'],
                        "Action": datapoint['Action']}

                pos, size = make_tuple(datapoint[f"Cluster{i}"])
                pos_plus_30 = make_tuple(metadata[f"Cluster{i}"][n + 30])[0]

                data.__setitem__("Cluster", i)
                data.__setitem__("Size", size)
                data.__setitem__("Pos0", pos)
                data.__setitem__("Pos1", pos_plus_30)

                if None in data.values() or (None, None) in data.values():
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

        if not n % 240:
            processed_csv.to_csv(PROCESSED_CSV)


