from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DataSet:
    input_d: "np.array"
    output_d: int


############# DATA SET 3 LOGIC START #############
dataset3 = pd.read_csv("data3.txt", delim_whitespace=True, dtype=str)
dataset3_len = len(dataset3)
dataset3_prepared = []

for num in range(0, dataset3_len):
    input_val_0 = dataset3.loc[num][0]
    input_val_1 = dataset3.loc[num][1]
    input_val_2 = dataset3.loc[num][2]
    input_val_3 = dataset3.loc[num][3]
    input_val_4 = dataset3.loc[num][4]
    input_val_5 = dataset3.loc[num][5]
    output_val = dataset3.loc[num][6]

    input_np_str = [
        input_val_0,
        input_val_1,
        input_val_2,
        input_val_3,
        input_val_4,
        input_val_5,
    ]
    input_np_float = np.asarray(input_np_str, dtype=np.float64, order="C")
    output_int = int(output_val)

    x = DataSet(input_np_float, output_int)
    dataset3_prepared.append(x)


print(len(dataset3_prepared))
