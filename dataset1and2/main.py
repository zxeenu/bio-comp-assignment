from dataclasses import dataclass

import numpy as np
import pandas as pd

from population import Popultion


@dataclass
class DataSet:
    input_d: "np.array"
    output_d: str


############# DATA SET 1 LOGIC START #############

dataset1 = pd.read_csv("data1.txt", delim_whitespace=True, dtype=str)
dataset1_len = len(dataset1)
dataset1_prepared = []

for num in range(0, dataset1_len):
    input_val = dataset1.loc[num][0]
    output_val = dataset1.loc[num][1]
    input_np = np.frombuffer(input_val.encode(), dtype=np.uint8) - ord("0")
    x = DataSet(input_np, output_val)
    dataset1_prepared.append(x)


# parameters for dataset2
D1_MUTATION_RATE = 0.1  # must be between 0 and 1
D1_POP_SIZE = 200
D1_MAX_GEN = 200
D1_CHROMOSOME_LENGTH = 5  # handles up to size 14 # 5 seems ideal

# variables
D1_PERFECT_SCORE = 32
D1_GRAPH_NAME = "DATA SET 1"

p = Popultion(
    D1_MUTATION_RATE,
    D1_POP_SIZE,
    D1_MAX_GEN,
    D1_PERFECT_SCORE,
    "5bit",
    dataset1_prepared,
    D1_GRAPH_NAME,
    D1_CHROMOSOME_LENGTH,
)
print("DATA SET 1 - START")
p.run_genetic_algorithm()
print("DATA SET 1 - END")

############ DATA SET 1 LOGIC END #############


print("++++++++++++++++++++================++++++++++++++++++")
print("++++++++++++++++++++================++++++++++++++++++")
print("++++++++++++++++++++================++++++++++++++++++")


############# DATA SET 2 LOGIC START #############

dataset2 = pd.read_csv("data2.txt", delim_whitespace=True, dtype=str)
dataset2_len = len(dataset2)
dataset2_prepared = []

for num in range(0, dataset2_len):
    input_val = dataset2.loc[num][0]
    output_val = dataset2.loc[num][1]
    input_np = np.frombuffer(input_val.encode(), dtype=np.uint8) - ord("0")
    x = DataSet(input_np, output_val)
    dataset2_prepared.append(x)

# parameters for dataset2
D2_MUTATION_RATE = 0.1  # must be between 0 and 1
D2_POP_SIZE = 100
D2_MAX_GEN = 100
D2_CHROMOSOME_LENGTH = 13

# variables
D2_PERFECT_SCORE = 64
D2_GRAPH_NAME = "DATA SET 2"

p2 = Popultion(
    D2_MUTATION_RATE,
    D2_POP_SIZE,
    D2_MAX_GEN,
    D2_PERFECT_SCORE,
    "6bit",
    dataset2_prepared,
    D2_GRAPH_NAME,
    D2_CHROMOSOME_LENGTH,
)
print("DATA SET 2 - START")
p2.run_genetic_algorithm()
print("DATA SET 2 - END")

############# DATA SET 2 LOGIC END #############
input("Press any button to close")
