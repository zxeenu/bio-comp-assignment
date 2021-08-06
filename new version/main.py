from dataclasses import dataclass

import numpy as np
import pandas as pd

# from chromosome import Chromosome
from population import Popultion

# x = Chromosome()
# x.initialize_for_5bit()
# print("x", x.dna)

# y = Chromosome()
# y.initialize_for_5bit()
# print("y", y.dna)

# z = y.cross_over(x)
# # print(x.dna)


# for chro in z:
#     print(chro.dna)


@dataclass
class DataSet:
    input_d: "np.array"
    output_d: str


dataset1 = pd.read_csv("data1.txt", delim_whitespace=True, dtype=str)  # 32
dataset1_len = len(dataset1)
dataset1_prepared = []

for num in range(0, dataset1_len):
    input_val = dataset1.loc[num][0]
    output_val = dataset1.loc[num][1]
    input_np = np.frombuffer(input_val.encode(), dtype=np.uint8) - ord("0")
    x = DataSet(input_np, output_val)
    dataset1_prepared.append(x)

# for data in dataset1_prepared:
#     print(data.input_d)
#     print(data.output_d)

# x = Chromosome()
# x.initialize_for_5bit()
# x.calculate_fitness(dataset1_prepared)

MUTATION_RATE = 0.5
POP_SIZE = 100
MAX_GEN = 500
PERFECT_SCORE = 32

p = Popultion(
    MUTATION_RATE, POP_SIZE, MAX_GEN, PERFECT_SCORE, "5bit", dataset1_prepared
)
p.initialize()
# p.print_pop()
p.calculate_fitness()
