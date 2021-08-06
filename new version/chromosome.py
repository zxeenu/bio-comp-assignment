import copy
import random
from enum import Enum

import numpy as np


class LogicGate(Enum):
    OR = 1
    AND = 2
    XOR = 3


class LogicGateSim:
    @staticmethod
    def AND(a, b) -> int:
        if a == True and b == True:
            return 1
        else:
            return 0

    @staticmethod
    def OR(a, b) -> int:
        if a == True:
            return 1
        elif b == True:
            return 1
        else:
            return 0

    @staticmethod
    def XOR(a, b) -> int:
        if a != b:
            return 1
        else:
            return 0

    @staticmethod
    def Smart(gene, a, b) -> int:
        if gene == LogicGate.OR:
            return LogicGateSim.OR(a, b)
        elif gene == LogicGate.AND:
            return LogicGateSim.AND(a, b)
        elif gene == LogicGate.XOR:
            return LogicGateSim.XOR(a, b)


class Chromosome:
    def __init__(self):
        self.dna = []
        self.fitness = 0
        self.dna_length = 0  # in case this data is needed
        self.type = ""

    def initialize(self, length):
        self.dna = np.random.choice(list(LogicGate), size=length, replace=True)

    def initialize_with_dna(self, _dna):
        self.dna = _dna

    def initialize_for_5bit(self):
        self.initialize(4)
        self.dna_length = 4
        self.type = "5bit"

    def cross_over(self, mate_chromosome) -> list:

        this_dna = copy.deepcopy(self.dna)
        mate_dna = copy.deepcopy(mate_chromosome.dna)

        this_chromosome_split = np.array_split(this_dna, 2)
        mate_chromosome_split = np.array_split(mate_dna, 2)

        child_a_dna = np.concatenate(
            (this_chromosome_split[0], mate_chromosome_split[1])
        )

        child_b_dna = np.concatenate(
            (mate_chromosome_split[0], this_chromosome_split[1])
        )

        # print("------this")
        # print(this_chromosome_split[0])
        # print(this_chromosome_split[1])
        # print("------mate")
        # print(mate_chromosome_split[0])
        # print(mate_chromosome_split[1])
        # print("------child a")
        # print(child_a_dna)
        # print("------child b")
        # child_clone = Chromosome()

        child_a = Chromosome()
        child_a.initialize_with_dna(child_a_dna)

        child_b = Chromosome()
        child_b.initialize_with_dna(child_b_dna)

        child_clone = Chromosome()
        child_clone.initialize_with_dna(self.dna)

        return [child_a, child_b, child_clone]

    def mutate(self, mutation_true=0.5) -> bool:
        mutation_false = 1 - mutation_true
        do_mutate = np.random.choice(
            [True, False], p=[mutation_true, mutation_false], size=1
        )

        if do_mutate:
            gene_random = np.random.choice(list(LogicGate), size=1)[0]
            # print("---", gene_random)
            gene_index = random.randint(0, len(self.dna) - 1)
            self.dna[gene_index] = gene_random
            # print(self.dna)
            return True
        else:
            return False

    def calculate_fitness(self, dataset: "array-like"):
        for data in dataset:
            if self.type == "5bit":
                for count, gene in enumerate(self.dna):
                    # applying the classifier against the dataset
                    if count == 0:
                        output_value = LogicGateSim.Smart(
                            gene, data.input_d[0], data.input_d[1]
                        )
                        # print(output_value)
                    elif count == 1:
                        # print("1", LogicGateSim.Smart(gene, 1, 0))
                        output_value = LogicGateSim.Smart(
                            gene, output_value, data.input_d[2]
                        )
                        # print(output_value)
                    elif count == 2:
                        output_value = LogicGateSim.Smart(
                            gene, output_value, data.input_d[3]
                        )
                        # print(output_value)
                    elif count == 3:
                        output_value = LogicGateSim.Smart(
                            gene, output_value, data.input_d[4]
                        )
                        # print(output_value)

                    # comparing the end result against the dataset's expected output
                if str(data.output_d) == str(output_value):
                    # print(data.output_d, output_value)
                    self.fitness += 1
                else:
                    # print(data.output_d, output_value)
                    pass
        # print("---")
        # print(self.fitness)
