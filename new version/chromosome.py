import copy
import random
from enum import Enum

import numpy as np


class LogicGate(Enum):
    OR = 1
    AND = 2
    XOR = 3
    NAND = 4
    NOR = 5
    XNOR = 6

# class LogicGate(Enum):
#     OR = 1
#     AND = 2
#     XOR = 3
#     XNOR = 4


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
    def XNOR(a, b) -> int:
        if a != b:
            return 0
        else:
            return 1

    @staticmethod
    def NOR(a, b) -> int:
        if a == True:
            return 0
        elif b == True:
            return 0
        else:
            return 1

    @staticmethod
    def NAND(a, b) -> int:
        if a == True and b == True:
            return 0
        else:
            return 1

    @staticmethod
    def Smart(gene, a, b) -> int:
        if gene == LogicGate.OR:
            return LogicGateSim.OR(a, b)
        elif gene == LogicGate.AND:
            return LogicGateSim.AND(a, b)
        elif gene == LogicGate.XOR:
            return LogicGateSim.XOR(a, b)
        elif gene == LogicGate.NAND:
            return LogicGateSim.NAND(a, b)
        elif gene == LogicGate.NOR:
            return LogicGateSim.NOR(a, b)
        elif gene == LogicGate.XNOR:
            return LogicGateSim.XNOR(a, b)



class Chromosome:
    def __init__(self):
        self.dna = []
        self.fitness = 0
        self.dna_length = 0  # in case this data is needed
        self.type = ""

    def initialize(self, length, type_name):
        self.type = type_name

        self.dna_length = length
        self.dna = np.random.choice(list(LogicGate), size=self.dna_length, replace=True)

    def initialize_with_dna(self, _dna, _type):
        self.dna = _dna
        self.type = _type

    # def initialize_for_5bit(self):
    #     dna_length = 10 # 4  # 8
    #     self.initialize(dna_length)
    #     self.type = "5bit"

    # def initialize_for_6bit(self):
    #     dna_length = 10 # 4  # 8
    #     self.initialize(dna_length)
    #     self.type = "6bit"

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

        child_a = Chromosome()
        child_a.initialize_with_dna(child_a_dna, self.type)

        child_b = Chromosome()
        child_b.initialize_with_dna(child_b_dna, self.type)

        child_clone = Chromosome()
        child_clone.initialize_with_dna(self.dna, self.type)

        return [child_a, child_b, child_clone]

    def mutate(self, mutation_true=0.5) -> bool:
        mutation_false = 1 - mutation_true
        do_mutate = np.random.choice(
            [True, False], p=[mutation_true, mutation_false], size=1
        )

        if do_mutate:
            gene_random = np.random.choice(list(LogicGate), size=1)[0]
            gene_index = random.randint(0, len(self.dna) - 1)
            self.dna[gene_index] = gene_random
            # print("mutation applied")
            return True
        else:
            return False

    def calculate_fitness(self, dataset: "array-like"):
        temp_fitness = 0
        output_value = 1 # bias
        for data in dataset:
            if self.type == "5bit":
                for count, gene in enumerate(self.dna):
                    # applying the classifier against the dataset

                    if count == 0:
                        output_value = LogicGateSim.Smart(
                            gene, output_value, data.input_d[0]
                        )
                    elif count == 1:
                        output_value = LogicGateSim.Smart(
                            gene, output_value, data.input_d[1]
                        )
                    elif count == 2:
                        output_value = LogicGateSim.Smart(
                            gene, output_value, data.input_d[2]
                        )
                    elif count == 3:
                        output_value = LogicGateSim.Smart(
                            gene, output_value, data.input_d[3]
                        )
                    elif count == 4:
                        output_value = LogicGateSim.Smart(
                            gene, output_value, data.input_d[4]
                        )
                    elif count == 5:
                        output_value = LogicGateSim.Smart(
                            gene, output_value, data.input_d[1]
                        )
                    elif count == 6:
                        output_value = LogicGateSim.Smart(
                            gene, output_value, data.input_d[2]
                        )
                    elif count == 7:
                        output_value = LogicGateSim.Smart(
                            gene, output_value, data.input_d[3]
                        )
                    elif count == 8:
                        output_value = LogicGateSim.Smart(
                            gene, output_value, data.input_d[4]
                        )
                    elif count == 9:
                        output_value = LogicGateSim.Smart(
                            gene, output_value, data.input_d[1]
                        )
                    elif count == 10:
                        output_value = LogicGateSim.Smart(
                            gene, output_value, data.input_d[0]
                        )

                    # comparing the end result against the dataset's expected output
                if str(data.output_d) == str(output_value):
                    # print(data.output_d, output_value)
                    temp_fitness += 1
            elif self.type == "6bit":
                for count, gene in enumerate(self.dna):
                    # applying the classifier against the dataset

                    if count == 0:
                        output_value = LogicGateSim.Smart(
                            gene, output_value, data.input_d[0]
                        )
                    elif count == 1:
                        # print("1", LogicGateSim.Smart(gene, 1, 0))
                        output_value = LogicGateSim.Smart(
                            gene, output_value, data.input_d[2]
                        )
                    elif count == 2:
                        output_value = LogicGateSim.Smart(
                            gene, output_value, data.input_d[5]
                        )
                    elif count == 3:
                        output_value = LogicGateSim.Smart(
                            gene, output_value, data.input_d[1]
                        )
                    elif count == 4:
                        output_value = LogicGateSim.Smart(
                            gene, output_value, data.input_d[4]
                        )
                    elif count == 5:
                        output_value = LogicGateSim.Smart(
                            gene, output_value, data.input_d[1]
                        )
                    elif count == 6:
                        output_value = LogicGateSim.Smart(
                            gene, output_value, data.input_d[3]
                        )
                    elif count == 7:
                        output_value = LogicGateSim.Smart(
                            gene, output_value, data.input_d[2]
                        )
                    elif count == 8:
                        output_value = LogicGateSim.Smart(
                            gene, output_value, data.input_d[0]
                        )
                    elif count == 9:
                        output_value = LogicGateSim.Smart(
                            gene, output_value, data.input_d[5]
                        )
                    elif count == 10:
                        output_value = LogicGateSim.Smart(
                            gene, output_value, data.input_d[3]
                        )

                    # comparing the end result against the dataset's expected output
                if str(data.output_d) == str(output_value):
                    # print(data.output_d, output_value)
                    temp_fitness += 1

        self.fitness = temp_fitness
