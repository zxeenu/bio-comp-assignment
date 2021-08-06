from chromosome import Chromosome


class Popultion:
    def __init__(
        self,
        _mutation_rate: "between 0 and 1",
        _population_size,
        _max_generations,
        _perfect_score,
        _type,
        _dataset,
    ):
        self.mutation_rate = _mutation_rate
        self.population_size = _population_size
        self.generations = _max_generations
        self.perfect_score = _perfect_score
        self.type = _type
        self.dataset = _dataset

        self.current_pop = []  # current population
        self.mating_pool = []
        self.best_Chromosome = Chromosome()
        self.fitness_records = []
        self.average_fitness = 0
        self.is_finished = False

    def initialize(self):
        for _ in range(0, self.population_size):
            if self.type == "5bit":
                chrom = Chromosome()
                chrom.initialize_for_5bit()
                self.current_pop.append(chrom)

    def calculate_fitness(self):
        for chromosome in self.current_pop:
            chromosome.calculate_fitness(self.dataset)
            print(chromosome.fitness)

    def print_pop(self):
        for _ in self.current_pop:
            print(_.dna)


# MUTATION_RATE = 0.5
# POP_SIZE = 100
# MAX_GEN = 500
# PERFECT_SCORE = 32

# p = Popultion(MUTATION_RATE, POP_SIZE, MAX_GEN, PERFECT_SCORE, "5bit")
# p.initialize()
# p.print_pop()
