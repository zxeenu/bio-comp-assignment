import matplotlib.pyplot as plt
import numpy as np

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
        _dataset_name,
        _chromosome_len
    ):
        self.mutation_rate = _mutation_rate
        self.population_size = _population_size
        self.max_generations = _max_generations
        self.perfect_score = _perfect_score
        self.type = _type
        self.dataset = _dataset
        self.chromosome_len = _chromosome_len

        self.current_pop = []  # current population

        self.current_best_chromosome = Chromosome()  # the best chromosome in the entire runtime
        self.best_fitness_records = []  # best fitness of each generation
        self.average_fitness_records = [] # average fitness of each generation

        self.is_finished = False
        self.current_generation = 0
        self.dataset_name = _dataset_name

    def initialize(self):
        for _ in range(0, self.population_size):
            if self.type == "5bit":
                chrom = Chromosome()
                chrom.initialize(self.chromosome_len, self.type)
                self.current_pop.append(chrom)
            elif self.type == "6bit":
                chrom = Chromosome()
                chrom.initialize(self.chromosome_len, self.type)
                self.current_pop.append(chrom)

    def calculate_fitness(self):
        # variables
        this_gen_fitness_records = []
        this_gen_best_chromosome = Chromosome()
        avg_fitness = 0.0
        total_fitness = 0

        # calculating fitness and getting ref to best chromeosome in current gen
        for chromosome in self.current_pop:
            chromosome.calculate_fitness(self.dataset)
            this_gen_fitness_records.append(chromosome.fitness)

            if chromosome.fitness > this_gen_best_chromosome.fitness:
                this_gen_best_chromosome = chromosome

        # sum all of this gens fitness scores
        for fit in this_gen_fitness_records:
            total_fitness += fit

        avg_fitness = total_fitness / len(this_gen_fitness_records)
        self.average_fitness_records.append(avg_fitness)
        self.best_fitness_records.append(this_gen_best_chromosome.fitness)

        # update runs times best fitness records
        if this_gen_best_chromosome.fitness > self.current_best_chromosome.fitness:
            self.current_best_chromosome = this_gen_best_chromosome

    def accept_reject(self):
        count = 0
        while True:
            count+=1
            potential_mate = np.random.choice(self.current_pop)
            random_fitness_value = np.random.randint(0, self.current_best_chromosome.fitness)

            if random_fitness_value < potential_mate.fitness:
                return potential_mate

            # failsafe in case stuck in infitie loop
            if count > 100:
                print("Warning: infinite loop failsafe used!")
                return potential_mate

    def natural_selection(self):
        new_population = []

        for _ in range(0, self.population_size):
            parent_a = self.accept_reject()
            parent_b = self.accept_reject()

            # cross over returns 2 children (mid point cross over) and a clone of parent_a
            children = parent_a.cross_over(parent_b)

            children[0].mutate(self.mutation_rate)
            children[1].mutate(self.mutation_rate)
            children[2].mutate(self.mutation_rate)

            children[0].calculate_fitness(self.dataset)
            children[1].calculate_fitness(self.dataset)
            children[2].calculate_fitness(self.dataset)

            # rank selection among the children
            if children[0].fitness > children[1].fitness:
                golden_child = children[0]
            else:
                golden_child = children[1]

            if children[2].fitness > golden_child.fitness:
                golden_child = children[2]

            # golden_child = np.random.choice(children)
            # golden_child.calculate_fitness(self.dataset)
            # golden_child.mutate(self.mutation_rate)

            new_population.append(golden_child)

        # replace the current population with the new population
        self.current_pop = new_population

    def evaluate(self):
        if self.perfect_score == self.current_best_chromosome.fitness:
            self.is_finished = True


    def print_details(self):
        print("current generation", self.current_generation)
        print("best fitness records up till for entire simulation", self.current_best_chromosome.fitness)
        print("current best classifier", self.current_best_chromosome.dna)

    def graph(self):
        x_axis = [x for x in range(0, self.max_generations)]
        fig = plt.figure(figsize=(7,5))
        ax = fig.add_subplot()
        plt.grid(b=True, which='major', linewidth='0.9')
        ax.plot(x_axis, self.average_fitness_records, label='Average Fitness')
        ax.plot(x_axis, self.best_fitness_records, label='Best Fitness')
        ax.set_xlabel('Generations')
        ax.set_ylabel('Fitness')
        ax.set_title(self.dataset_name)
        chart_text = f"Mutation Rate: {self.mutation_rate}\nPopulation Size: {self.population_size}\nMax Generation: {self.max_generations}\nPerfect Fitness: {self.perfect_score}\nBest Fitness: {self.current_best_chromosome.fitness}"
        ax.text(-0.15, .99, chart_text, size=8,
         transform=ax.transAxes, color='r')
        ax.legend()
        plt.show()

    def run_genetic_algorithm(self):
        self.initialize()

        local_counter = 0
        for current_gen in range(0, self.max_generations):
            self.calculate_fitness()
            self.natural_selection()
            self.evaluate()
            self.current_generation += 1

            local_counter += 1
            if local_counter > 20:
                print("+++++++++++==================+++++++++++")
                self.print_details()
                print("+++++++++++==================+++++++++++")
                local_counter = 0

            if self.is_finished == True:
                break

        print("average fitness records", self.average_fitness_records)
        print("current best fitness records", self.best_fitness_records)
        print("+++++++++++==================+++++++++++")
        self.graph()

        return self.is_finished

