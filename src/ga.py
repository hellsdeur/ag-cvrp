from typing import Callable, List
from random import uniform
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Solution:
    def __init__(self, f: Callable, individuals: List = [], chromosome: List = []):
        if individuals:
            indexes = np.random.choice([i for i in range(len(individuals))], len(individuals), replace=False)
            self.chromosome = [individuals[idx] for idx in indexes]
        else:
            self.chromosome = chromosome
        self.fitness = f(self.chromosome)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < len(self.chromosome):
            current = self.chromosome[self.i]
            self.i += 1
            return current
        else:
            raise StopIteration

    def __getitem__(self, i: int):
        return self.chromosome[i]

    def __setitem__(self, idx, value):
        self.chromosome[idx] = value

    def __len__(self):
        return len(self.chromosome)

    def __str__(self):
        str_chromosome = "Chromosomes: " + "\t|\t".join([str(gene) for gene in self.chromosome])
        str_fitness = f"Fitness: {self.fitness}"
        return f"{str_chromosome}\n{str_fitness}"


class Population:
    def __init__(self, individuals: List, k: int, f: Callable):
        self.f = f
        self.solutions = [Solution(f=self.f, individuals=individuals) for i in range(k)]

    def append(self, solution):
        self.solutions.append(solution)

    def visualize(self):
        df = pd.DataFrame()
        for i in range(len(self.solutions)):
            s = pd.Series(
                data=[gene.id for gene in self.solutions[i]] + [self.solutions[i].fitness],
                index=[f"x_{j:02d}" for j in range(len(self.solutions[i]))] + ["y"],
            )
            df = df.append(s, ignore_index=True)
            df_int = df.iloc[:, :-1].astype(int)
            df_int["y"] = df["y"]
            df = df_int

        return df

    def __getitem__(self, i: int):
        return self.solutions[i]

    def __len__(self):
        return len(self.solutions)

    def __str__(self):
        return '\n'.join([str(solution) for solution in self.solutions])


class GeneticAlgorithm:
    def __init__(self, f: Callable, individuals: List, k: int, minmax: str, selection: str):
        self.f = f                      # objective function
        self.individuals = individuals  # combinatorial candidates
        self.k = k                      # size of population
        self.population = None          # current population of solutions
        self.minmax = minmax            # min or max
        self.selection = selection      # generational, elitist or steady
        self.gbest_list = []            # list of global best

    def generate_population(self, k: int):
        self.population = Population(self.individuals, k, self.f)

    def select_parents(self):
        sum_fitnesses = sum([solution.fitness for solution in self.population])
        probability_distribution = [solution.fitness / sum_fitnesses for solution in self.population]

        # 2 indexes among k individuals are chosen, without repetition, accordingly with probability distribution
        parents_index = np.random.choice(len(self.population), 2, p=probability_distribution, replace=False)

        return [self.population[i] for i in parents_index]

    def crossover(self, parents: List[List], operator: str = "LOX"):
        if operator == "LOX":
            # get two sorted indexes, from a index list
            indexes = [i for i in range(1, len(parents[0])-1)]
            cut = sorted(np.random.choice(indexes, 2, replace=False))

            # store subchains
            subchain_p0 = parents[0][cut[0]: cut[1]]
            subchain_p1 = parents[1][cut[0]: cut[1]]

            # if element in parent belongs to substring of the other parent, then hide element
            for i in range(len(parents[0])):
                if parents[0][i] in subchain_p1:
                    parents[0] = parents[0][:i] + ['h'] + parents[0][i+1:]
                if parents[1][i] in subchain_p0:
                    parents[1] = parents[1][:i] + ['h'] + parents[1][i + 1:]

            # remove hidden
            parents[0] = [gene for gene in parents[0] if gene != "h"]
            parents[1] = [gene for gene in parents[1] if gene != "h"]

            # insert subchain of the other parent at the cut begin
            parents[0] = parents[0][:cut[0]] + subchain_p1 + parents[0][cut[0]:]
            parents[1] = parents[1][:cut[0]] + subchain_p0 + parents[1][cut[0]:]

            return [Solution(f=self.f, chromosome=parents[0]), Solution(f=self.f, chromosome=parents[1])]

    def mutate(self, child: Solution, mutation_rate):
        # if a drawn number between 0 and 1 is less or equal to mutation_rate, then mutate
        if uniform(0, 1) <= mutation_rate:
            # get two indexes, from a index list
            indexes = [i for i in range(0, len(child))]
            permute = np.random.choice(indexes, 2, replace=False)

            # permute the positions
            child[permute[0]], child[permute[1]] = child[permute[1]], child[permute[0]]

        return child

    def select_survivors(self, mutated_children: List):
        if self.selection == "generational":
            # overwrite population with an empty sample
            self.population = Population(self.individuals, 0, self.f)

            # append mutated children
            for child in mutated_children:
                self.population.append(child)

            # fill the blank spots
            while len(self.population) < self.k:
                self.population.append(Solution(f=self.f, individuals=self.individuals))

        elif self.selection == "elitist":
            # only the mutated children are candidates to selection
            children_population = Population(self.individuals, 0, self.f)
            for child in mutated_children:
                children_population.append(child)
            candidates = sorted(
                children_population.solutions,
                key=lambda x: x.fitness,
                reverse=True if self.minmax == "max" else False
            )

            # overwrite population with an empty sample
            self.population = Population(self.individuals, 0, self.f)

            # append best mutated children
            self.population.append(candidates[0])

            # fill the blank spots
            while len(self.population) < self.k:
                self.population.append(Solution(f=self.f, individuals=self.individuals))

        elif self.selection == "steady":
            # mutated children are added to population
            for child in mutated_children:
                self.population.append(child)

            # whole population is candidate to selection
            candidates = sorted(
                self.population.solutions,
                key=lambda x: x.fitness,
                reverse=True if self.minmax == "max" else False
            )

            # overwrite population with an empty sample
            self.population = Population(self.individuals, 0, self.f)

            # append the top half candidates
            for i in range(0, round(self.k/2)):
                self.population.append(candidates[i])

            # fill the blank spots
            while len(self.population) < self.k:
                self.population.append(Solution(f=self.f, individuals=self.individuals))

    def compute_gbest(self):
        candidates = sorted(
            self.population.solutions,
            key=lambda x: x.fitness,
            reverse=True if self.minmax == "max" else False
        )
        self.gbest_list.append(candidates[0])

    def run(self, mutation_rate: float, generations: int):

        def progress_bar(i):
            pc = int(i / generations * 100)
            logic_char = 0 if i == 1 else int(pc / 2)
            print(f"Current generation {i}/{generations}:|{'█' * logic_char + '.' * (50 - logic_char)}|{pc}%",
                  end='\r', flush=True)

        self.generate_population(self.k)

        for g in range(0, generations):

            parents = self.select_parents()

            children = self.crossover(parents)

            mutated_children = [self.mutate(child, mutation_rate) for child in children]

            self.select_survivors(mutated_children)

            self.compute_gbest()

            progress_bar(g+1)

        print(f"\n\tBest fitness: {self.gbest_list[-1].fitness}")


class MeanBehavior:
    def __init__(self, ga: GeneticAlgorithm, n_run: int, mutation_rate: float, generations: int):
        self.ga = ga
        self.n_run = n_run
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.run_list = []

    def process(self):
        for i in range(self.n_run):
            print(f"Run {i + 1}/{self.n_run}")
            current_ga = deepcopy(self.ga)
            current_ga.run(mutation_rate=self.mutation_rate, generations=self.generations)
            self.run_list.append(current_ga)

    def describe(self):
        costs = []
        for run in self.run_list:
            costs.append([solution.fitness for solution in run.gbest_list])

        mean_solutions = []
        std_solutions = []

        for i in range(len(costs[0])):
            mean = []
            std = []
            for j in range(len(costs)):
                mean.append(costs[j][i])
                std.append(costs[j][i])
            mean_solutions.append(np.mean(mean))
            std_solutions.append(np.std(std))

        return pd.DataFrame([mean_solutions, std_solutions], index=['mean', 'std']).T

    def plot(self, description: pd.DataFrame):
        x = np.arange(0, self.generations, 1)
        y = description["mean"].values
        yerr = description["std"].values

        colors = ['red', 'blue', 'green', 'orange', 'black', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

        fig, (ax0) = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(12, 6))
        ax0.plot(x, y, color='red', lw=1)
        ax0.set_title('Convergence between runs')
        ax0.set_xlabel('Generation')
        ax0.set_ylabel('Cost')

        ax0.fill_between(x, y - yerr, y + yerr, color='blue', alpha=0.2, linewidth=0.0)
        ax0.legend(['Mean best solution', 'Standard Deviation solution'])
        plt.savefig('results/graphic_convergence_executions', dpi=300)
        plt.show()
