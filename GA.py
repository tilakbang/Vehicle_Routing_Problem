import numpy as np
import pandas as pd
import random
from datetime import datetime
import math
import itertools
from tqdm import tqdm
from math import radians, cos, sin, asin, sqrt

def haversine(lat1,lon1,lat2,lon2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

class Chromosome:
    def __init__(self, genes, fitness=None):
        self.genes = genes
        self.fitness = fitness

    def __str__(self):
        return f'Genes: {self.genes}, Fitness: {self.fitness}'

    def __repr__(self):
        return self.__str__()

    def calculate_fitness(self, start_lat, start_lon, data, capacity):
        total_distance = 0
        total_load = 0
        current_lat = start_lat
        current_lon = start_lon

        for i in range(len(self.genes)):
            current_point = data[self.genes[i]]
            next_point = data[self.genes[(i+1) % len(self.genes)]]
            total_distance += haversine(current_lat, current_lon, next_point[0], next_point[1])
            total_load += current_point[2]

            if total_load > capacity:
                self.fitness = 1 / total_distance
                return

            current_lat = next_point[0]
            current_lon = next_point[1]

        self.fitness = 1 / total_distance

class GeneticAlgorithm:
    def __init__(self, population_size, data, start_lat, start_lon, capacity, num_generations=20, elitism_ratio=0.1, mutation_probability=1):
        self.population_size = population_size
        self.data = data
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.capacity = capacity
        self.num_generations = num_generations
        self.elitism_ratio = elitism_ratio
        self.mutation_probability = mutation_probability
        self.population = []

    def run(self):
        self.initialize_population()
        for i in range(self.num_generations):
            self.evaluate_fitness()
            self.sort_population()
            if i % 1 == 0:
                print(f'Generation {i}: {self.population[0]}')
            next_generation = self.get_elites()
            while len(next_generation) < self.population_size:
                parent1 = self.select_parent()
                parent2 = self.select_parent()
                child1, child2 = self.crossover(parent1, parent2)
                self.mutate(child1)
                self.mutate(child2)
                next_generation.append(child1)
                if len(next_generation) < self.population_size:
                    next_generation.append(child2)
            self.population = next_generation

    def initialize_population(self):
        for i in range(self.population_size):
            genes = list(range(len(self.data)))
            random.shuffle(genes)
            chromosome = Chromosome(genes)
            self.population.append(chromosome)

    def evaluate_fitness(self):
        for chromosome in self.population:
            if chromosome.fitness is None:
                chromosome.calculate_fitness(self.start_lat, self.start_lon, self.data, self.capacity)

    def sort_population(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)

    def get_elites(self):
        num_elites = int(self.elitism_ratio * self.population_size)
        return self.population[:num_elites]

    def select_parent(self):
        tournament_size = 3
        tournament_population = random.sample(self.population, tournament_size)
        tournament_population.sort(key=lambda x: x.fitness, reverse=True)
        return tournament_population[0]

    def crossover(self, parent1, parent2):
        genes1 = parent1.genes.copy()
        genes2 = parent2.genes.copy()
        crossover_point1 = random.randint(0, len(genes1))
        crossover_point2 = random.randint(0, len(genes2))
        genes1[crossover_point1:crossover_point2], genes2[crossover_point1:crossover_point2] = genes2[crossover_point1:crossover_point2], genes1[crossover_point1:crossover_point2]
        child1 = Chromosome(genes1)
        child2 = Chromosome(genes2)
        return child1, child2

    def mutate(self, chromosome):
        for i in range(len(chromosome.genes)):
            if random.random() < self.mutation_probability:
                j = random.randint(0, len(chromosome.genes) - 1)
                chromosome.genes[i], chromosome.genes[j] = chromosome.genes[j], chromosome.genes[i]


def load_data(filename):
    data = []
    with open(filename) as f:
        lines = f.readlines()[1:]
    for line in lines:
        parts = line.split(',')
        lat = float(parts[0])
        lon = float(parts[1])
        demand = float(parts[2])
        data.append((lat, lon, demand))
    return data

data = load_data('Route_Data_Small.csv')
start_lat, start_lon = 51.752022, -1.257677
capacity = 200
genetic_algorithm = GeneticAlgorithm(population_size=100, data=data, start_lat=start_lat, start_lon=start_lon, capacity=capacity)
genetic_algorithm.run()
print(f'Best Route: {genetic_algorithm.population[0].genes}')



