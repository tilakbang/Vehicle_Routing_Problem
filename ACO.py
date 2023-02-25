import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd

class AntColonyOptimization:
    def __init__(self, num_ants, num_iterations, evaporation_rate, alpha, beta, q, pheromone_init, demand, capacity, coordinates, start_idx):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.q = q
        self.pheromone_init = pheromone_init
        self.demand = demand
        self.capacity = capacity
        self.coordinates = coordinates
        self.start_idx = start_idx
        self.num_nodes = len(coordinates)
        self.pheromone = np.ones((self.num_nodes, self.num_nodes)) * pheromone_init
        self.distances = cdist(coordinates, coordinates, 'euclidean')
        np.fill_diagonal(self.distances, np.inf)

    def solve(self):
        best_route = None
        best_distance = np.inf
        for iteration in range(self.num_iterations):
            routes = []
            distances = []
            for ant in range(self.num_ants):
                route, distance = self.generate_route()
                routes.append(route)
                distances.append(distance)
            best_ant_idx = np.argmin(distances)
            if distances[best_ant_idx] < best_distance:
                best_distance = distances[best_ant_idx]
                best_route = routes[best_ant_idx]
            self.update_pheromone(routes, distances)
        return best_route, best_distance

    def generate_route(self):
        route = [self.start_idx]
        remaining_capacity = self.capacity
        remaining_nodes = set(range(self.num_nodes)) - {self.start_idx}
        while remaining_nodes:
            node = self.select_next_node(route[-1], remaining_nodes, remaining_capacity)
            if node is None:
                break
            route.append(node)
            remaining_nodes.remove(node)
            remaining_capacity -= self.demand[node]
        route.append(self.start_idx)
        distance = self.calculate_distance(route)
        return route, distance

    def select_next_node(self, current_node, remaining_nodes, remaining_capacity):
        choices = []
        for node in remaining_nodes:
            if self.demand[node] <= remaining_capacity:
                choices.append((node, self.pheromone[current_node, node]**self.alpha * (1/self.distances[current_node, node])**self.beta))
        if not choices:
            return None
        prob = np.array([choice[1] for choice in choices])
        prob /= prob.sum()
        next_node_idx = np.random.choice(len(choices), p=prob)
        return choices[next_node_idx][0]

    def calculate_distance(self, route):
        return sum(self.distances[route[i], route[i+1]] for i in range(len(route)-1))

    def update_pheromone(self, routes, distances):
        pheromone_delta = np.zeros((self.num_nodes, self.num_nodes))
        for route, distance in zip(routes, distances):
            for i in range(len(route)-1):
                pheromone_delta[route[i], route[i+1]] += self.q/distance
        self.pheromone = self.pheromone*(1-self.evaporation_rate) + pheromone_delta*self.evaporation_rate

# load dataset

data = pd.read_csv('Route_Data_Small.csv')

# extract data from dataset
coordinates = data[['Latitude', 'Longitude']].values
demand = data['load'].values
capacity = 600

# set parameters for ant colony optimization
num_ants = 10
num_iterations = 100
evaporation_rate = 0.5
alpha = 1
beta = 2
q = 100
pheromone_init = 1
start_idx = 0

# run ant colony optimization
aco = AntColonyOptimization(num_ants, num_iterations, evaporation_rate, alpha, beta, q, pheromone_init, demand, capacity, coordinates, start_idx)
best_route, best_distance = aco.solve()

# print results
print(len(best_route))
print(f'Best route: {best_route}')
print(f'Best distance: {best_distance}')
