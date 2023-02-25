import pandas as pd
import numpy as np
from haversine import haversine

# Define the starting point
start_lat, start_lon = 18, 73

# Define the vehicle capacity
vehicle_capacity = 250

# Load the dataset
data = pd.read_csv("Route_Data_Small.csv")

# Calculate the distance between each pair of nodes
def calculate_distance(lat1, lon1, lat2, lon2):
    return haversine((lat1, lon1), (lat2, lon2))

n = data.shape[0]
dist_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        dist_matrix[i][j] = calculate_distance(data["Latitude"][i], data["Longitude"][i],
                                                data["Latitude"][j], data["Longitude"][j])

# Form clusters
clusters = []
cluster_loads = []
visited = [False] * n
for i in range(n):
    if not visited[i]:
        cluster = [i]
        cluster_load = data["load"][i]
        visited[i] = True
        for j in range(i + 1, n):
            if not visited[j] and cluster_load + data["load"][j] <= vehicle_capacity:
                cluster.append(j)
                cluster_load += data["load"][j]
                visited[j] = True
        clusters.append(cluster)
        cluster_loads.append(cluster_load)

# Clarke-Wright savings algorithm
def clarke_wright_savings(cluster):
    n = len(cluster)
    savings = {}
    for i in range(1, n):
        for j in range(i + 1, n):
            savings[(i, j)] = dist_matrix[cluster[0]][i] + dist_matrix[cluster[0]][j] - dist_matrix[i][j]
    sorted_savings = sorted(savings.items(), key=lambda x: x[1], reverse=True)
    used_nodes = set()
    route = [cluster[0]]
    for (i, j), s in sorted_savings:
        if i not in used_nodes and j not in used_nodes and cluster[i] != cluster[j]:
            route.append(i)
            route.append(j)
            used_nodes.add(i)
            used_nodes.add(j)
    for i in range(n):
        if i not in used_nodes:
            route.append(i)
    return route

# Apply Clarke-Wright savings algorithm to each cluster
for i, cluster in enumerate(clusters):
    route = clarke_wright_savings(cluster)
    route = [cluster[i] for i in route]
    route_distance = sum(dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
    print(f"Route {i+1}: {route} - Total distance: {route_distance:.2f}")
