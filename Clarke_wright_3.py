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
        dist_matrix[i][j] = calculate_distance(data["Latitude"][i],
                                               data["Longitude"][i],
                                               data["Latitude"][j],
                                               data["Longitude"][j])
        print(dist_matrix[i][j])
    print()

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
            if not visited[
                    j] and cluster_load + data["load"][j] <= vehicle_capacity:
                cluster.append(j)
                cluster_load += data["load"][j]
                visited[j] = True
        clusters.append(cluster)
        cluster_loads.append(cluster_load)

# print(f'r: {r}')
# print(f'clusters: {clusters}')
# print(f'cluster_loads: {cluster_loads}')

# Calculate savings
savings = {}
for i in range(1, n):
    for j in range(i + 1, n):
        savings[(i, j)] = dist_matrix[0][i] + dist_matrix[0][j] - dist_matrix[i][j]

# Sort savings
nodes = sorted(savings.items(), key=lambda x: x[1], reverse=True)


# Define a function to generate routes for a given cluster
def generate_routes(cluster):
    # Create a list of tuples (index, load) for each node in the cluster
    # nodes = [(i, data["load"][i]) for i in cluster]

    # Sort the nodes by descending order of their loads
    # nodes = sorted(nodes, key=lambda x: x[1], reverse=True)

    # Create a route for each node
    routes = [[node] for node in nodes]

    # Apply Clarke-Wright savings algorithm to merge routes
    for i in range(len(nodes) - 1):
        for j in range(i + 1, len(nodes)):
            # Calculate the savings for merging the two routes
            # savings = dist_matrix[nodes[i][0]][0] + dist_matrix[
            #     nodes[j][0]][0] - dist_matrix[nodes[i][0]][nodes[j][0]]

            # Check if merging the routes would violate capacity constraint
            if nodes[i][1] + nodes[j][1] <= vehicle_capacity:
                # Find the routes that contain the two nodes
                route_i_index = -1
                route_j_index = -1
                for k in range(len(routes)):
                    if nodes[i] in routes[k]:
                        route_i_index = k
                    if nodes[j] in routes[k]:
                        route_j_index = k

                # Merge the two routes
                if route_i_index != route_j_index:
                    routes[route_i_index] += routes[route_j_index]
                    routes.pop(route_j_index)

    # Add the starting and ending point to each route
    for i in range(len(routes)):
        routes[i].insert(0, (0, 0))
        routes[i].append((0, 0))

    # Calculate the total distance of each route
    total_distances = []
    for route in routes:
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += dist_matrix[route[i][0]][route[i + 1][0]]
        total_distances.append(total_distance)

    return routes, total_distances


# Generate routes for each cluster
cluster_routes = []
cluster_distances = []
for i in range(len(clusters)):
    routes, total_distances = generate_routes(clusters[i])
    cluster_routes.append(routes)
    cluster_distances.append(total_distances)

# Print the routes and distances for each cluster
for i in range(len(clusters)):
    print(f"Routes for cluster {i+1}:")
    for j in range(len(cluster_routes[i])):
        route = cluster_routes[i][j]
        distance = cluster_distances[i][j]
        print(
            f"Route {j+1}: {' -> '.join(str(node[0]) for node in route)}, Load: {sum(node[1] for node in route[1:-1])}, Distance: {distance:.2f}"
        )
