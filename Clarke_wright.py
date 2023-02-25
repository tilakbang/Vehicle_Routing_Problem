import numpy as np
from sklearn.cluster import KMeans

# Load the data from the CSV file
data = np.genfromtxt('Route_Data_Small.csv', delimiter=',')
capacity=200

# Define a function to calculate the Euclidean distance between two points
def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Define a function to calculate the savings for each pair of nodes
def calculate_savings(cluster):
    n = len(cluster)
    savings = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            savings[i][j] = savings[j][i] = distance(cluster[i], cluster[-1]) + distance(cluster[j], cluster[-1]) - distance(cluster[i], cluster[j])
    return savings

# Define a function to find the best route for a given cluster using the Clarke-Wright heuristic
def solve_cvrp(clusters, capacity):
    routes = []
    for cluster in clusters:
        n = len(cluster)

        # Calculate the savings for each pair of nodes
        savings = calculate_savings(cluster)

        # Sort the savings in descending order
        sorted_indices = np.argsort(savings.flatten())[::-1]

        # Initialize the routes and the remaining capacity
        routes_cluster = [[0, i, 0] for i in range(1, n)]
        remaining_capacity = [capacity - cluster[i][2] for i in range(1, n)]

        # Combine the routes by connecting them based on the savings
        for index in sorted_indices:
            i, j = divmod(index, n)
            if savings[i][j] <= 0:
                break
            i_route, j_route = None, None
            for route in routes_cluster:
                if i in route:
                    i_route = route
                if j in route:
                    j_route = route
            if i_route != j_route and remaining_capacity[i] + remaining_capacity[j] >= 0:
                i_index = i_route.index(i)
                j_index = j_route.index(j)
                if i_index == 0 and j_index == len(j_route) - 1:
                    i_route.reverse()
                    j_route.pop()
                    i_route.extend(j_route)
                    routes_cluster.remove(j_route)
                elif i_index == 0 and j_index == 0:
                    i_route.reverse()
                    j_route.reverse()
                    i_route.extend(j_route)
                    routes_cluster.remove(j_route)
                elif i_index == len(i_route) - 1 and j_index == len(j_route) - 1:
                    j_route.pop()
                    i_route.extend(j_route)
                    routes_cluster.remove(j_route)
                elif i_index == 0 and j_index != len(j_route) - 1:
                    i_route.reverse()
                    j_route.pop(j_index)
                    i_route.extend(j_route)
                elif i_index == len(i_route) - 1 and j_index != 0:
                    j_route.reverse()
                    i_route.pop()
                    j_route.extend(i_route)
                    routes_cluster.remove(i_route)
                    routes_cluster.append(j_route)
                elif j_index == 0 and i_index != len(i_route) - 1:
                    j_route.reverse()
                    i_route.pop(i_index)
                    j_route.extend(i_route)
                elif j_index == len(j_route) - 1 and i_index != 0:
                    i_route.reverse()
                    j_route.pop()
                    i_route.extend(j_route)
                    routes_cluster.remove(j_route)
                    routes_cluster.append(i_route)

                remaining_capacity[i] += remaining_capacity[j]
                remaining_capacity[j] = 0

        for route in routes_cluster:
            route_load = sum([cluster[i][2] for i in route[1:-1]])
            route.insert(1, route_load)
            routes.append(route)

    return routes



def solve_vrp():
    # Separate the latitude, longitude, and load data into separate arrays

    # load data from the file
    data = np.loadtxt("Route_Small_Data_1.txt", delimiter=",")

    # extract latitudes, longitudes and loads from the data
    latitudes = data[:, 0]
    longitudes = data[:, 1]
    loads = data[:, 2]

    # set the number of clusters you want to form
    n_clusters = 3

    # form the coordinates array
    coordinates = np.column_stack((latitudes, longitudes))

    # Use k-means clustering to form clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coordinates)
    labels = kmeans.labels_

    # Group the data points into clusters based on the k-means labels
    clusters = [[] for _ in range(n_clusters)]
    for i, label in enumerate(labels):
        clusters[label].append((latitudes[i], longitudes[i], loads[i]))

    # print the clusters
    for i in range(n_clusters):
        print("Cluster ", i+1, ": ", clusters[i])


    routes = solve_cvrp(clusters,capacity)
        # routes.append(route)

    return routes

routes = solve_vrp()
for i, route in enumerate(routes):
    print(f'Route {i+1}: {route}')
