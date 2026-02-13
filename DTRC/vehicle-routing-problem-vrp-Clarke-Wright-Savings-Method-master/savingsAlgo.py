import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time

# Function to read the file and extract relevant data
def read_cvrp_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Variables to store extracted data
    node_coords = []
    demands = {}
    num_trucks = None
    capacity = None
    section = None

    for line in lines:
        parts = line.strip().split()
        
        if not parts:
            continue

        if parts[0] == "COMMENT" and "No of trucks" in line:
            # Extract number of trucks
            truck_info = line.split("No of trucks:")[-1].split(",")[0].strip()
            num_trucks = int(truck_info)

        elif parts[0] == "CAPACITY":
              capacity = int(parts[-1])

        elif parts[0] == "NODE_COORD_SECTION":
            section = "NODE_COORD"

        elif parts[0] == "DEMAND_SECTION":
            section = "DEMAND"

        elif parts[0] == "DEPOT_SECTION":
            section = "DEPOT"

        elif section == "NODE_COORD":
            node_coords.append((int(parts[0]), float(parts[1]), float(parts[2])))

        elif section == "DEMAND":
            demands[int(parts[0])] = int(parts[1])

    # Convert to NumPy arrays
    node_coords = np.array(node_coords)
    
    # Extracting x and y coordinates
    x_coords = node_coords[:, 1]
    y_coords = node_coords[:, 2]

    return num_trucks, capacity, x_coords, y_coords, demands


# Function to calculate the distance matrix
def distance_matrix_from_xy(x_coordinates, y_coordinates):
    n = len(x_coordinates)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            dist = math.sqrt((x_coordinates[i] - x_coordinates[j])**2 + (y_coordinates[i] - y_coordinates[j])**2)
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist

    return pd.DataFrame(dist_matrix)


# Function to apply Clarke-Wright Savings Algorithm
def clarke_wright_savings(num_trucks, capacity, x_coords, y_coords, demands):
    n_patients = len(demands) - 1  # Exclude depot
    N = np.arange(1, n_patients+1)
    V = np.arange(0, n_patients+1)

    # Distance matrix
    distances_df = distance_matrix_from_xy(x_coords, y_coords)

    # Convert demand dictionary to DataFrame
    nodes = pd.DataFrame({'d0': distances_df.iloc[:, 0], 'demand': [demands[i + 1] for i in range(len(demands))]})

    # Pairwise distance matrix excluding depot
    pw = distances_df.iloc[1:, 1:]

    # Calculate savings
    savings = {}
    for r in pw.index:
        for c in pw.columns:
            if int(c) != int(r):            
                a = max(int(r), int(c))
                b = min(int(r), int(c))
                key = f'({a},{b})'
                savings[key] = nodes['d0'][int(r)] + nodes['d0'][int(c)] - pw[c][r]

    # Sort savings in descending order
    sv = pd.DataFrame.from_dict(savings, orient='index', columns=['saving']).sort_values(by='saving', ascending=False)

    # Clarke-Wright Savings Heuristic
    routes = []
    node_list = list(nodes.index)
    node_list.remove(0)

    def get_node(link):
        return [int(n) for n in link.strip("()").split(",")]

    def sum_cap(route):
        return sum(nodes.demand[node] for node in route)

    def merge(route0, route1, link):
        if route0.index(link[0]) != (len(route0) - 1):
            route0.reverse()
        if route1.index(link[1]) != 0:
            route1.reverse()
        return route0 + route1

    start_time = time.time()
    
    for link in sv.index:
        link = get_node(link)
        n1, n2 = link

        # Find which route (if any) the nodes belong to
        route_indices = [-1, -1]
        for i, route in enumerate(routes):
            if n1 in route:
                route_indices[0] = i
            if n2 in route:
                route_indices[1] = i

        if route_indices[0] == route_indices[1] == -1:  # New route
            if sum_cap(link) <= capacity:
                routes.append(link)
                node_list.remove(n1)
                node_list.remove(n2)
        
        elif route_indices[0] != -1 and route_indices[1] == -1:  # Extend existing route
            route_idx = route_indices[0]
            if sum_cap(routes[route_idx] + [n2]) <= capacity:
                routes[route_idx].append(n2)
                node_list.remove(n2)
        
        elif route_indices[0] == -1 and route_indices[1] != -1:  # Extend existing route
            route_idx = route_indices[1]
            if sum_cap(routes[route_idx] + [n1]) <= capacity:
                routes[route_idx].append(n1)
                node_list.remove(n1)
        
        elif route_indices[0] != -1 and route_indices[1] != -1 and route_indices[0] != route_indices[1]:  # Merge routes
            if sum_cap(routes[route_indices[0]] + routes[route_indices[1]]) <= capacity:
                merged_route = merge(routes[route_indices[0]], routes[route_indices[1]], link)
                route_1 = routes[route_indices[0]]
                route_2 = routes[route_indices[1]]
                routes.remove(route_1)
                routes.remove(route_2)
                routes.append(merged_route)

    # Assign remaining nodes
    for node in node_list:
        routes.append([node])

    # Add depot at start and end of each route
    for route in routes:
        route.insert(0, 0)
        route.append(0)

    end_time = time.time()
    
    # Calculate total distance
    total_distance = 0
    for route in routes:
        route_distance = 0
        for k in range(len(route)-1):
            i, j = route[k], route[k+1]
            route_distance += distances_df.iloc[i, j]
        total_distance += route_distance

    # Print results
    print("\n------")
    print(f"Routes found ({num_trucks} trucks):")
    for route in routes:
        print(route)
    print(f"Total Distance: {total_distance:.2f}")
    print(f"Execution Time: {end_time - start_time:.4f} seconds")

    return routes, total_distance


# **Main Execution**
if __name__ == "__main__":
    file_path = "A-n32-k5.vrp"  # Change this to your actual file path

    # Read file and extract data
    num_trucks, capacity, x_coords, y_coords, demands = read_cvrp_file(file_path)

    # Apply Clarke-Wright Savings Algorithm
    routes, total_distance = clarke_wright_savings(num_trucks, capacity, x_coords, y_coords, demands)

    # Plot results
    plt.figure(figsize=(10, 6))
    for route in routes:
        for k in range(len(route)-1):
            i, j = route[k], route[k+1]
            plt.plot([x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]], 'g-')

    plt.scatter(x_coords[1:], y_coords[1:], c='b', label="Patients")
    plt.scatter(x_coords[0], y_coords[0], c='r', marker='^', label="Depot")
    plt.title('Clarke-Wright Savings Algorithm')
    plt.legend()
    plt.show()
