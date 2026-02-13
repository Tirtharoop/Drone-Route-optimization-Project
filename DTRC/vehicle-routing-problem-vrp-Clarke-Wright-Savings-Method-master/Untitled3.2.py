import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Function to read the file and extract relevant data
def read_cvrp_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

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
            num_trucks = int(''.join(filter(str.isdigit, line.split("No of trucks:")[-1])))

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

    node_coords = np.array(node_coords)
    x_coords = node_coords[:, 1]
    y_coords = node_coords[:, 2]

    return num_trucks, capacity, x_coords, y_coords, demands

# Function to calculate distance matrix
def distance_matrix_from_xy(x_coordinates, y_coordinates):
    n = len(x_coordinates)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = math.sqrt((x_coordinates[i] - x_coordinates[j])**2 + (y_coordinates[i] - y_coordinates[j])**2)
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist

    return pd.DataFrame(dist_matrix)

# 2-opt optimization
def two_opt(route, dist_matrix):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                if calculate_route_cost(new_route, dist_matrix) < calculate_route_cost(best, dist_matrix):
                    best = new_route
                    improved = True
        route = best
    return best

# Simple relocate optimization
def simple_relocate(route, dist_matrix):
    best = route
    for i in range(1, len(route) - 1):
        for j in range(1, len(route)):
            if i == j: continue
            new_route = route[:i] + route[i+1:j] + [route[i]] + route[j:]
            if calculate_route_cost(new_route, dist_matrix) < calculate_route_cost(best, dist_matrix):
                best = new_route
    return best

# Swap move optimization
def swap_move(route, dist_matrix):
    best = route
    for i in range(1, len(route) - 1):
        for j in range(i + 1, len(route)):
            new_route = route[:i] + [route[j]] + route[i+1:j] + [route[i]] + route[j+1:]
            if calculate_route_cost(new_route, dist_matrix) < calculate_route_cost(best, dist_matrix):
                best = new_route
    return best

# Calculate route cost
def calculate_route_cost(route, dist_matrix):
    cost = 0
    for i in range(len(route) - 1):
        cost += dist_matrix.iloc[route[i], route[i+1]]
    return cost

# Clarke-Wright Savings Algorithm with optimizations
def clarke_wright_savings(num_trucks, capacity, x_coords, y_coords, demands):
    distances_df = distance_matrix_from_xy(x_coords, y_coords)

    nodes = pd.DataFrame({'d0': distances_df.iloc[:, 0], 'demand': [demands[i + 1] for i in range(len(demands))]})

    savings = {}
    for r in distances_df.index[1:]:
        for c in distances_df.columns[1:]:
            if r != c:
                key = f'({r},{c})'
                savings[key] = distances_df.iloc[r, 0] + distances_df.iloc[c, 0] - distances_df.iloc[r, c]

    sv = pd.DataFrame.from_dict(savings, orient='index', columns=['saving']).sort_values(by='saving', ascending=False)

    routes = []
    node_list = list(nodes.index)
    node_list.remove(0)

    def sum_cap(route):
        return sum(nodes.demand[node] for node in route)

    def merge(route0, route1):
        return route0 + route1

    for link in sv.index:
        n1, n2 = [int(n) for n in link.strip("()").split(",")]

        route_indices = [-1, -1]
        for i, route in enumerate(routes):
            if n1 in route:
                route_indices[0] = i
            if n2 in route:
                route_indices[1] = i

        if route_indices[0] == route_indices[1] == -1:
            if sum_cap([n1, n2]) <= capacity:
                routes.append([n1, n2])
                node_list.remove(n1)
                node_list.remove(n2)

        elif route_indices[0] != -1 and route_indices[1] == -1:
            route_idx = route_indices[0]
            if sum_cap(routes[route_idx] + [n2]) <= capacity:
                routes[route_idx].append(n2)
                node_list.remove(n2)

        elif route_indices[0] == -1 and route_indices[1] != -1:
            route_idx = route_indices[1]
            if sum_cap(routes[route_idx] + [n1]) <= capacity:
                routes[route_idx].append(n1)
                node_list.remove(n1)

        elif route_indices[0] != -1 and route_indices[1] != -1 and route_indices[0] != route_indices[1]:
            if sum_cap(routes[route_indices[0]] + routes[route_indices[1]]) <= capacity:
                merged_route = merge(routes[route_indices[0]], routes[route_indices[1]])
                for index in sorted(route_indices, reverse=True):
                    routes.pop(index)
                routes.append(merged_route)

    for node in node_list:
        routes.append([node])

    for route in routes:
        route.insert(0, 0)
        route.append(0)

    # Apply 2-opt, simple relocate, and swap move optimizations
    optimized_routes = []
    for route in routes:
        optimized_route = two_opt(route, distances_df)
        optimized_route = simple_relocate(optimized_route, distances_df)
        optimized_route = swap_move(optimized_route, distances_df)
        optimized_routes.append(optimized_route)

    return optimized_routes

# Apply DTRC algorithm with two drones
def apply_dtrc(truck_routes, drone_nodes, distances_df, demands):
    drone_routes = [[] for _ in range(2)]  # Two drones
    used_nodes = set()  # Track nodes assigned to drones

    drone_idx = 0
    for truck_route in truck_routes:
        possible_takeoff_nodes = truck_route[1:-1]  # Exclude depot

        for node in possible_takeoff_nodes:
            if node in drone_nodes and node not in used_nodes:
                # Select delivery node using the lowest ratio of ϑ = D / W
                delivery_node = None
                min_ratio = float('inf')
                for candidate in drone_nodes[node]:
                    if candidate not in used_nodes:
                        D = distances_df.iloc[node, candidate]
                        W = demands[candidate]
                        ratio = D / W
                        if ratio < min_ratio:
                            min_ratio = ratio
                            delivery_node = candidate

                if delivery_node:
                    # Ensure the landing node is a truck delivery node
                    landing_node = None
                    for truck_route in truck_routes:
                        if delivery_node in truck_route:
                            landing_node = delivery_node
                            break

                    if landing_node:
                        drone_routes[drone_idx].append([node] + [delivery_node] + [landing_node])  # Assign round trip to a drone
                        used_nodes.update([node, delivery_node])  # Mark nodes as used
                        drone_idx = (drone_idx + 1) % 2  # Alternate between the two drones

    return drone_routes

# Plot Truck Routes (after Clarke-Wright Savings)
def plot_truck_routes(truck_routes, x_coords, y_coords):
    plt.figure(figsize=(10, 6))

    # Plot truck routes
    for route in truck_routes:
        x_points = [x_coords[node] for node in route]
        y_points = [y_coords[node] for node in route]
        plt.plot(x_points, y_points, marker='o', linestyle='-', color='blue', alpha=0.8, label="Truck Route" if route == truck_routes[0] else None)

    # Plot depot
    plt.scatter(x_coords[0], y_coords[0], color='black', marker='s', s=100, label="Depot")

    plt.legend()
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Optimized Truck Routes (Clarke-Wright Savings)")
    plt.grid()
    plt.show()

# Plot Combined Truck and Drone Routes
def plot_combined_routes(truck_routes, drone_routes, x_coords, y_coords, drone_nodes):
    plt.figure(figsize=(10, 6))

    # Plot truck routes
    for route in truck_routes:
        x_points = [x_coords[node] for node in route if node not in drone_nodes]
        y_points = [y_coords[node] for node in route if node not in drone_nodes]
        plt.plot(x_points, y_points, marker='o', linestyle='-', color='blue', alpha=0.8, label="Truck Route" if route == truck_routes[0] else None)

    # Plot drone routes
    for route_group in drone_routes:
        for route in route_group:
            x_points = [x_coords[node] for node in route]
            y_points = [y_coords[node] for node in route]
            plt.plot(x_points, y_points, marker='x', linestyle='--', color='red', alpha=0.7, label="Drone Route" if route == route_group[0] else None)

    # Plot depot
    plt.scatter(x_coords[0], y_coords[0], color='black', marker='s', s=100, label="Depot")

    plt.legend()
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Combined Truck and Drone Routes")
    plt.grid()
    plt.show()

# Calculate total cost
def calculate_total_cost(truck_routes, drone_routes, distances_df):
    total_cost = 0

    # Calculate truck route costs
    for route in truck_routes:
        total_cost += calculate_route_cost(route, distances_df)

    # Calculate drone route costs
    for route_group in drone_routes:
        for route in route_group:
            total_cost += calculate_route_cost(route, distances_df)

    return total_cost

# Main Execution
file_path = "A-n32-k5.vrp"
num_trucks, capacity, x_coords, y_coords, demands = read_cvrp_file(file_path)
truck_routes = clarke_wright_savings(num_trucks, capacity, x_coords, y_coords, demands)

# Plot truck routes after Clarke-Wright Savings
plot_truck_routes(truck_routes, x_coords, y_coords)

# Apply DTRC algorithm
distances_df = distance_matrix_from_xy(x_coords, y_coords)
drone_nodes = {1: [12, 15], 4: [7, 11]}
drone_routes = apply_dtrc(truck_routes, drone_nodes, distances_df, demands)

# Plot combined truck and drone routes
plot_combined_routes(truck_routes, drone_routes, x_coords, y_coords, drone_nodes)

# Print total cost
total_cost = calculate_total_cost(truck_routes, drone_routes, distances_df)
print(f"Total Cost: {total_cost}")