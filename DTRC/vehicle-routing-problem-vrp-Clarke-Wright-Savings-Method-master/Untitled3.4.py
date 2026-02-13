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
    cost += dist_matrix.iloc[route[-1], 0]     
    return cost


# Clarke-Wright Savings Algorithm with return-cost optimization
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

    # Ensure routes start and end at depot (0) and consider return cost
    optimized_routes = []
    for route in routes:
        route = [0] + route  # Ensure depot at start

        # Choose the best last node by minimizing return cost
        last_node = min(route[1:], key=lambda node: distances_df.iloc[node, 0])
        route.remove(last_node)
        route.append(last_node)
        route.append(0)  # Append depot

        # Apply optimizations
        optimized_route = two_opt(route, distances_df)
        optimized_route = simple_relocate(optimized_route, distances_df)
        optimized_route = swap_move(optimized_route, distances_df)

        optimized_routes.append(optimized_route)

    #print(optimized_routes)
    return optimized_routes




def apply_dtrc(truck_routes, drone_nodes, distances_df, demands):
    drone_routes = [[] for _ in range(2)]  # Two drones
    used_nodes = set()  # Track nodes assigned to drones
    drone_idx = 0
    
    # Deep copy of truck routes to modify
    modified_truck_routes = [route.copy() for route in truck_routes]
    
    for route_idx, truck_route in enumerate(truck_routes):
        possible_takeoff_nodes = [node for node in truck_route[1:-1] if node in drone_nodes]
        
        for takeoff_node in possible_takeoff_nodes:
            if takeoff_node in used_nodes:
                continue
                
            # Find best delivery node for this takeoff node
            delivery_node = None
            min_ratio = float('inf')
            
            for candidate in drone_nodes.get(takeoff_node, []):
                if candidate not in used_nodes:
                    D = distances_df.iloc[takeoff_node, candidate]
                    W = demands[candidate + 1] if candidate + 1 in demands else demands[candidate]  # Handle indexing
                    ratio = D / W
                    
                    if ratio < min_ratio:
                        min_ratio = ratio
                        delivery_node = candidate
            
            if delivery_node:
                # Find a valid landing node on ANY truck route (could be different truck)
                landing_node = None
                min_landing_distance = float('inf')
                landing_route_idx = -1
                
                # Check all truck routes for potential landing nodes
                for r_idx, r in enumerate(truck_routes):
                    for node in r[1:-1]:  # Exclude depot
                        if node not in used_nodes and node != takeoff_node and node != delivery_node:
                            # Calculate drone flight distance from delivery to landing
                            flight_distance = distances_df.iloc[delivery_node, node]
                            
                            if flight_distance < min_landing_distance:
                                min_landing_distance = flight_distance
                                landing_node = node
                                landing_route_idx = r_idx
                
                if landing_node:
                    # Add drone route: takeoff -> delivery -> landing
                    drone_routes[drone_idx].append([takeoff_node, delivery_node, landing_node])
                    
                    # Mark delivery node as used (but not takeoff/landing as trucks still visit these)
                    used_nodes.add(delivery_node)
                    
                    # Remove delivery node from truck routes
                    for idx, route in enumerate(modified_truck_routes):
                        if delivery_node in route:
                            modified_truck_routes[idx].remove(delivery_node)
                    
                    # Ensure proper sequencing in truck routes
                    # For landing route, ensure landing node is visited after the drone delivery time
                    # For simplicity, we're assuming the truck reaches the landing node
                    # after the drone completes its delivery
                    
                    drone_idx = (drone_idx + 1) % 2  # Alternate between two drones
    
    return drone_routes, modified_truck_routes



# Plot Truck Routes (after Clarke-Wright Savings)
def plot_truck_routes(truck_routes, x_coords, y_coords):
    plt.figure(figsize=(10, 6))

    # Plot truck routes with arrows
    for idx, route in enumerate(truck_routes):
        for i in range(len(route) - 1):
            plt.arrow(x_coords[route[i]], y_coords[route[i]], 
                      x_coords[route[i+1]] - x_coords[route[i]], 
                      y_coords[route[i+1]] - y_coords[route[i]], 
                      head_width=0.5, length_includes_head=True, color='blue', alpha=0.8)
        
        # Plot truck stops
        plt.scatter([x_coords[node] for node in route], 
                    [y_coords[node] for node in route], 
                    color='blue', marker='o', alpha=0.7, label="Truck Route" if idx == 0 else None)

    # Plot depot
    plt.scatter(x_coords[0], y_coords[0], color='black', marker='s', s=100, label="Depot")

    plt.legend()
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Optimized Truck Routes (Clarke-Wright Savings)")
    plt.grid()
    plt.show()

def plot_combined_routes(truck_routes, drone_routes, x_coords, y_coords, drone_nodes):
    plt.figure(figsize=(10, 6))
    
    # Plot truck routes with arrows
    for idx, route in enumerate(truck_routes):
        route_nodes = []
        # Collect route nodes for plotting
        for i in range(len(route) - 1):
            plt.arrow(x_coords[route[i]], y_coords[route[i]], 
                      x_coords[route[i+1]] - x_coords[route[i]], 
                      y_coords[route[i+1]] - y_coords[route[i]], 
                      head_width=0.5, length_includes_head=True, color='blue', alpha=0.8)
            route_nodes.append(route[i])
        route_nodes.append(route[-1])  # Add last node
        
        # Plot truck stops
        plt.scatter([x_coords[node] for node in route_nodes], 
                    [y_coords[node] for node in route_nodes], 
                    color='blue', marker='o', alpha=0.7, label="Truck Route" if idx == 0 else None)
        
        # Label nodes
        for node in route_nodes:
            plt.text(x_coords[node], y_coords[node], str(node), fontsize=8, 
                     verticalalignment='bottom', horizontalalignment='right')

    # Collect all nodes involved in drone operations
    drone_operation_nodes = set()
    delivery_only_nodes = set()
    takeoff_nodes = set()
    landing_nodes = set()
    
    # Plot drone routes with dashed arrows
    for idx, route_group in enumerate(drone_routes):
        for route in route_group:
            takeoff, delivery, landing = route[0], route[1], route[2]
            
            # Add to sets for later plotting
            takeoff_nodes.add(takeoff)
            delivery_only_nodes.add(delivery)
            landing_nodes.add(landing)
            drone_operation_nodes.update([takeoff, delivery, landing])
            
            # Draw arrows for drone flight path
            # Takeoff -> Delivery
            plt.arrow(x_coords[takeoff], y_coords[takeoff], 
                     x_coords[delivery] - x_coords[takeoff], 
                     y_coords[delivery] - y_coords[takeoff], 
                     head_width=0.5, length_includes_head=True, color='red', linestyle='dashed', alpha=0.7)
            
            # Delivery -> Landing
            plt.arrow(x_coords[delivery], y_coords[delivery], 
                     x_coords[landing] - x_coords[delivery], 
                     y_coords[landing] - y_coords[delivery], 
                     head_width=0.5, length_includes_head=True, color='red', linestyle='dashed', alpha=0.7)
    
    # Plot special nodes with distinct markers
    # Takeoff nodes
    if takeoff_nodes:
        plt.scatter([x_coords[node] for node in takeoff_nodes], 
                    [y_coords[node] for node in takeoff_nodes], 
                    color='green', marker='^', s=100, label="Takeoff Points")
    
    # Landing nodes
    if landing_nodes:
        plt.scatter([x_coords[node] for node in landing_nodes], 
                    [y_coords[node] for node in landing_nodes], 
                    color='purple', marker='v', s=100, label="Landing Points")
    
    # Delivery-only nodes (nodes only visited by drones)
    if delivery_only_nodes:
        plt.scatter([x_coords[node] for node in delivery_only_nodes], 
                    [y_coords[node] for node in delivery_only_nodes], 
                    color='red', marker='x', s=80, label="Drone Delivery Points")
        
        # Label delivery nodes
        for node in delivery_only_nodes:
            plt.text(x_coords[node], y_coords[node], str(node), fontsize=8, 
                     verticalalignment='top', horizontalalignment='left', color='red')

    # Plot depot
    plt.scatter(x_coords[0], y_coords[0], color='black', marker='s', s=100, label="Depot")
    plt.text(x_coords[0], y_coords[0], "0", fontsize=10, color='white',
             verticalalignment='center', horizontalalignment='center')

    plt.legend()
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Combined Truck and Drone Routes")
    plt.grid(True)
    plt.show()

def calculate_total_cost(truck_routes, drone_routes, distances_df):
    # Calculate full cost (including depot connections)
    total_cost = 0
    for route in truck_routes:
        total_cost += calculate_route_cost(route, distances_df)
    print("Only truck cost - ",total_cost)    
    
    for route_group in drone_routes:
        for route in route_group:
            # For drone routes, add the cost of takeoff->delivery->landing
            drone_cost = 0
            for i in range(len(route) - 1):
                drone_cost += distances_df.iloc[route[i], route[i+1]]
            total_cost += drone_cost
    
    # Calculate delivery-only cost (excluding depot connections)
    delivery_cost = 0
    for route in truck_routes:
        if len(route) > 2:  # Only if route has at least one delivery point
            # Sum distances between delivery points only
            for i in range(1, len(route) - 2):  # Start at first delivery, end before last delivery
                delivery_cost += distances_df.iloc[route[i], route[i+1]]
    
    # Add drone costs to delivery cost (these are all part of delivery)
    for route_group in drone_routes:
        for route in route_group:
            for i in range(len(route) - 1):
                delivery_cost += distances_df.iloc[route[i], route[i+1]]
    
    return total_cost, delivery_cost

# Main Execution
file_path = "A-n32-k5.vrp"
num_trucks, capacity, x_coords, y_coords, demands = read_cvrp_file(file_path)
truck_routes = clarke_wright_savings(num_trucks, capacity, x_coords, y_coords, demands)

# Plot truck routes after Clarke-Wright Savings
print("-----Truck routes-----")
print(truck_routes)
plot_truck_routes(truck_routes, x_coords, y_coords)

# Apply DTRC algorithm
distances_df = distance_matrix_from_xy(x_coords, y_coords)
drone_nodes = {1: [12, 15], 4: [7, 11]}
drone_routes,truck_routes = apply_dtrc(truck_routes, drone_nodes, distances_df, demands)

# Plot combined truck and drone routes
print("-----Truck routes-----")
print(truck_routes)
print("-----Drone routes-----")
print(drone_routes)
plot_combined_routes(truck_routes, drone_routes, x_coords, y_coords, drone_nodes)

# Print total cost
total_cost,delivery_cost = calculate_total_cost(truck_routes, drone_routes, distances_df)
print(f"Total Cost: {total_cost}")
print(f"delivery Cost: {delivery_cost}")