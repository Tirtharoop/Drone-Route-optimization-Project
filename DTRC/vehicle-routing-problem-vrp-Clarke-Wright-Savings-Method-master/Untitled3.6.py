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
        routes = [
        two_opt(route, distances_df),
        simple_relocate(route, distances_df),
        swap_move(route, distances_df)]
    
        # Select the route with the minimum cost
        best_route = min(routes, key=lambda r: calculate_route_cost(r, distances_df))

        optimized_routes.append(best_route)

    #print(optimized_routes)
    return optimized_routes




def apply_dtrc(truck_routes, drone_nodes, distances_df, demands):
    # Each truck now has its own pair of drones
    all_drone_routes = []
    used_nodes = set()  # Track nodes assigned to drones
    modified_truck_routes = [route.copy() for route in truck_routes]
    
    # Process each truck route separately
    for route_idx, truck_route in enumerate(truck_routes):
        # Initialize two drones for this specific truck
        truck_drone_routes = [[] for _ in range(2)]
        drone_idx = 0  # Alternate between the two drones for this truck
        
        # Consider all nodes in the truck route as potential takeoff points
        for takeoff_idx in range(1, len(truck_route)-1):
            takeoff_node = truck_route[takeoff_idx]
            
            # Skip if this node is already used by a drone
            if takeoff_node in used_nodes:
                continue
                
            # Find potential delivery nodes for this takeoff node
            # Only consider nodes that are in this truck's route
            potential_deliveries = []
            truck_node_set = set(truck_route)
            
            for delivery_idx in range(1, len(truck_route)-1):
                delivery_node = truck_route[delivery_idx]
                
                # Skip if it's the takeoff node or already used
                if delivery_node == takeoff_node or delivery_node in used_nodes:
                    continue
                
                # Only consider nodes that appear AFTER the takeoff node in the route
                if truck_route.index(delivery_node) > takeoff_idx:
                    # Get weight (handle different indexing in demands dictionary)
                    weight = demands.get(delivery_node+1, demands.get(delivery_node, 0))
                    
                    # Only include if within drone capacity
                    if weight <= 35:
                        D = distances_df.iloc[takeoff_node, delivery_node]
                        ratio = D / max(1, weight)  # Avoid division by zero
                        potential_deliveries.append((delivery_node, D, weight, ratio))
            
            # Sort potential deliveries by efficiency ratio (most efficient first)
            potential_deliveries.sort(key=lambda x: x[3])
            
            # Build multi-delivery route with capacity constraint
            if potential_deliveries:
                delivery_route = []
                current_load = 0
                delivery_indices = []  # Track indices for ordering
                
                for delivery_node, distance, weight, ratio in potential_deliveries:
                    # Check if adding this delivery exceeds drone capacity
                    if current_load + weight <= 35:  # 35 unit capacity constraint
                        delivery_route.append(delivery_node)
                        delivery_indices.append(truck_route.index(delivery_node))
                        current_load += weight
                        
                # If we have deliveries, find a landing node on the SAME truck route
                if delivery_route:
                    # Sort deliveries based on their order in the truck route
                    # This ensures drone visits nodes in the same order as the truck would
                    delivery_with_idx = list(zip(delivery_route, delivery_indices))
                    delivery_with_idx.sort(key=lambda x: x[1])  # Sort by truck route order
                    delivery_route = [node for node, _ in delivery_with_idx]
                    
                    # Find landing node that comes AFTER all deliveries in the route
                    last_delivery_idx = max(delivery_indices)
                    potential_landing_nodes = []
                    
                    for landing_idx in range(last_delivery_idx+1, len(truck_route)-1):
                        landing_node = truck_route[landing_idx]
                        if landing_node not in used_nodes and landing_node != takeoff_node and landing_node not in delivery_route:
                            landing_distance = distances_df.iloc[delivery_route[-1], landing_node]
                            potential_landing_nodes.append((landing_node, landing_distance))
                    
                    landing_node = None
                    if potential_landing_nodes:
                        # Choose landing node that minimizes distance from last delivery
                        landing_node, _ = min(potential_landing_nodes, key=lambda x: x[1])
                    else:
                        # If no suitable landing node found, use the takeoff node (return to same spot)
                        landing_node = takeoff_node
                    
                    if landing_node:
                        # Create complete drone route: takeoff -> [deliveries] -> landing
                        complete_route = [takeoff_node] + delivery_route + [landing_node]
                        
                        # Calculate total payload for this route
                        payload = sum(demands.get(node+1, demands.get(node, 0)) for node in delivery_route)
                        
                        # Calculate maximum allowed distance based on the formula
                        max_distance = 14815 / ((65 + payload) ** 1.5)
                        
                        # Calculate actual route distance
                        route_distance = 0
                        for i in range(len(complete_route) - 1):
                            route_distance += distances_df.iloc[complete_route[i], complete_route[i+1]]
                        
                        # Only add this route if it's within the maximum allowed distance
                        if route_distance <= max_distance:
                            truck_drone_routes[drone_idx].append(complete_route)
                            
                            # Mark delivery nodes as used
                            for node in delivery_route:
                                used_nodes.add(node)
                                
                            # Alternate between drones for this truck
                            drone_idx = (drone_idx + 1) % 2
                            
                            # Remove delivery nodes from this truck's route
                            for node in delivery_route:
                                if node in modified_truck_routes[route_idx]:
                                    modified_truck_routes[route_idx].remove(node)
        
        # Add this truck's drone routes to the overall collection
        all_drone_routes.append(truck_drone_routes)
    
    return all_drone_routes, modified_truck_routes


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

def plot_combined_routes(truck_routes, all_drone_routes, x_coords, y_coords, drone_nodes):
    plt.figure(figsize=(12, 8))
    
    # Colors for different truck routes
    truck_colors = ['blue', 'green', 'purple', 'brown', 'orange']
    # Colors for drone routes (matching their trucks but lighter)
    drone_colors = ['red', 'lime', 'magenta', 'chocolate', 'gold']
    
    # Plot truck routes with arrows
    for idx, route in enumerate(truck_routes):
        truck_color = truck_colors[idx % len(truck_colors)]
        route_nodes = []
        
        # Collect route nodes for plotting
        for i in range(len(route) - 1):
            plt.arrow(x_coords[route[i]], y_coords[route[i]], 
                      x_coords[route[i+1]] - x_coords[route[i]], 
                      y_coords[route[i+1]] - y_coords[route[i]], 
                      head_width=0.5, length_includes_head=True, color=truck_color, alpha=0.8)
            route_nodes.append(route[i])
        route_nodes.append(route[-1])  # Add last node
        
        # Plot truck stops
        plt.scatter([x_coords[node] for node in route_nodes], 
                    [y_coords[node] for node in route_nodes], 
                    color=truck_color, marker='o', alpha=0.7, 
                    label=f"Truck {idx+1} Route" if idx == 0 else f"Truck {idx+1}")
        
        # Label nodes
        for node in route_nodes:
            plt.text(x_coords[node], y_coords[node], str(node), fontsize=8, 
                     verticalalignment='bottom', horizontalalignment='right')

    # Collect all nodes involved in drone operations
    delivery_only_nodes = set()
    takeoff_nodes = set()
    landing_nodes = set()
    
    # Plot drone routes with dashed arrows - each truck's drones
    for truck_idx, truck_drone_routes in enumerate(all_drone_routes):
        drone_color = drone_colors[truck_idx % len(drone_colors)]
        
        for drone_idx, drone_route_list in enumerate(truck_drone_routes):
            for route in drone_route_list:
                takeoff = route[0]
                landing = route[-1]
                delivery_nodes = route[1:-1]  # All nodes between takeoff and landing
                
                # Add to sets for later plotting
                takeoff_nodes.add(takeoff)
                landing_nodes.add(landing)
                delivery_only_nodes.update(delivery_nodes)
                
                # Draw arrows for drone flight path with multi-point delivery
                prev_node = takeoff
                for next_node in route[1:]:  # Start from first delivery node
                    plt.arrow(x_coords[prev_node], y_coords[prev_node], 
                             x_coords[next_node] - x_coords[prev_node], 
                             y_coords[next_node] - y_coords[prev_node], 
                             head_width=0.5, length_includes_head=True, color=drone_color, 
                             linestyle='dashed', alpha=0.7)
                    prev_node = next_node
                
                # Add a small annotation to show which drone it is
                mid_x = (x_coords[takeoff] + x_coords[landing]) / 2
                mid_y = (y_coords[takeoff] + y_coords[landing]) / 2
                plt.text(mid_x, mid_y, f"T{truck_idx+1}D{drone_idx+1}", fontsize=8, 
                         color=drone_color, weight='bold')
    
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
    plt.title("Combined Truck and Drone Routes (Each Truck with 2 Drones)")
    plt.grid(True)
    plt.show()

def calculate_total_cost(truck_routes, all_drone_routes, distances_df):
    # Calculate full cost (including depot connections)
    total_cost = 0
    truck_arrival_times = {}  # To track when trucks arrive at each node
    
    # Calculate truck routes cost and establish arrival times
    for route_idx, route in enumerate(truck_routes):
        route_cost = calculate_route_cost(route, distances_df)
        total_cost += route_cost
        
        # Calculate arrival times for each node in the truck route
        current_time = 0
        for i in range(len(route) - 1):
            current_time += distances_df.iloc[route[i], route[i+1]]
            truck_arrival_times[(route_idx, route[i+1])] = current_time
    
    print("Only truck cost - ", total_cost)    
    
    # Calculate drone cost - considering multi-delivery routes and takeoff/landing costs
    drone_cost = 0
    waiting_cost = 0
    
    for truck_idx, truck_drone_routes in enumerate(all_drone_routes):
        truck_drone_cost = 0
        truck_waiting_cost = 0
        
        for drone_idx, drone_route_list in enumerate(truck_drone_routes):
            for route in drone_route_list:
                takeoff_node = route[0]
                landing_node = route[-1]
                
                # 1. Add fixed takeoff and landing cost (1 unit each)
                fixed_cost = 2  # 1 for takeoff + 1 for landing
                
                # Calculate drone's flight time
                drone_time = 0
                for i in range(len(route) - 1):
                    # Drones are 1.5x faster than trucks
                    segment_time = distances_df.iloc[route[i], route[i+1]] / 1.5
                    drone_time += segment_time
                
                # Calculate route cost (including fixed costs)
                route_cost = drone_time + fixed_cost
                
                # 2. Calculate waiting cost if truck arrives before drone
                # Get takeoff time (when drone leaves the truck)
                takeoff_time = truck_arrival_times.get((truck_idx, takeoff_node), 0)
                
                # Calculate when drone will arrive at landing node
                drone_arrival_time = takeoff_time + drone_time
                
                # Calculate when truck will arrive at landing node
                truck_arrival_time = truck_arrival_times.get((truck_idx, landing_node), float('inf'))
                
                # If truck arrives before drone, add waiting cost
                if truck_arrival_time < drone_arrival_time:
                    wait_time = (drone_arrival_time/1.5) - truck_arrival_time
                    if wait_time>0:
                      truck_waiting_cost += wait_time
                    else:
                      truck_waiting_cost +=0
                    print(f"Truck {truck_idx+1} waits {wait_time:.2f} units at node {landing_node} for drone {drone_idx+1}")
                
                truck_drone_cost += route_cost
        
        print(f"Truck {truck_idx+1} drone cost - {truck_drone_cost}")
        print(f"Truck {truck_idx+1} waiting cost - {truck_waiting_cost}")
        
        drone_cost += truck_drone_cost
        waiting_cost += truck_waiting_cost
    
    print("Total drone cost - ", drone_cost)
    print("Total waiting cost - ", waiting_cost)
    
    # Add drone and waiting costs to total cost
    total_cost += drone_cost + waiting_cost
    
    # Calculate delivery-only cost (excluding depot connections)
    delivery_cost = 0
    for route in truck_routes:
        if len(route) > 2:  # Only if route has at least one delivery point
            # Sum distances between delivery points only
            for i in range(1, len(route) - 2):  # Start at first delivery, end before last delivery
                delivery_cost += distances_df.iloc[route[i], route[i+1]]
    
    # Add drone delivery costs (these are all part of delivery)
    for truck_drone_routes in all_drone_routes:
        for drone_route_list in truck_drone_routes:
            for route in drone_route_list:
                # Add fixed takeoff/landing costs to delivery cost
                delivery_cost += 2  # 1 for takeoff + 1 for landing
                
                for i in range(len(route) - 1):
                    # Accounting for drone speed
                    delivery_cost += distances_df.iloc[route[i], route[i+1]] / 1.5
    
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
drone_routes, truck_routes = apply_dtrc(truck_routes, drone_nodes, distances_df, demands)

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