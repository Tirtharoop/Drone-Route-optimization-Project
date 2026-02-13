import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def battery_model(total_payload): 
  return (155885 / ((200 + total_payload) ** 1.5))
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


import pandas as pd

def clarke_wright_savings(num_trucks, capacity, x_coords, y_coords, demands):
    distances_df = distance_matrix_from_xy(x_coords, y_coords)

    # Create a dataframe with demand info
    nodes = pd.DataFrame({'d0': distances_df.iloc[:, 0], 
                          'demand': [demands[i + 1] for i in range(len(demands))]})

    # Compute Savings Values (Ensure i < j to avoid duplicate pairs)
    savings = {}
    for i in range(1, len(nodes)):  # Skip depot (0)
        for j in range(i + 1, len(nodes)):
            savings[(i, j)] = distances_df.iloc[i, 0] + distances_df.iloc[j, 0] - distances_df.iloc[i, j]

    # Sort savings in descending order
    sv = sorted(savings.items(), key=lambda x: x[1], reverse=True)

    # Initialize routes
    routes = []
    node_list = set(nodes.index) - {0}  # All nodes except depot

    def sum_cap(route):
        """Calculate total demand of a route."""
        return sum(nodes.demand[node] for node in route)

    def merge(route0, route1):
        """Merge two routes."""
        return route0 + route1

    # Clarke-Wright Savings Algorithm - Route Merging
    for (n1, n2), _ in sv:
        route_indices = [-1, -1]
        for i, route in enumerate(routes):
            if n1 in route:
                route_indices[0] = i
            if n2 in route:
                route_indices[1] = i

        if route_indices[0] == -1 and route_indices[1] == -1:
            if sum_cap([n1, n2]) <= capacity:
                routes.append([n1, n2])
                node_list.discard(n1)
                node_list.discard(n2)

        elif route_indices[0] != -1 and route_indices[1] == -1:
            route_idx = route_indices[0]
            if sum_cap(routes[route_idx] + [n2]) <= capacity:
                routes[route_idx].append(n2)
                node_list.discard(n2)

        elif route_indices[0] == -1 and route_indices[1] != -1:
            route_idx = route_indices[1]
            if sum_cap(routes[route_idx] + [n1]) <= capacity:
                routes[route_idx].append(n1)
                node_list.discard(n1)

        elif route_indices[0] != -1 and route_indices[1] != -1 and route_indices[0] != route_indices[1]:
            if sum_cap(routes[route_indices[0]] + routes[route_indices[1]]) <= capacity:
                merged_route = merge(routes[route_indices[0]], routes[route_indices[1]])
                for index in sorted(route_indices, reverse=True):
                    routes.pop(index)
                routes.append(merged_route)

    # Assign remaining ungrouped nodes to their own route
    for node in node_list:
        routes.append([node])

    # Ensure depot (0) is at the start and end of each route
    optimized_routes = []
    for route in routes:
        route = [0] + route  # Start at depot
        last_node = min(route[1:], key=lambda node: distances_df.loc[node, 0])
        route.remove(last_node)
        route.append(last_node)
        route.append(0)  # End at depot

        # Apply optimizations
        possible_routes = [
            two_opt(route, distances_df),
            simple_relocate(route, distances_df),
            swap_move(route, distances_df)
        ]
        
        # Choose the best optimized route
        best_route = min(possible_routes, key=lambda r: calculate_route_cost(r, distances_df))

        optimized_routes.append(best_route)

    return optimized_routes




def apply_dtrc(truck_routes, drone_nodes, distances_df, demands):
    # Each truck now has its own pair of drones
    all_drone_routes = []
    modified_truck_routes = [route.copy() for route in truck_routes]
    
    # Process each truck route separately
    for route_idx, truck_route in enumerate(truck_routes):
        # Initialize two drones for this specific truck
        truck_drone_routes = [[] for _ in range(2)]
        
        # Track nodes assigned to drones
        used_nodes = set()
        
        # Track the current position and status of each drone
        drone_positions = [0, 0]  # Initial position in the truck route (index)
        
        # Process the truck route from start to end
        i = 1  # Start from the first node after depot
        while i < len(truck_route) - 1:  # Until the last node before returning to depot
            current_node = truck_route[i]
            
            # Try to deploy each drone from this position
            for drone_idx in range(2):
                # Skip if this drone is not at this position in the route
                if drone_positions[drone_idx] > i:
                    continue
                
                # This drone is available at this position
                # Find all possible delivery nodes from this position
                potential_deliveries = []
                
                # Look ahead in the truck route for delivery candidates
                for j in range(i + 1, len(truck_route) - 1):
                    candidate_node = truck_route[j]
                    
                    # Skip if already used by a drone
                    if candidate_node in used_nodes:
                        continue
                    
                    # Get weight (handle different indexing in demands dictionary)
                    weight = demands.get(candidate_node+1, demands.get(candidate_node, 0))
                    
                    # Only include if within drone weight capacity (35 units)
                    if weight <= 35:
                        D = distances_df.iloc[current_node, candidate_node]
                        ratio = D / max(1, weight)  # Avoid division by zero
                        potential_deliveries.append((candidate_node, j, D, weight, ratio))
                
                # Sort potential deliveries by efficiency ratio (most efficient first)
                potential_deliveries.sort(key=lambda x: x[4])
                
                if potential_deliveries:
                    # Try to build a valid drone route
                    best_route = None
                    best_payload = 0
                    best_value = 0  # Value = nodes delivered / distance
                    
                    # Try from 1 up to min(3, len(potential_deliveries)) nodes in the route
                    max_nodes = min(3, len(potential_deliveries))
                    
                    for num_nodes in range(1, max_nodes + 1):
                        for combo in [potential_deliveries[:num_nodes]]:  # Just take the first num_nodes
                            # Extract nodes and calculate total payload
                            delivery_nodes = []
                            indices = []
                            total_payload = 0
                            
                            for node, idx, _, weight, _ in combo:
                                delivery_nodes.append(node)
                                indices.append(idx)
                                total_payload += weight
                            
                            # Check if payload exceeds capacity
                            if total_payload > 35:
                                continue
                                
                            # Calculate max distance based on payload
                            max_distance = battery_model(total_payload)
                            
                            # For each possible landing node (including takeoff node)
                            for landing_idx in range(max(indices) + 1, len(truck_route)):
                                landing_node = truck_route[landing_idx]
                                
                                # Skip if landing node is already used by drones
                                if landing_node in used_nodes and landing_node not in delivery_nodes:
                                    continue
                                    
                                # Build complete route: takeoff -> deliveries -> landing
                                complete_route = [current_node] + delivery_nodes + [landing_node]
                                
                                # Calculate total distance
                                total_distance = 0
                                for k in range(len(complete_route) - 1):
                                    total_distance += distances_df.iloc[complete_route[k], complete_route[k+1]]
                                    
                                # Check if route exceeds max distance
                                if total_distance <= max_distance:
                                    # Calculate value (nodes per distance)
                                    value = len(delivery_nodes) / (total_distance + 0.1)  # Avoid division by zero
                                    
                                    # Update best route if this one is better
                                    if value > best_value:
                                        best_route = complete_route
                                        best_indices = indices + [landing_idx]
                                        best_value = value
                                        best_payload = total_payload
                    
                    # If we found a valid route, use it
                    if best_route:
                        takeoff_node = best_route[0]
                        landing_node = best_route[-1]
                        delivery_nodes = best_route[1:-1]
                        
                        # Add the route to this drone's routes
                        truck_drone_routes[drone_idx].append(best_route)
                        
                        # Mark delivery nodes as used
                        for node in delivery_nodes:
                            used_nodes.add(node)
                            
                            # Remove these nodes from the truck's route
                            if node in modified_truck_routes[route_idx]:
                                modified_truck_routes[route_idx].remove(node)
                        
                        # Update drone position to the landing node index
                        drone_positions[drone_idx] = best_indices[-1]
                        
                        # Skip to the landing node
                        i = best_indices[-1]
                        break
            
            # Move to the next node if no drone was deployed
            i += 1
        
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
                delivery_nodes = route[1:-1]  # Nodes between takeoff and landing
                
                # Calculate payload for this route
                payload = 0
                for node in delivery_nodes:
                    weight = demands.get(node+1, demands.get(node, 0))
                    payload += weight
                
                # Calculate max distance based on payload
                max_distance = battery_model(payload)
                
                # 1. Add fixed takeoff and landing cost (1 unit each)
                fixed_cost = 2  # 1 for takeoff + 1 for landing
                
                # Calculate drone's flight distance and time
                drone_distance = 0
                drone_time = 0
                
                for i in range(len(route) - 1):
                    segment_distance = distances_df.iloc[route[i], route[i+1]]
                    drone_distance += segment_distance
                    
                    # Drones are 1.5x faster than trucks
                    segment_time = segment_distance / 1.5
                    drone_time += segment_time
                
                # Verify distance constraint
                if drone_distance > max_distance:
                    print(f"WARNING: Drone {drone_idx+1} of Truck {truck_idx+1} exceeds max distance!")
                    print(f"  Route: {route}")
                    print(f"  Distance: {drone_distance:.2f}, Max allowed: {max_distance:.2f}")
                    print(f"  Payload: {payload}")
                
                # Get takeoff time (when drone leaves the truck)
                takeoff_time = truck_arrival_times.get((truck_idx, takeoff_node), 0)
                
                # Calculate when drone will arrive at landing node
                drone_arrival_time = takeoff_time + drone_time
                
                # Calculate when truck will arrive at landing node
                truck_arrival_time = truck_arrival_times.get((truck_idx, landing_node), float('inf'))
                
                # If truck arrives before drone, add waiting cost
                if truck_arrival_time < drone_arrival_time:
                    wait_time = drone_arrival_time - truck_arrival_time
                    truck_waiting_cost += wait_time
                    print(f"Truck {truck_idx+1} waits {wait_time:.2f} units at node {landing_node} for drone {drone_idx+1}")
                
                # Calculate route cost (including fixed costs)
                route_cost = drone_time + fixed_cost
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
file_path = "A-n33-k5.vrp"
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