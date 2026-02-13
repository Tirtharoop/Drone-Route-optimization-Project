import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
from networkx.algorithms.approximation import traveling_salesman_problem
from networkx.algorithms.approximation import christofides


def dynamic_battery_model(remaining_payload, distance_traveled=0):
    return (155885 / ((200 + remaining_payload) ** 1.5)) - distance_traveled

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

def christofides_route(route, dist_matrix):
    G = nx.Graph()
    
    # Add only the relevant nodes
    for i in route:
        G.add_node(i)

    # Add weighted edges between all pairs in the subroute
    for i in route:
        for j in route:
            if i != j:
                G.add_edge(i, j, weight=dist_matrix.iloc[i, j])

    tsp_path = traveling_salesman_problem(G, cycle=True, method=christofides)

    # Ensure the route starts and ends at depot (node 0)
    if tsp_path[0] != 0:
        zero_index = tsp_path.index(0)
        tsp_path = tsp_path[zero_index:] + tsp_path[1:zero_index] + [0]

    return tsp_path

# Calculate route cost
def calculate_route_cost(route, dist_matrix):
    cost = 0
    for i in range(len(route) - 1):
        cost += dist_matrix.iloc[route[i], route[i+1]]
    cost += dist_matrix.iloc[route[-1], 0]     
    return cost


def i_k_means(num_trucks, capacity, x_coords, y_coords, demands, num_iterations=100):
    coords = np.column_stack((x_coords[1:], y_coords[1:]))
    
    demand_weights = np.array([demands.get(i+1, demands.get(i, 0)) for i in range(len(coords))])
    
    # Normalize coordinates and demands
    scaler = StandardScaler()
    scaled_coords = scaler.fit_transform(coords)
    
    # Perform weighted k-means clustering
    def weighted_kmeans(n_clusters):
        # Custom k-means with demand-based weights
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        
        # Compute cluster centers with demand-weighted approach
        kmeans.fit(scaled_coords, sample_weight=demand_weights)
        return kmeans
    
    # Perform clustering
    cluster_model = weighted_kmeans(num_trucks)
    
    # Assign cluster labels
    cluster_labels = cluster_model.labels_
    
    # Function to calculate route for a specific cluster
    def construct_cluster_route(cluster_indices):
        # If no nodes in cluster, return depot route
        if len(cluster_indices) == 0:
            return [0, 0]
        
        # Extract cluster coordinates and demands
        cluster_coords_x = x_coords[1:][cluster_indices]
        cluster_coords_y = y_coords[1:][cluster_indices]
        cluster_node_indices = np.where(np.isin(range(1, len(x_coords)), cluster_indices + 1))[0] + 1
        
        # Initialize route
        route = [0]  # Start at depot
        current_node = 0
        current_load = 0
        
        # Track visited nodes
        unvisited = set(cluster_node_indices)
        
        while unvisited:
            # Find valid next nodes (respecting capacity)
            valid_nodes = [
                node for node in unvisited 
                if current_load + demands.get(node, 0) <= capacity
            ]
            
            # If no valid nodes, allow node selection with slight capacity excess
            if not valid_nodes:
                valid_nodes = list(unvisited)  # Allow visiting any remaining node
            
            # Calculate distances and select best next node
            node_scores = []
            for node in valid_nodes:
                # Node index adjustment
                node_idx = node - 1
                
                # Calculate distance
                distance = math.sqrt(
                    (x_coords[current_node] - x_coords[node])**2 + 
                    (y_coords[current_node] - y_coords[node])**2
                )
                
                # Calculate node importance
                demand = demands.get(node, 0)
                importance = demand / (distance + 0.1)
                
                node_scores.append((node, importance))
            
            # Select node with highest score
            if node_scores:
                next_node = max(node_scores, key=lambda x: x[1])[0]
                
                # Update route
                route.append(next_node)
                current_load += demands.get(next_node, 0)
                unvisited.remove(next_node)
                current_node = next_node
            else:
                # Force return to depot if no valid nodes left
                route.append(0)
                break
        
        # Ensure route returns to depot
        if route[-1] != 0:
            route.append(0)
        
        return route
    
    # Construct routes for each cluster
    initial_routes = []
    for i in range(num_trucks):
        # Find nodes in this cluster
        cluster_indices = np.where(cluster_labels == i)[0]
        
        # Construct route for this cluster
        route = construct_cluster_route(cluster_indices)
        initial_routes.append(route)
    
    # Create distance matrix for optimizations
    num_nodes = len(x_coords)
    distances = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            distances[i, j] = math.sqrt((x_coords[i] - x_coords[j])**2 + (y_coords[i] - y_coords[j])**2)
    
    # Convert to pandas DataFrame for compatibility with the optimization functions
    import pandas as pd
    distances_df = pd.DataFrame(distances)
    
    # Apply optimization techniques to each route
    optimized_routes = []
    for route in initial_routes:
        # Skip optimization for routes with just depot-depot
        if len(route) <= 2:
            optimized_routes.append(route)
            continue
            
        # Apply all optimization techniques
        possible_routes = {
            "two_opt": two_opt(route, distances_df),
            "simple_relocate": simple_relocate(route, distances_df),
            "swap_move": swap_move(route, distances_df),
            "christofides_route": christofides_route(route, distances_df)
        }
        
        best_function, best_route = min(possible_routes.items(),
            key=lambda item: calculate_route_cost(item[1], distances_df))
        
        print(f"The best route is given by: {best_function}")
        optimized_routes.append(best_route)
    
    return optimized_routes


def apply_dtrc(truck_routes, distances_df, demands, d_capacity, x_coords, y_coords):
    import math
    from itertools import combinations
    
    # Drone capacity limit
    DRONE_CAPACITY = d_capacity
    
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
        i = 0  # Start from depot
        while i < len(truck_route) - 1:  # Until the last node before returning to depot
            current_node = truck_route[i]
            next_node = truck_route[i+1]
            best_route = None
            best_landing_idx = None
            best_drone = None
            
            # Use a limited set of takeoff options to reduce computation
            # Try current node first, then a single virtual point in the middle of segment
            takeoff_options = [
                (current_node, i, False),  # Actual node
                (f"V({(x_coords[current_node] + x_coords[next_node])/2:.1f},{(y_coords[current_node] + y_coords[next_node])/2:.1f})", i + 0.5, True)  # Virtual middle point
            ]
            
            # Try each takeoff option with both drones
            for takeoff_point, takeoff_idx, is_virtual_takeoff in takeoff_options:
                for drone_idx in range(2):
                    # Skip if this drone is not available at this position
                    if drone_positions[drone_idx] > takeoff_idx:
                        continue
                    
                    # This drone is available at this position
                    potential_deliveries = []
                    
                    # Look ahead in the truck route for limited number of delivery candidates
                    # Limit the search range to improve performance
                    max_search = min(i + 8, len(truck_route) - 1)  # Only look 8 nodes ahead max
                    
                    for j in range(i + 1, max_search):
                        candidate_node = truck_route[j]
                        
                        # Skip if already used by a drone
                        if candidate_node in used_nodes:
                            continue
                        
                        # Get weight
                        weight = demands.get(candidate_node, demands.get(candidate_node+1, 0))
                        
                        # Only include if within drone weight capacity
                        if weight <= DRONE_CAPACITY:
                            # Calculate distance from takeoff point to candidate node
                            if is_virtual_takeoff:
                                # Extract coordinates from virtual takeoff
                                coords = takeoff_point.strip('V()').split(',')
                                from_x, from_y = float(coords[0]), float(coords[1])
                                
                                # Calculate Euclidean distance from virtual point to candidate node
                                D = math.sqrt((from_x - x_coords[candidate_node])**2 + 
                                            (from_y - y_coords[candidate_node])**2)
                            else:
                                # Normal node-to-node distance
                                D = distances_df.iloc[takeoff_point, candidate_node]
                                
                            potential_deliveries.append((candidate_node, j, D, weight, D / max(1, weight)))
                    
                    # Sort potential deliveries by position in truck route (to maintain order)
                    potential_deliveries.sort(key=lambda x: x[1])
                    
                    if potential_deliveries:
                        # Try to build a valid drone route
                        best_value_for_this_drone = 0
                        
                        # Limit to at most 2 delivery nodes to reduce complexity
                        max_nodes = min(2, len(potential_deliveries))
                        
                        for num_nodes in range(1, max_nodes + 1):
                            # Limit the number of combinations to try
                            max_candidates = min(4, len(potential_deliveries))
                            
                            for combo_indices in combinations(range(max_candidates), min(num_nodes, max_candidates)):
                                combo = [potential_deliveries[idx] for idx in sorted(combo_indices)]
                                
                                # Extract nodes and calculate total payload
                                delivery_nodes = []
                                indices = []
                                total_payload = 0
                                
                                for node, idx, _, weight, _ in combo:
                                    delivery_nodes.append(node)
                                    indices.append(idx)
                                    total_payload += weight
                                
                                # Check if payload exceeds capacity
                                if total_payload > DRONE_CAPACITY:
                                    continue
                                
                                # Find last delivery node's position in truck route
                                last_delivery_idx = max(indices)
                                
                                # For efficiency, try only a few landing points
                                # Choose the node after the last delivery and one virtual point
                                landing_options = []
                                
                                # Include the node after the last delivery
                                if last_delivery_idx + 1 < len(truck_route):
                                    landing_node = truck_route[last_delivery_idx + 1]
                                    landing_options.append((landing_node, last_delivery_idx + 1, False))
                                
                                # Include a virtual landing point if there's a segment after the last delivery
                                if last_delivery_idx + 2 < len(truck_route):
                                    node1 = truck_route[last_delivery_idx + 1]
                                    node2 = truck_route[last_delivery_idx + 2]
                                    
                                    # Create a single virtual landing point mid-segment
                                    virtual_x = (x_coords[node1] + x_coords[node2]) / 2
                                    virtual_y = (y_coords[node1] + y_coords[node2]) / 2
                                    virtual_landing = f"V({virtual_x:.1f},{virtual_y:.1f})"
                                    
                                    landing_options.append((virtual_landing, last_delivery_idx + 1.5, True))
                                
                                # For each landing option
                                for landing_point, landing_idx, is_virtual_landing in landing_options:
                                    # Build route with this sequence
                                    route = [takeoff_point]
                                    route.extend(delivery_nodes)
                                    route.append(landing_point)
                                    
                                    # Calculate route feasibility with dynamic battery model
                                    valid_route = True
                                    remaining_payload = total_payload
                                    battery_consumed = 0
                                    total_dist = 0
                                    
                                    # Check each segment
                                    for idx in range(len(route) - 1):
                                        from_node = route[idx]
                                        to_node = route[idx + 1]
                                        
                                        # Calculate distance for this segment
                                        if isinstance(from_node, str) and from_node.startswith('V('):
                                            # Extract coordinates from virtual takeoff
                                            coords = from_node.strip('V()').split(',')
                                            from_x, from_y = float(coords[0]), float(coords[1])
                                            
                                            if isinstance(to_node, str) and to_node.startswith('V('):
                                                # Virtual-to-virtual
                                                coords = to_node.strip('V()').split(',')
                                                to_x, to_y = float(coords[0]), float(coords[1])
                                                segment_distance = math.sqrt((from_x - to_x)**2 + (from_y - to_y)**2)
                                            else:
                                                # Virtual-to-node
                                                segment_distance = math.sqrt((from_x - x_coords[to_node])**2 + 
                                                                           (from_y - y_coords[to_node])**2)
                                        elif isinstance(to_node, str) and to_node.startswith('V('):
                                            # Node-to-virtual
                                            coords = to_node.strip('V()').split(',')
                                            to_x, to_y = float(coords[0]), float(coords[1])
                                            segment_distance = math.sqrt((x_coords[from_node] - to_x)**2 + 
                                                                       (y_coords[from_node] - to_y)**2)
                                        else:
                                            # Normal node-to-node
                                            segment_distance = distances_df.iloc[from_node, to_node]
                                        
                                        # Check battery with current payload
                                        max_distance = dynamic_battery_model(remaining_payload, battery_consumed)
                                        if segment_distance > max_distance:
                                            valid_route = False
                                            break
                                        
                                        # Update battery consumed and distance
                                        battery_consumed += segment_distance
                                        total_dist += segment_distance
                                        
                                        # Update payload if delivered
                                        if to_node in delivery_nodes:
                                            node_weight = demands.get(to_node, demands.get(to_node+1, 0))
                                            remaining_payload -= node_weight
                                    
                                    if valid_route:
                                        # Calculate value (nodes per distance)
                                        value = len(delivery_nodes) / (total_dist + 0.1)
                                        
                                        if value > best_value_for_this_drone:
                                            best_value_for_this_drone = value
                                            best_route = route
                                            best_landing_idx = landing_idx
                                            best_drone = drone_idx
                        
                        # If we found a valid route, use it
                        if best_route:
                            break  # Found a route with this drone, no need to try other takeoff points
                
                # If we found a route for any drone from this takeoff point, break out of takeoff loop
                if best_route:
                    break
            
            # If we found a valid route, use it
            if best_route:
                # Add the route to this drone's routes
                truck_drone_routes[best_drone].append(best_route)
                
                # Mark delivery nodes as used
                for node in best_route[1:-1]:
                    if not isinstance(node, str):  # Skip virtual nodes
                        used_nodes.add(node)
                
                # Update drone position to the landing index
                drone_positions[best_drone] = best_landing_idx
                
                # Skip to the landing node
                i = int(best_landing_idx)
            else:
                # Move to the next node if no drone was deployed
                i += 1
        
        # Add this truck's drone routes to the collection
        all_drone_routes.append(truck_drone_routes)
    
    # Remove all used nodes from modified truck routes
    for route_idx, route in enumerate(modified_truck_routes):
        nodes_to_remove = set()
        
        # Collect nodes to remove from this truck's route
        for drone_idx, drone_routes in enumerate(all_drone_routes[route_idx]):
            for drone_route in drone_routes:
                for node in drone_route[1:-1]:  # Skip takeoff and landing points
                    # Check if node is a valid delivery node (not a virtual point)
                    if not isinstance(node, str):
                        nodes_to_remove.add(int(node))  # Convert to int to ensure proper comparison
        
        # Remove the nodes from the truck route
        # We need to create a new list to avoid modifying during iteration
        modified_truck_routes[route_idx] = [node for node in route if int(node) not in nodes_to_remove]
    
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

def plot_combined_routes(truck_routes, all_drone_routes, x_coords, y_coords):
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
    virtual_takeoffs = []
    virtual_landings = []
    
    # Plot drone routes with dashed arrows
    for truck_idx, truck_drone_routes in enumerate(all_drone_routes):
        drone_color = drone_colors[truck_idx % len(drone_colors)]
        
        for drone_idx, drone_route_list in enumerate(truck_drone_routes):
            for route in drone_route_list:
                if not route:
                    continue
                    
                takeoff = route[0]
                landing = route[-1]
                
                # Identify delivery nodes (filter out virtual nodes)
                delivery_nodes = []
                for node in route[1:-1]:
                    if isinstance(node, (int, float)) and not isinstance(node, str) and not isinstance(node, tuple):
                        delivery_nodes.append(node)
                
                # Handle virtual takeoff points
                if isinstance(takeoff, str) and takeoff.startswith('V('):
                    coords = takeoff.strip('V()').split(',')
                    if len(coords) == 2:
                        try:
                            x, y = float(coords[0]), float(coords[1])
                            virtual_takeoffs.append((x, y, truck_idx, drone_idx))
                        except ValueError:
                            pass
                else:
                    takeoff_nodes.add(takeoff)
                
                # Handle virtual landing points
                if isinstance(landing, str) and landing.startswith('V('):
                    coords = landing.strip('V()').split(',')
                    if len(coords) == 2:
                        try:
                            x, y = float(coords[0]), float(coords[1])
                            virtual_landings.append((x, y, truck_idx, drone_idx))
                        except ValueError:
                            pass
                else:
                    landing_nodes.add(landing)
                
                delivery_only_nodes.update(delivery_nodes)
                
                # Draw arrows for drone flight path
                prev_node = route[0]
                for next_node in route[1:]:
                    # Handle virtual nodes for takeoff
                    if isinstance(prev_node, str) and prev_node.startswith('V('):
                        coords = prev_node.strip('V()').split(',')
                        if len(coords) == 2:
                            try:
                                prev_x, prev_y = float(coords[0]), float(coords[1])
                                
                                if isinstance(next_node, str) and next_node.startswith('V('):
                                    # Virtual to virtual
                                    coords = next_node.strip('V()').split(',')
                                    next_x, next_y = float(coords[0]), float(coords[1])
                                    plt.arrow(prev_x, prev_y,
                                             next_x - prev_x, next_y - prev_y,
                                             head_width=0.5, length_includes_head=True, color=drone_color,
                                             linestyle='dashed', alpha=0.7)
                                else:
                                    # Virtual to node
                                    plt.arrow(prev_x, prev_y,
                                             x_coords[next_node] - prev_x, y_coords[next_node] - prev_y,
                                             head_width=0.5, length_includes_head=True, color=drone_color,
                                             linestyle='dashed', alpha=0.7)
                            except ValueError:
                                pass
                    # Handle virtual nodes for landing
                    elif isinstance(next_node, str) and next_node.startswith('V('):
                        coords = next_node.strip('V()').split(',')
                        if len(coords) == 2:
                            try:
                                next_x, next_y = float(coords[0]), float(coords[1])
                                plt.arrow(x_coords[prev_node], y_coords[prev_node],
                                         next_x - x_coords[prev_node], next_y - y_coords[prev_node],
                                         head_width=0.5, length_includes_head=True, color=drone_color,
                                         linestyle='dashed', alpha=0.7)
                            except ValueError:
                                pass
                    else:
                        # Normal node-to-node
                        plt.arrow(x_coords[prev_node], y_coords[prev_node], 
                                 x_coords[next_node] - x_coords[prev_node], 
                                 y_coords[next_node] - y_coords[prev_node], 
                                 head_width=0.5, length_includes_head=True, color=drone_color, 
                                 linestyle='dashed', alpha=0.7)
                    
                    prev_node = next_node
                
                # Add annotation for drone
                mid_idx = len(route) // 2
                if mid_idx < len(route) and isinstance(route[mid_idx], (int, float)) and not isinstance(route[mid_idx], str):
                    plt.text(x_coords[route[mid_idx]], y_coords[route[mid_idx]], 
                             f"T{truck_idx+1}D{drone_idx+1}", fontsize=8, color=drone_color, weight='bold')
    
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
    
    # Virtual takeoff points
    if virtual_takeoffs:
        plt.scatter([x for x, y, _, _ in virtual_takeoffs], 
                    [y for x, y, _, _ in virtual_takeoffs], 
                    color='cyan', marker='*', s=120, label="Moving Truck Takeoff")
        
        # Label virtual takeoffs
        for x, y, truck_idx, drone_idx in virtual_takeoffs:
            plt.text(x, y, f"VT-T{truck_idx+1}D{drone_idx+1}", fontsize=8, 
                     verticalalignment='bottom', horizontalalignment='right', color='cyan')
    
    # Virtual landing points
    if virtual_landings:
        plt.scatter([x for x, y, _, _ in virtual_landings], 
                    [y for x, y, _, _ in virtual_landings], 
                    color='magenta', marker='*', s=120, label="Moving Truck Landing")
        
        # Label virtual landings
        for x, y, truck_idx, drone_idx in virtual_landings:
            plt.text(x, y, f"VL-T{truck_idx+1}D{drone_idx+1}", fontsize=8, 
                     verticalalignment='bottom', horizontalalignment='right', color='magenta')
    
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
    plt.title("Combined Truck and Drone Routes with Virtual Takeoff/Landing Points")
    plt.grid(True)
    plt.show()

def calculate_total_cost(truck_routes, all_drone_routes, distances_df):
    # Calculate full cost (including depot connections)
    total_cost = 0
    truck_positions = {}  # To track truck position over time
    
    # Calculate truck routes cost and establish movement profile
    for route_idx, route in enumerate(truck_routes):
        route_cost = calculate_route_cost(route, distances_df)
        total_cost += route_cost
        
        # Calculate position at each time step for this truck
        current_time = 0
        truck_positions[route_idx] = [(0, route[0])]  # (time, node)
        
        for i in range(len(route) - 1):
            segment_time = distances_df.iloc[route[i], route[i+1]]
            current_time += segment_time
            truck_positions[route_idx].append((current_time, route[i+1]))
    
    print("Only truck cost - ", total_cost)    
    
    # Calculate drone cost - considering multi-delivery routes and dynamic battery model
    drone_cost = 0
    waiting_cost = 0
    
    for truck_idx, truck_drone_routes in enumerate(all_drone_routes):
        truck_drone_cost = 0
        truck_waiting_cost = 0
        
        # Get this truck's movement profile
        truck_schedule = truck_positions[truck_idx]
        
        for drone_idx, drone_route_list in enumerate(truck_drone_routes):
            for route in drone_route_list:
                if not route:
                    continue
                    
                takeoff_point = route[0]
                landing_point = route[-1]
                
                # Initialize drone flight cost calculation
                fixed_cost = 2  # 1 for takeoff + 1 for landing
                flight_distance = 0
                
                # Calculate total flight distance
                prev_node = route[0]
                for next_node in route[1:]:
                    # Handle different node types
                    if isinstance(prev_node, str) and prev_node.startswith('V('):
                        # Virtual takeoff
                        coords = prev_node.strip('V()').split(',')
                        if len(coords) == 2:
                            try:
                                prev_x, prev_y = float(coords[0]), float(coords[1])
                                
                                if isinstance(next_node, str) and next_node.startswith('V('):
                                    # Virtual to virtual
                                    coords = next_node.strip('V()').split(',')
                                    next_x, next_y = float(coords[0]), float(coords[1])
                                    segment_distance = math.sqrt((prev_x - next_x)**2 + (prev_y - next_y)**2)
                                else:
                                    # Virtual to node
                                    segment_distance = math.sqrt((prev_x - x_coords[next_node])**2 + 
                                                               (prev_y - y_coords[next_node])**2)
                                
                                flight_distance += segment_distance
                            except ValueError:
                                pass
                    elif isinstance(next_node, str) and next_node.startswith('V('):
                        # Node to virtual
                        coords = next_node.strip('V()').split(',')
                        if len(coords) == 2:
                            try:
                                next_x, next_y = float(coords[0]), float(coords[1])
                                segment_distance = math.sqrt((x_coords[prev_node] - next_x)**2 + 
                                                           (y_coords[prev_node] - next_y)**2)
                                flight_distance += segment_distance
                            except ValueError:
                                pass
                    else:
                        # Normal node-to-node
                        segment_distance = distances_df.iloc[prev_node, next_node]
                        flight_distance += segment_distance
                    
                    prev_node = next_node
                
                # Drones are 1.5x faster than trucks
                flight_time = flight_distance / 1.5
                
                # Calculate route cost
                route_cost = flight_time + fixed_cost
                truck_drone_cost += route_cost
        
        print(f"Truck {truck_idx+1} drone cost - {truck_drone_cost}")
        print(f"Truck {truck_idx+1} waiting cost - {truck_waiting_cost}")
        
        drone_cost += truck_drone_cost
        waiting_cost += truck_waiting_cost
    
    print("Total drone cost - ", drone_cost)
    print("Total waiting cost - ", waiting_cost)
    
    # Add drone and waiting costs to total cost
    total_cost += drone_cost + waiting_cost
    
    # Calculate delivery-only cost
    delivery_cost = 0
    for route in truck_routes:
        if len(route) > 2:  # Only if route has at least one delivery point
            # Sum distances between delivery points only
            for i in range(1, len(route) - 2):  # Start at first delivery, end before last delivery
                delivery_cost += distances_df.iloc[route[i], route[i+1]]
    
    # Add drone delivery costs
    for truck_drone_routes in all_drone_routes:
        for drone_route_list in truck_drone_routes:
            for route in drone_route_list:
                if not route:
                    continue
                    
                # Add fixed takeoff/landing costs
                delivery_cost += 2
                
                prev_node = route[0]
                for next_node in route[1:]:
                    # Handle different node types (similar to above)
                    if isinstance(prev_node, str) and prev_node.startswith('V('):
                        coords = prev_node.strip('V()').split(',')
                        if len(coords) == 2:
                            try:
                                prev_x, prev_y = float(coords[0]), float(coords[1])
                                
                                if isinstance(next_node, str) and next_node.startswith('V('):
                                    coords = next_node.strip('V()').split(',')
                                    next_x, next_y = float(coords[0]), float(coords[1])
                                    segment_distance = math.sqrt((prev_x - next_x)**2 + (prev_y - next_y)**2)
                                else:
                                    segment_distance = math.sqrt((prev_x - x_coords[next_node])**2 + 
                                                               (prev_y - y_coords[next_node])**2)
                                
                                delivery_cost += segment_distance / 1.5
                            except ValueError:
                                pass
                    elif isinstance(next_node, str) and next_node.startswith('V('):
                        coords = next_node.strip('V()').split(',')
                        if len(coords) == 2:
                            try:
                                next_x, next_y = float(coords[0]), float(coords[1])
                                segment_distance = math.sqrt((x_coords[prev_node] - next_x)**2 + 
                                                           (y_coords[prev_node] - next_y)**2)
                                delivery_cost += segment_distance / 1.5
                            except ValueError:
                                pass
                    else:
                        segment_distance = distances_df.iloc[prev_node, next_node]
                        delivery_cost += segment_distance / 1.5
                    
                    prev_node = next_node
    
    return total_cost, delivery_cost

# Main Execution
file_path = "A-n32-k5.vrp"
d_capacity = 35
num_trucks, capacity, x_coords, y_coords, demands = read_cvrp_file(file_path)
num_trucks = 5
print(f"Number of trucks: {num_trucks}")
print(demands)
truck_routes = i_k_means(num_trucks, capacity, x_coords, y_coords, demands)

# Plot truck routes after Clarke-Wright Savings
print("-----Truck routes-----")
for route in truck_routes:
    print([int(x) for x in route])
plot_truck_routes(truck_routes, x_coords, y_coords)

# Apply DTRC algorithm
distances_df = distance_matrix_from_xy(x_coords, y_coords)
drone_routes, modified_truck_routes = apply_dtrc(truck_routes, distances_df, demands, d_capacity,x_coords,y_coords)

# Plot combined truck and drone routes
print("-----Modified Truck routes-----")
for route in modified_truck_routes:
    print([int(x) for x in route])

formatted_drone_routes = []
for truck_drones in drone_routes:
    truck_formatted = []
    for drone_trips in truck_drones:
        drone_formatted = []
        for route in drone_trips:
            route_formatted = []
            for node in route:
                if isinstance(node, (int, float)) and not isinstance(node, str) and not isinstance(node, bool):
                    # Only convert numeric nodes to int
                    route_formatted.append(int(node))
                else:
                    # Keep virtual nodes as they are
                    route_formatted.append(node)
            drone_formatted.append(route_formatted)
        truck_formatted.append(drone_formatted)
    formatted_drone_routes.append(truck_formatted)

print("-----Drone routes-----")
print(formatted_drone_routes)

plot_combined_routes(modified_truck_routes, drone_routes, x_coords, y_coords)

# Print total cost
total_cost, delivery_cost = calculate_total_cost(modified_truck_routes, drone_routes, distances_df)
print(f"Total Cost: {total_cost}")
print(f"Delivery Cost: {delivery_cost}")