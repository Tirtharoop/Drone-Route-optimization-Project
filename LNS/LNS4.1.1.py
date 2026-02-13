import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class TruckDroneRouting:
    def __init__(self, truck_routes, drone_routes, demands,
                 euclidean_distance_matrix,
                 max_iterations=1000, p1=0.3, p2=0.3, p3=0.3,
                 drone_capacity=99999, drone_cost_factor=0.8,
                 #truck_speed=1, drone_speed,
                 time_weight=0.5, forace_use_drones=True, num_drones=5):
        """
        Initialize the Truck-Drone Routing problem with LNS algorithm

        Args:
            truck_routes: List of truck routes, each route is a list of customer indices
            drone_routes: List of drone routes for each truck, where each truck has up to num_drones drones
            demands: Dictionary of node demands (node_id -> demand value)
	    euclidean_distance_matrix: Matrix of Euclidean distances between nodes for drones
            max_iterations: Maximum number of iterations for the LNS algorithm
            p1: Percentage of drone-only nodes to remove
            p2: Percentage of truck-only nodes to remove
            p3: Percentage of sub-drone routes to remove
            drone_capacity: Maximum capacity of drones
            drone_cost_factor: Cost multiplier for drone routes (< 1 makes drones cheaper)
            truck_speed: Speed of trucks in km/h
            drone_speed: Speed of drones in km/h
            time_weight: Weight for time optimization (1-time_weight is cost weight)
            force_use_drones: Whether to force the solution to use drones
            num_drones: Number of drones available for each truck
        """
        self.truck_routes = truck_routes
        self.drone_routes = drone_routes
        self.demands = demands
        #self.euclidean_distance_matrix = euclidean_distance_matrix
        self.euclidean_distance_matrix = euclidean_distance_matrix
        self.max_iterations = max_iterations
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.drone_capacity = drone_capacity
        self.drone_cost_factor = drone_cost_factor
        self.truck_speed = 1  # km/h
        self.drone_speed = 1.5  # km/h
        self.time_weight = time_weight  # weight for time optimization
        self.force_use_drones = force_use_drones
        self.num_drones = num_drones
        self.best_solution = None
        self.best_objective = float('inf')
        self.current_solution = None
        self.current_objective = float('inf')

        # Initialize node locations for visualization
        self.num_nodes = max(demands.keys()) + 1
        # Use a fixed seed for reproducibility
        np.random.seed(42)
        self.node_locations = np.random.rand(self.num_nodes, 2) * 100

        # Validate input
        self._validate_input()

    def _validate_initial_solution(self):
        """Ensure all nodes with non-zero demand are included in the initial solution."""
        all_nodes = set(self.demands.keys())
        truck_nodes = set()
        drone_nodes = set()

        for route in self.truck_routes:
            for node in route:
                if node != 0:  # Exclude depot
                    truck_nodes.add(node)

        for truck_drones in self.drone_routes:
            for drone_list in truck_drones:
                for drone_route in drone_list:
                    for node in drone_route:
                        if node != 0:  # Exclude depot
                            drone_nodes.add(node)

        all_included_nodes = truck_nodes.union(drone_nodes)
        missing_nodes = all_nodes - all_included_nodes - {0}  # Exclude depot

        if missing_nodes:
            raise ValueError(f"Nodes {missing_nodes} are not included in any route.")

    def _validate_input(self):
        """Validate that input matches expected format"""
        if len(self.truck_routes) != len(self.drone_routes):
            raise ValueError("Number of truck routes must match number of drone route sets")

        for i, truck_drones in enumerate(self.drone_routes):
            if len(truck_drones) > self.num_drones:
                raise ValueError(f"Truck {i} has more than {self.num_drones} drones assigned")

        # Validate distance matrices
        expected_size = self.num_nodes
        #if self.euclidean_distance_matrix.shape != (expected_size, expected_size):
            #raise ValueError(f"Road distance matrix should be {expected_size}x{expected_size}")
        if self.euclidean_distance_matrix.shape != (expected_size, expected_size):
            raise ValueError(f"Euclidean distance matrix should be {expected_size}x{expected_size}")

    def truck_distance(self, node1, node2):
        """Calculate road distance between two nodes for truck by directly using the distance matrix"""
        return self.euclidean_distance_matrix[node1, node2]

    def drone_distance(self, node1, node2):
        """Calculate Euclidean distance between two nodes for drone"""
        return self.euclidean_distance_matrix[node1, node2]

    def calculate_route_distance(self, route, is_drone=False):
        """Calculate the total distance of a single route"""
        if not route or len(route) < 2:
            return 0

        total_distance = 0
        for i in range(len(route) - 1):
            if is_drone:
                total_distance += self.drone_distance(route[i], route[i+1])
            else:
                total_distance += self.truck_distance(route[i], route[i+1])
        return total_distance

    def calculate_route_cost(self, route, is_drone=False):
        """Calculate the cost of a single route by summing distances between consecutive nodes"""
        if not route or len(route) < 2:
            return 0

        total_cost = 0
        for i in range(len(route) - 1):
            if is_drone:
                # For drone routes, use Euclidean distance plus cost factor
                distance = self.drone_distance(route[i], route[i+1])
                total_cost += distance + self.drone_cost_factor
            else:
                # For truck routes, use direct distance from matrix
                distance = self.truck_distance(route[i], route[i+1])
                total_cost += distance

        return total_cost

    def calculate_route_time(self, route, is_drone=False):
        """Calculate the time taken for a single route in hours"""
        distance = self.calculate_route_distance(route, is_drone)
        if is_drone:
            return distance / self.drone_speed  # hours
        else:
            return distance / self.truck_speed  # hours

    def calculate_solution_objective(self, truck_routes, drone_routes):
        """
        Calculate the total objective value (weighted sum of cost and time) of a solution
        """
        total_cost = 0
        max_time = 0  # Consider the maximum route time (makespan)

        # Calculate truck route costs and times
        for route in truck_routes:
            route_cost = self.calculate_route_cost(route)
            route_time = self.calculate_route_time(route)
            total_cost += route_cost
            max_time = max(max_time, route_time)

        # Calculate drone route costs and times
        for truck_idx, truck_drones in enumerate(drone_routes):
            truck_drone_max_time = 0
            for drone_list in truck_drones:
                for drone_route in drone_list:
                    if len(drone_route) >= 2:
                        route_cost = self.calculate_route_cost(drone_route, is_drone=True)
                        route_time = self.calculate_route_time(drone_route, is_drone=True)
                        total_cost += route_cost
                        truck_drone_max_time = max(truck_drone_max_time, route_time)

            # Consider parallel operation of truck and drones
            if truck_idx < len(truck_routes):
                truck_time = self.calculate_route_time(truck_routes[truck_idx])
                max_time = max(max_time, max(truck_time, truck_drone_max_time))

        # Penalize solutions with no drone routes if force_use_drones is True
        if self.force_use_drones:
            drone_routes_count = sum(len(drone_route) >= 2
                                   for truck_drones in drone_routes
                                   for drone_list in truck_drones
                                   for drone_route in drone_list)
            if drone_routes_count == 0:
                total_cost *= 1.5  # Apply penalty
                max_time *= 1.5    # Apply penalty

        # Weighted objective: balance between cost and time
        objective = (1 - self.time_weight) * total_cost + self.time_weight * (max_time * 1000)  # Scale time for balance

        return objective, total_cost, max_time

    def get_drone_only_nodes(self, truck_routes, drone_routes):
        """Identify nodes that are served by drones only"""
        truck_nodes = set()
        for route in truck_routes:
            for node in route:
                truck_nodes.add(node)

        drone_nodes = set()
        for truck_drones in drone_routes:
            for drone_list in truck_drones:
                for drone_route in drone_list:
                    for node in drone_route:
                        drone_nodes.add(node)

        # Nodes that appear only in drone routes
        drone_only_nodes = drone_nodes - truck_nodes
        return list(drone_only_nodes)

    def get_truck_only_nodes(self, truck_routes, drone_routes):
        """Identify nodes that are served by trucks only (excluding depot and launch/landing nodes)"""
        # First, identify all launch/landing nodes
        launch_landing_nodes = set()
        for i, truck_drones in enumerate(drone_routes):
            for drone_list in truck_drones:
                for drone_route in drone_list:
                    if len(drone_route) >= 2:
                        # First and last nodes in drone route are launch/landing
                        launch_landing_nodes.add(drone_route[0])
                        launch_landing_nodes.add(drone_route[-1])

        # Identify all nodes in truck routes
        truck_nodes = set()
        for route in truck_routes:
            for node in route:
                if node != 0:  # Exclude depot
                    truck_nodes.add(node)

        # Nodes that appear in truck routes but are not launch/landing nodes
        truck_only_nodes = truck_nodes - launch_landing_nodes
        return list(truck_only_nodes)

    def drone_node_removal(self, truck_routes, drone_routes):
        """
        Remove p1% of drone-only nodes from the solution
        Returns the list of removed nodes
        """
        removed_nodes = []

        # Get drone-only nodes
        drone_only_nodes = self.get_drone_only_nodes(truck_routes, drone_routes)

        if not drone_only_nodes:
            # If no drone-only nodes, identify nodes suitable for drone delivery
            # based on low demand and try to remove some from truck routes
            potential_drone_nodes = []
            for node, demand in self.demands.items():
                if node != 0 and demand <= self.drone_capacity / 2:  # Nodes with small demands
                    potential_drone_nodes.append(node)

            if potential_drone_nodes:
                num_nodes_to_remove = max(1, min(3, len(potential_drone_nodes)))
                nodes_to_remove = random.sample(potential_drone_nodes, num_nodes_to_remove)

                for node in nodes_to_remove:
                    for i, route in enumerate(truck_routes):
                        if node in route:
                            truck_routes[i].remove(node)
                            removed_nodes.append(('drone', node))
                            break

            return removed_nodes

        # Determine number of nodes to remove
        num_nodes_to_remove = max(1, int(len(drone_only_nodes) * self.p1))
        nodes_to_remove = random.sample(drone_only_nodes, min(num_nodes_to_remove, len(drone_only_nodes)))

        # Remove nodes from drone routes
        for node in nodes_to_remove:
            for i, truck_drones in enumerate(drone_routes):
                for j, drone_list in enumerate(truck_drones):
                    for k, drone_route in enumerate(drone_list):
                        if node in drone_route:
                            # Don't remove first and last nodes (launch/landing)
                            if node != drone_route[0] and node != drone_route[-1]:
                                drone_routes[i][j][k].remove(node)
                                removed_nodes.append(('drone', node))

                            # If all nodes in a sub-drone route are removed, make landing/launching node available
                            if len(drone_routes[i][j][k]) <= 2:  # Only launch and landing nodes left
                                drone_routes[i][j][k] = []

        return removed_nodes

    def truck_node_removal(self, truck_routes, drone_routes):
        """
        Remove p2% of truck-only nodes from the solution
        Returns the list of removed nodes
        """
        removed_nodes = []

        # Get truck-only nodes
        truck_only_nodes = self.get_truck_only_nodes(truck_routes, drone_routes)

        if not truck_only_nodes:
            return removed_nodes

        # Determine number of nodes to remove
        num_nodes_to_remove = max(1, int(len(truck_only_nodes) * self.p2))
        nodes_to_remove = random.sample(truck_only_nodes, min(num_nodes_to_remove, len(truck_only_nodes)))

        # Remove nodes from truck routes
        for node in nodes_to_remove:
            for i, route in enumerate(truck_routes):
                if node in route:
                    truck_routes[i].remove(node)
                    removed_nodes.append(('truck', node))

                    # If all customer nodes are removed, truck stays at depot
                    if len(truck_routes[i]) <= 2:  # Only depot node left
                        truck_routes[i] = [0, 0]

        return removed_nodes

    def sub_drone_route_removal(self, truck_routes, drone_routes):
        """
        Remove p3% of sub-drone routes from the solution
        Returns the list of removed sub-routes
        """
        removed_subroutes = []
        all_subroutes = []

        # Gather all sub-drone routes
        for i, truck_drones in enumerate(drone_routes):
            for j, drone_list in enumerate(truck_drones):
                for k, drone_route in enumerate(drone_list):
                    if len(drone_route) > 2:  # Only consider non-empty routes
                        all_subroutes.append((i, j, k))

        if not all_subroutes:
            return removed_subroutes

        # Determine number of sub-routes to remove
        num_routes_to_remove = max(1, int(len(all_subroutes) * self.p3))
        routes_to_remove = random.sample(all_subroutes, min(num_routes_to_remove, len(all_subroutes)))

        # Remove the selected sub-routes
        for i, j, k in routes_to_remove:
            removed_nodes = []
            for node in drone_routes[i][j][k][1:-1]:  # Skip launch/landing nodes
                if node != 0:  # Don't add depot to removed nodes
                    removed_nodes.append(('subroute', node))

            # Get launch and landing nodes
            launch_node = drone_routes[i][j][k][0]
            landing_node = drone_routes[i][j][k][-1]

            # Clear the sub-route
            drone_routes[i][j][k] = []
            removed_subroutes.extend(removed_nodes)

        return removed_subroutes

    def destroy_solution(self, truck_routes, drone_routes):
        """
        Execute the three destroy operators in sequence
        Returns list of all removed nodes/subroutes
        """
        # Execute the operators with equal probability
        operator_choice = random.random()

        if operator_choice < 0.33:
            # Execute drone node removal
            removed_nodes = self.drone_node_removal(truck_routes, drone_routes)
        elif operator_choice < 0.66:
            # Execute truck node removal
            removed_nodes = self.truck_node_removal(truck_routes, drone_routes)
        else:
            # Execute sub-drone route removal
            removed_nodes = self.sub_drone_route_removal(truck_routes, drone_routes)

        # If not enough nodes removed, apply another operator
        if len(removed_nodes) < 2:
            if random.random() < 0.5:
                removed_nodes.extend(self.truck_node_removal(truck_routes, drone_routes))
            else:
                removed_nodes.extend(self.drone_node_removal(truck_routes, drone_routes))

        return removed_nodes

    def calculate_drone_load(self, drone_route):
        """Calculate the total load of a drone route"""
        total_load = 0
        for node in drone_route:
            if node in self.demands:
                total_load += self.demands[node]
        return total_load

    def drone_node_insertion(self, truck_routes, drone_routes, node):
        """
        Insert a node into an existing drone sub-route
        Returns the best resulting solution and its objective value
        """
        best_objective = float('inf')
        best_solution = None

        # Create copies for manipulation
        truck_routes_copy = copy.deepcopy(truck_routes)
        drone_routes_copy = copy.deepcopy(drone_routes)

        # Try inserting into each drone sub-route
        for i, truck_drones in enumerate(drone_routes_copy):
            for j, drone_list in enumerate(truck_drones):
                for k, drone_route in enumerate(drone_list):
                    # Skip empty routes
                    if len(drone_route) < 2:
                        continue

                    # Try each insertion position
                    for pos in range(1, len(drone_route)):  # Skip inserting at launch node
                        # Create a new route with node inserted
                        new_route = drone_route.copy()
                        new_route.insert(pos, node)

                        # Check capacity constraint
                        if self.calculate_drone_load(new_route) <= self.drone_capacity:
                            # Update the route
                            drone_routes_copy[i][j][k] = new_route

                            # Calculate new objective
                            objective, _, _ = self.calculate_solution_objective(truck_routes_copy, drone_routes_copy)

                            # Update best if improved
                            if objective < best_objective:
                                best_objective = objective
                                best_solution = (copy.deepcopy(truck_routes_copy), copy.deepcopy(drone_routes_copy))

                            # Restore original route for next iteration
                            drone_routes_copy[i][j][k] = drone_route

        return best_solution, best_objective

    def truck_node_insertion(self, truck_routes, drone_routes, node):
        """
        Insert a node into an existing truck route
        Returns the best resulting solution and its objective value
        """
        best_objective = float('inf')
        best_solution = None

        # Create copies for manipulation
        truck_routes_copy = copy.deepcopy(truck_routes)
        drone_routes_copy = copy.deepcopy(drone_routes)

        # Try inserting into each truck route
        for i, route in enumerate(truck_routes_copy):
            # Try each insertion position
            for pos in range(1, len(route)):  # Skip inserting at depot
                # Create a new route with node inserted
                new_route = route.copy()
                new_route.insert(pos, node)

                # Update the route
                truck_routes_copy[i] = new_route

                # Calculate new objective
                objective, _, _ = self.calculate_solution_objective(truck_routes_copy, drone_routes_copy)

                # Update best if improved
                if objective < best_objective:
                    best_objective = objective
                    best_solution = (copy.deepcopy(truck_routes_copy), copy.deepcopy(drone_routes_copy))

                # Restore original route for next iteration
                truck_routes_copy[i] = route

        # If no feasible insertion found, try creating a new truck route
        if best_solution is None:
            # Create a new truck route
            new_route = [0, node, 0]
            truck_routes_copy.append(new_route)

            # Create empty drone routes for the new truck
            drone_routes_copy.append([[] for _ in range(self.num_drones)])

            # Calculate new objective
            objective, _, _ = self.calculate_solution_objective(truck_routes_copy, drone_routes_copy)
            best_objective = objective
            best_solution = (truck_routes_copy, drone_routes_copy)

        return best_solution, best_objective

    def drone_route_creation(self, truck_routes, drone_routes, node):
        """
        Create a new drone sub-route by inserting the node between a pair of truck nodes
        Returns the best resulting solution and its objective value
        """
        best_objective = float('inf')
        best_solution = None

        # Create copies for manipulation
        truck_routes_copy = copy.deepcopy(truck_routes)
        drone_routes_copy = copy.deepcopy(drone_routes)

        # Try creating a new drone route for each truck route
        for i, route in enumerate(truck_routes_copy):
            # Skip routes with less than 2 nodes
            if len(route) < 2:
                continue

            # Try each pair of consecutive nodes in the truck route
            for j in range(len(route) - 1):
                launch_node = route[j]
                landing_node = route[j+1]

                # Skip if launch or landing is depot
                if launch_node == 0 or landing_node == 0:
                    continue

                # Create a new drone route
                new_drone_route = [launch_node, node, landing_node]

                # Check capacity constraint
                if self.calculate_drone_load(new_drone_route) <= self.drone_capacity:
                    # Find an available drone
                    for d in range(self.num_drones):
                        # Find an empty slot in the drone list
                        found_slot = False
                        for e in range(len(drone_routes_copy[i][d])):
                            if not drone_routes_copy[i][d][e]:
                                drone_routes_copy[i][d][e] = new_drone_route
                                found_slot = True
                                break

                        if not found_slot and len(drone_routes_copy[i][d]) < 5:  # Limit number of sub-routes
                            drone_routes_copy[i][d].append(new_drone_route)
                            found_slot = True

                        if found_slot:
                            # Calculate new objective
                            objective, _, _ = self.calculate_solution_objective(truck_routes_copy, drone_routes_copy)

                            # Update best if improved
                            if objective < best_objective:
                                best_objective = objective
                                best_solution = (copy.deepcopy(truck_routes_copy), copy.deepcopy(drone_routes_copy))

                            # Reset for next iteration
                            if 'e' in locals():
                                if e < len(drone_routes_copy[i][d]):
                                    drone_routes_copy[i][d][e] = []
                                else:
                                    drone_routes_copy[i][d].pop()
                            break

        return best_solution, best_objective

    def adaptive_removal(self, truck_routes, drone_routes, removed_nodes):
        """
        Apply adaptive removal heuristic to adjust removal strategy based on solution quality and diversity
        """
        # Calculate current solution quality
        current_objective, _, _ = self.calculate_solution_objective(truck_routes, drone_routes)

        # Calculate diversity of removed nodes
        diversity = len(set(node[1] for node in removed_nodes))

        # Adjust removal probabilities based on solution quality and diversity
        if current_objective < self.best_objective:
            # Improve exploration by increasing removal probabilities
            self.p1 = min(0.5, self.p1 * 1.1)
            self.p2 = min(0.5, self.p2 * 1.1)
            self.p3 = min(0.5, self.p3 * 1.1)
        elif diversity < len(removed_nodes) / 2:
            # Improve exploitation by decreasing removal probabilities
            self.p1 = max(0.1, self.p1 * 0.9)
            self.p2 = max(0.1, self.p2 * 0.9)
            self.p3 = max(0.1, self.p3 * 0.9)

        return removed_nodes

    def repair_solution(self, truck_routes, drone_routes, removed_nodes):
        """
        Repair the solution by applying the three repair operators
        and selecting the best insertion for each node. Ensure all nodes are included.
        """
        # Apply adaptive removal heuristic
        removed_nodes = self.adaptive_removal(truck_routes, drone_routes, removed_nodes)

        # Shuffle removed nodes to add randomness
        random.shuffle(removed_nodes)

        # Current working solution
        current_truck_routes = copy.deepcopy(truck_routes)
        current_drone_routes = copy.deepcopy(drone_routes)

        # Track nodes that have been reinserted
        reinserted_nodes = set()

        # Bias the repair to favor drone insertions for time-sensitive operations
        for item in removed_nodes:
            # Extract node information
            if isinstance(item, tuple) and len(item) >= 2:
                node_type, node = item[0], item[1]
            else:
                continue  # Skip if unknown format

            # Check node demand
            node_demand = self.demands.get(node, 0)

            # Bias repair based on node type and demand
            if node_demand <= self.drone_capacity:
                # For low-demand nodes, favor drone operations (faster delivery)
                operations = [
                    (self.drone_route_creation, 0.6),  # 60% chance of trying drone route creation first
                    (self.drone_node_insertion, 0.3),  # 30% chance of trying drone insertion first
                    (self.truck_node_insertion, 0.1)   # 10% chance of trying truck insertion first
                ]
            else:
                # For high-demand nodes, bias toward truck insertion
                operations = [
                    (self.truck_node_insertion, 0.7),
                    (self.drone_route_creation, 0.2),
                    (self.drone_node_insertion, 0.1)
                ]

            # Sort operations based on random probability
            operations.sort(key=lambda x: random.random() / x[1])

            best_objective = float('inf')
            best_solution = None

            # Try each repair operation
            for op, _ in operations:
                solution, obj_value = op(current_truck_routes, current_drone_routes, node)
                if solution and obj_value < best_objective:
                    best_objective = obj_value
                    best_solution = solution

            # Update current solution with best repair
            if best_solution:
                current_truck_routes, current_drone_routes = best_solution
                reinserted_nodes.add(node)

        # Ensure all nodes are included
        all_nodes = set(self.demands.keys()) - {0}  # Exclude depot
        included_nodes = set()

        for route in current_truck_routes:
            for node in route:
                if node != 0:
                    included_nodes.add(node)

        for truck_drones in current_drone_routes:
            for drone_list in truck_drones:
                for drone_route in drone_list:
                    for node in drone_route:
                        if node != 0:
                            included_nodes.add(node)

        missing_nodes = all_nodes - included_nodes

        # Reinsert missing nodes
        for node in missing_nodes:
            if self.demands[node] <= self.drone_capacity:
                # Try drone insertion first
                solution, _ = self.drone_node_insertion(current_truck_routes, current_drone_routes, node)
                if solution:
                    current_truck_routes, current_drone_routes = solution
                    continue

            # If drone insertion fails, try truck insertion
            solution, _ = self.truck_node_insertion(current_truck_routes, current_drone_routes, node)
            if solution:
                current_truck_routes, current_drone_routes = solution

        return current_truck_routes, current_drone_routes

    def enforce_drone_usage(self, truck_routes, drone_routes):
        """Ensure that drones are used in the solution by moving eligible nodes from trucks to drones"""
        # Count active drone routes
        active_drone_routes = sum(len(drone_route) > 0
                               for truck_drones in drone_routes
                               for drone_list in truck_drones
                               for drone_route in drone_list)

        # If we already have drone routes, no need to enforce
        if active_drone_routes > 0:
            return truck_routes, drone_routes

        # Make copies for manipulation
        truck_routes_copy = copy.deepcopy(truck_routes)
        drone_routes_copy = copy.deepcopy(drone_routes)

        # Identify nodes eligible for drone delivery (based on demand)
        eligible_nodes = []
        for i, route in enumerate(truck_routes):
            for j, node in enumerate(route):
                if node != 0 and self.demands.get(node, 0) <= self.drone_capacity:
                    # Store truck index, position in route, and node
                    eligible_nodes.append((i, j, node))

        # Sort eligible nodes by time savings potential (drone vs truck)
        eligible_nodes.sort(key=lambda x: self.calculate_time_savings_potential(x[2], truck_routes[x[0]]), reverse=True)

        # Try to create drone routes using eligible nodes
        success = False
        for truck_idx, _, node in eligible_nodes:
            if truck_idx >= len(truck_routes_copy):
                continue

            route = truck_routes_copy[truck_idx]
            if node in route:
                pos = route.index(node)
                if pos > 0 and pos < len(route) - 1:
                    prev_node = route[pos - 1]
                    next_node = route[pos + 1]

                    # Skip if adjacent nodes are depot
                    if prev_node == 0 or next_node == 0:
                        continue

                    # Try to create a drone route
                    new_drone_route = [prev_node, node, next_node]

                    # Check drone capacity
                    if self.calculate_drone_load(new_drone_route) <= self.drone_capacity:
                        # Find an available drone
                        for j in range(self.num_drones):
                            # Find an empty slot
                            for k in range(len(drone_routes_copy[truck_idx][j])):
                                if not drone_routes_copy[truck_idx][j][k]:
                                    # Assign the route
                                    drone_routes_copy[truck_idx][j][k] = new_drone_route
                                    # Remove node from truck route
                                    route.remove(node)
                                    success = True
                                    break

                            if success:
                                break

                    if success:
                        break

        if success:
            return truck_routes_copy, drone_routes_copy
        else:
            return truck_routes, drone_routes

    def calculate_time_savings_potential(self, node, truck_route):
        """Calculate potential time savings by moving a node from truck to drone"""
        if node not in truck_route or len(truck_route) < 3:
            return 0

        pos = truck_route.index(node)
        if pos == 0 or pos == len(truck_route) - 1:
            return 0  # Can't move first or last node (depot)

        # Calculate time using truck
        prev_node = truck_route[pos - 1]
        next_node = truck_route[pos + 1]
        truck_time = (self.truck_distance(prev_node, node) +
                      self.truck_distance(node, next_node)) / self.truck_speed

        # Calculate time using drone
        drone_time = (self.drone_distance(prev_node, node) +
                      self.drone_distance(node, next_node)) / self.drone_speed

        # Return time savings
        return truck_time - drone_time

    def run_lns(self):
        """Run the Local Neighborhood Search algorithm"""
        # Validate initial solution


        # Initialize current and best solution
        self.current_solution = (copy.deepcopy(self.truck_routes), copy.deepcopy(self.drone_routes))
        self.current_objective, current_cost, current_time = self.calculate_solution_objective(*self.current_solution)
        self.best_solution = copy.deepcopy(self.current_solution)
        self.best_objective = self.current_objective
        best_cost = current_cost
        best_time = current_time
        self._validate_initial_solution()
        # Apply initial drone enforcement if needed
        if self.force_use_drones:
            truck_routes, drone_routes = self.enforce_drone_usage(*self.current_solution)
            self.current_solution = (truck_routes, drone_routes)
            self.current_objective, current_cost, current_time = self.calculate_solution_objective(*self.current_solution)

            # Update best solution if improved
            if self.current_objective < self.best_objective:
                self.best_solution = copy.deepcopy(self.current_solution)
                self.best_objective = self.current_objective
                best_cost = current_cost
                best_time = current_time

        # Track progress
        objective_history = [self.current_objective]
        no_improvement_count = 0

        # Main LNS loop
        for iteration in range(self.max_iterations):
            # Make a copy of the current solution
            candidate_truck_routes = copy.deepcopy(self.current_solution[0])
            candidate_drone_routes = copy.deepcopy(self.current_solution[1])

            # Destroy part of the solution
            removed_elements = self.destroy_solution(candidate_truck_routes, candidate_drone_routes)

            # Repair the solution
            candidate_truck_routes, candidate_drone_routes = self.repair_solution(
                candidate_truck_routes, candidate_drone_routes, removed_elements
            )

            # Enforce drone usage if needed (randomly to avoid cycling)
            if self.force_use_drones and random.random() < 0.3:
                candidate_truck_routes, candidate_drone_routes = self.enforce_drone_usage(
                    candidate_truck_routes, candidate_drone_routes
                )

            # Calculate objective of candidate solution
            candidate_objective, candidate_cost, candidate_time = self.calculate_solution_objective(
                candidate_truck_routes, candidate_drone_routes
            )

            # Accept if better
            if candidate_objective < self.current_objective:
                self.current_solution = (candidate_truck_routes, candidate_drone_routes)
                self.current_objective = candidate_objective
                current_cost = candidate_cost
                current_time = candidate_time
                no_improvement_count = 0

                # Update best solution if needed
                if candidate_objective < self.best_objective:
                  self.best_solution = copy.deepcopy(self.current_solution)
                  self.best_objective = candidate_objective
                  best_cost = candidate_cost
                  best_time = candidate_time
            else:
                # Simulated annealing-like acceptance for diversification
                temperature = max(0.01, 1.0 - (iteration / self.max_iterations))
                acceptance_probability = np.exp(-(candidate_objective - self.current_objective) / temperature)

                if random.random() < acceptance_probability:
                    self.current_solution = (candidate_truck_routes, candidate_drone_routes)
                    self.current_objective = candidate_objective
                    current_cost = candidate_cost
                    current_time = candidate_time
                else:
                    no_improvement_count += 1

            # Record progress
            objective_history.append(self.current_objective)

            # Early stopping if no improvement for many iterations
            if no_improvement_count > 100:
                print(f"Early stopping at iteration {iteration} due to no improvement")
                break

            # Print progress every 100 iterations
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Current objective: {self.current_objective:.2f}, Best: {self.best_objective:.2f}")

        # Return best solution found
        return self.best_solution, self.best_objective, objective_history

    def visualize_solution(self, solution=None, title="Truck-Drone Routing Solution"):
        """Visualize the solution with trucks and drones"""
        if solution is None:
            solution = self.best_solution

        if solution is None:
            print("No solution available to visualize")
            return

        truck_routes, drone_routes = solution

        # Create figure
        plt.figure(figsize=(14, 10))

        # Plot depot
        plt.scatter(self.node_locations[0, 0], self.node_locations[0, 1],
                    color='black', s=200, marker='*', label='Depot')

        # Plot customer nodes
        for i in range(1, self.num_nodes):
            plt.scatter(self.node_locations[i, 0], self.node_locations[i, 1],
                        color='gray', s=100, alpha=0.7)
            plt.text(self.node_locations[i, 0]+1, self.node_locations[i, 1]+1,
                     str(i), fontsize=10)

        # Plot truck routes - using different colors
        truck_colors = ['blue', 'green', 'red', 'purple', 'brown', 'orange']

        for i, route in enumerate(truck_routes):
            color = truck_colors[i % len(truck_colors)]
            # Plot truck routes
            for j in range(len(route)-1):
                plt.plot([self.node_locations[route[j], 0], self.node_locations[route[j+1], 0]],
                         [self.node_locations[route[j], 1], self.node_locations[route[j+1], 1]],
                         color=color, linewidth=2, label=f'Truck {i+1}' if j == 0 else "")

            # Plot drone routes for this truck
            if i < len(drone_routes):
                drone_color = truck_colors[i % len(truck_colors)]
                for j, drone_list in enumerate(drone_routes[i]):
                    for k, drone_route in enumerate(drone_list):
                        if len(drone_route) >= 2:
                            # Plot drone route with dashed line
                            for l in range(len(drone_route)-1):
                                plt.plot([self.node_locations[drone_route[l], 0], self.node_locations[drone_route[l+1], 0]],
                                         [self.node_locations[drone_route[l], 1], self.node_locations[drone_route[l+1], 1]],
                                         color=drone_color, linestyle='--', linewidth=1,
                                         label=f'Drone {i+1}.{j+1}' if l == 0 and k == 0 else "")

                            # Highlight drone customers
                            for node in drone_route[1:-1]:  # Skip launch/landing nodes
                                plt.scatter(self.node_locations[node, 0], self.node_locations[node, 1],
                                           color=drone_color, s=80, alpha=0.5)

        # Calculate statistics for the solution
        objective, total_cost, max_time = self.calculate_solution_objective(truck_routes, drone_routes)

        # Count nodes served
        truck_served = sum(len(route) - 2 for route in truck_routes)  # -2 for depot at start/end
        drone_served = sum(len(drone_route) - 2
                          for truck_drones in drone_routes
                          for drone_list in truck_drones
                          for drone_route in drone_list if len(drone_route) >= 2)

        # Add title and legend
        plt.title(f"{title}\nObjective: {objective:.2f}, Cost: {total_cost:.2f}, Time: {max_time:.2f}h\n"
                 f"Truck nodes: {truck_served}, Drone nodes: {drone_served}")

        # Add a legend without duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt

    def run_and_visualize(self):
        """Run the LNS algorithm and visualize results"""
        try:
            print("Starting Truck-Drone Routing optimization...")
            best_solution, best_objective, history = self.run_lns()

            # Final check to ensure all nodes are included
            self._validate_initial_solution()

            print("\nOptimization completed:")
            print(f"Best objective: {best_objective:.2f}")

            # Visualize best solution
            plt.figure(figsize=(10, 6))
            plt.plot(history)
            plt.title('Objective Value History')
            plt.xlabel('Iteration')
            plt.ylabel('Objective Value')
            plt.grid(True, alpha=0.3)
            plt.show()

            # Visualize solution
            plt_solution = self.visualize_solution()
            plt_solution.show()

            # Print detailed route information
            self.print_detailed_routes(best_solution)

            return best_solution, best_objective

        except ValueError as e:
            print(f"Error: {e}")
            return None, None

    def process_input(self, truck_routes_input, drone_routes_input):
        """Process string input of truck and drone routes"""
        # For truck routes, each route should be a list starting and ending with depot 0
        processed_truck_routes = eval(truck_routes_input) if isinstance(truck_routes_input, str) else truck_routes_input

        # For drone routes, we need to process the nested structure
        processed_drone_routes = eval(drone_routes_input) if isinstance(drone_routes_input, str) else drone_routes_input

        # Validate and set
        self.truck_routes = processed_truck_routes
        self.drone_routes = processed_drone_routes

        # Print summary
        print(f"Loaded {len(self.truck_routes)} truck routes and {len(self.drone_routes)} drone route sets")
        print(f"Total demand nodes: {len(self.demands) - 1}")  # -1 for depot

        return self.truck_routes, self.drone_routes

    def evaluate_current_solution(self):
        """Evaluate the current solution and print statistics"""
        truck_routes, drone_routes = self.truck_routes, self.drone_routes
        objective, total_cost, max_time = self.calculate_solution_objective(truck_routes, drone_routes)

        # Count nodes served
        truck_served = sum(len(route) - route.count(0) for route in truck_routes)
        drone_served = sum(len(drone_route) - 2
                          for truck_drones in drone_routes
                          for drone_list in truck_drones
                          for drone_route in drone_list if len(drone_route) >= 2)

        # Calculate utilization
        truck_distances = [self.calculate_route_distance(route) for route in truck_routes]
        drone_distances = [self.calculate_route_distance(drone_route, is_drone=True)
                         for truck_drones in drone_routes
                         for drone_list in truck_drones
                         for drone_route in drone_list if len(drone_route) >= 2]

        print("\nCurrent Solution Statistics:")
        print(f"Objective value: {objective:.2f}")
        print(f"Total cost: {total_cost:.2f}")
        print(f"Makespan (max time): {max_time:.2f} hours")
        print(f"Nodes served by trucks: {truck_served}")
        print(f"Nodes served by drones: {drone_served}")

        if truck_distances:
            print(f"Total truck distance: {sum(truck_distances):.2f}")
            print(f"Average truck route distance: {np.mean(truck_distances):.2f}")
            print(f"Max truck route distance: {max(truck_distances):.2f}")

        if drone_distances:
            print(f"Total drone distance: {sum(drone_distances):.2f}")
            print(f"Average drone route distance: {np.mean(drone_distances):.2f}")
            print(f"Max drone route distance: {max(drone_distances):.2f}")

        return objective, total_cost, max_time
    def print_detailed_routes(self, solution=None):
        """Print detailed information about each route in the solution"""
        if solution is None:
            solution = self.best_solution

        if solution is None:
            print("No solution available to print")
            return

        truck_routes, drone_routes = solution

        print("\n======= DETAILED ROUTE INFORMATION =======")

        for i, truck_route in enumerate(truck_routes):
            # Print truck route
            print(f"\nOptimized Route {i+1}:")
            print(f"Truck Route: {truck_route}")

            # Calculate truck metrics
            truck_distance = self.calculate_route_distance(truck_route)
            truck_time = self.calculate_route_time(truck_route)
            truck_cost = self.calculate_route_cost(truck_route)

            print(f"  Distance: {truck_distance:.2f} km")
            print(f"  Travel Time: {truck_time:.2f} hours")
            print(f"  Cost: {truck_cost:.2f}")

            # Print drone routes for this truck if available
            if i < len(drone_routes):
                print(f"Drone Routes:")

                for j, drone_list in enumerate(drone_routes[i]):
                    print(f"  Drone {j+1}:")

                    for k, drone_route in enumerate(drone_list):
                        if len(drone_route) >= 2:  # Only print non-empty routes
                            # Calculate drone metrics
                            drone_distance = self.calculate_route_distance(drone_route, is_drone=True)
                            drone_time = self.calculate_route_time(drone_route, is_drone=True)
                            drone_cost = self.calculate_route_cost(drone_route, is_drone=True)
                            drone_load = self.calculate_drone_load(drone_route)

                            print(f"    Route {k+1}: {drone_route}")
                            print(f"      Distance: {drone_distance:.2f} km")
                            print(f"      Travel Time: {drone_time:.2f} hours")
                            print(f"      Cost: {drone_cost:.2f}")
                            print(f"      Load: {drone_load} (Capacity: {self.drone_capacity})")

        # Print summary statistics
        objective, total_cost, max_time = self.calculate_solution_objective(truck_routes, drone_routes)
        print("\n======= SOLUTION SUMMARY =======")
        print(f"Total Objective Value: {objective:.2f}")
        print(f"Total Cost: {total_cost:.2f}")
        print(f"Maximum Operation Time (Makespan): {max_time:.2f} hours")

        # Count total nodes served
        truck_served = sum(len(route) - 2 for route in truck_routes)  # -2 for depot at start/end
        drone_served = sum(len(drone_route) - 2
                          for truck_drones in drone_routes
                          for drone_list in truck_drones
                          for drone_route in drone_list if len(drone_route) >= 2)

        print(f"Total Customers Served: {truck_served + drone_served}")
        print(f"  - By Truck: {truck_served}")
        print(f"  - By Drone: {drone_served}")
        print("======================================")

# Example usage with the provided input
if __name__ == "__main__":
    # Create sample data
    num_nodes = 33  # Including depot 0

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Generate node locations for calculating distances
    node_locations = np.random.rand(num_nodes, 2) * 100

    # Generate demand for each node (0 is depot)
    demands = {i: random.randint(10, 200) if i > 0 else 0 for i in range(num_nodes)}

    # Calculate distance matrices
    #euclidean_distance_matrix = pd.read_csv('euclidean_distances.csv', header=0, index_col=0).values
    euclidean_distance_matrix = pd.read_csv('distance_matrix.csv', header=0, index_col=0).values

    for i in range(num_nodes):
        for j in range(num_nodes):
            # Euclidean distance for drones
            dist = np.sqrt(np.sum((node_locations[i] - node_locations[j])**2))
            euclidean_distance_matrix[i, j] = dist

            # Road distance for trucks (with some detours)
        '''    detour_factor = random.uniform(1.1, 1.5)
            euclidean_distance_matrix[i, j] = dist * detour_factor'''

    # Parse input
    truck_routes_input = "[[0, 26, 5, 25, 30, 12, 0], [0, 2, 28, 11, 0], [0, 20, 32, 13, 8, 7, 0], [0, 15, 17, 9, 16, 18, 0], [0, 6, 19, 14, 21, 1, 29, 0]]"
    drone_routes_input = "[[[[26, 27, 25], [30, 10, 12]], []], [[[2, 22, 23, 28], [11, 24, 0]], []], [[[20, 4, 0]], []], [[[15, 3, 16]], []], [[[19, 31, 29]], []]]"


    # Create Truck-Drone Routing instance
    tdr = TruckDroneRouting(
        truck_routes=[],
        drone_routes=[],
        demands=demands,
        #euclidean_distance_matrix=euclidean_distance_matrix,
        euclidean_distance_matrix=euclidean_distance_matrix,
        max_iterations=500,
        p1=0.3,  # percentage of drone-only nodes to remove
        p2=0.3,  # percentage of truck-only nodes to remove
        p3=0.3,  # percentage of sub-drone routes to remove
        drone_capacity=9999,
        drone_cost_factor=2,
        #truck_speed=truck_speed,  # km/h
        #drone_speed=1.5*truck_speed,  # km/h
        time_weight=0.5,  # balancing cost and time
        force_use_drones=True,
        num_drones=5
    )

    # Process input
    tdr.process_input(truck_routes_input, drone_routes_input)

    # Evaluate initial solution
    print("Initial solution evaluation:")
    tdr.evaluate_current_solution()

    # Visualize initial solution
    plt_initial = tdr.visualize_solution((tdr.truck_routes, tdr.drone_routes), "Initial Solution")
    plt_initial.savefig("initial_solution.png")

    # Run optimization
    print("\nStarting optimization...")
    best_solution, best_objective = tdr.run_and_visualize()

    # Save final solution visualization
    plt_final = tdr.visualize_solution(best_solution, "Optimized Solution")
    plt_final.savefig("optimized_solution.png")

    print("\nOptimization completed successfully!")
