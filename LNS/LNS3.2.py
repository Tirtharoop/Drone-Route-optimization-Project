import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from itertools import permutations
import matplotlib.pyplot as plt
import re

class DroneTruckRoutingOptimizer:
    def __init__(self, vrp_filepath, image_filepath, drone_capacity=40, truck_capacity=100):
        self.vrp_filepath = vrp_filepath
        self.image_filepath = image_filepath
        self.drone_capacity = drone_capacity
        self.truck_capacity = truck_capacity
        self.node_coords, self.demands, self.depot = self.parse_vrp_file(vrp_filepath)

    def parse_vrp_file(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()

        lines = [line.strip() for line in lines]

        # Extract number of trucks from filename if available
        match = re.search(r'-k(\d+)', filepath)
        self.num_trucks = int(match.group(1)) if match else 1

        # Attempt to find section headers with common variations
        node_coord_start = next((i for i, line in enumerate(lines) if "NODE_COORD_SECTION" in line or "NODE_COORD" in line), None) + 1
        demand_start = next((i for i, line in enumerate(lines) if "DEMAND_SECTION" in line or "DEMAND" in line), None) + 1
        depot_start = next((i for i, line in enumerate(lines) if "DEPOT_SECTION" in line or "DEPOT" in line), None) + 1

        if node_coord_start is None or demand_start is None or depot_start is None:
            raise ValueError("One or more required sections are missing in the VRP file.")

        # Extract capacity
        capacity_line = next((line for line in lines if 'CAPACITY' in line), None)
        if capacity_line is None:
            raise ValueError("Capacity information is missing in the VRP file.")
        truck_capacity = int(capacity_line.split()[-1])

        # Get node coordinates
        node_coords = {}
        i = node_coord_start
        while i < len(lines) and not (lines[i].startswith("DEMAND") or lines[i].startswith("DEPOT")):
            parts = lines[i].split()
            node_id, x, y = int(parts[0]), float(parts[1]), float(parts[2])
            node_coords[node_id] = (x, y)
            i += 1

        # Get demands
        demands = {}
        i = demand_start
        while i < len(lines) and not lines[i].startswith("DEPOT"):
            parts = lines[i].split()
            node_id, demand = int(parts[0]), int(parts[1])
            demands[node_id] = demand
            i += 1

        # Get depot node
        depot_node = int(lines[depot_start].strip())

        return node_coords, demands, depot_node

    def cluster_nodes(self, n_clusters=5):
        # Create DataFrame excluding depot
        data = {
            'node': [],
            'x': [],
            'y': [],
            'demand': []
        }

        for node, (x, y) in self.node_coords.items():
            if node == self.depot:
                continue
            data['node'].append(node)
            data['x'].append(x)
            data['y'].append(y)
            data['demand'].append(self.demands[node])

        df = pd.DataFrame(data)

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(df[['x', 'y']])

        return df, kmeans

    def check_cluster_capacities(self, df):
        cluster_info = {}
        for c in df['cluster'].unique():
            cluster_df = df[df['cluster'] == c]
            total_demand = cluster_df['demand'].sum()

            assignment = 'drone' if total_demand <= self.drone_capacity else 'truck'

            cluster_info[c] = {
                'nodes': list(cluster_df['node']),
                'total_demand': total_demand,
                'assignment': assignment
            }
        return cluster_info

    def find_drone_candidates(self, cluster_info, df):
        drone_clusters = {cid for cid, info in cluster_info.items() if info['assignment'] == 'drone'}
        truck_clusters = {cid for cid, info in cluster_info.items() if info['assignment'] == 'truck'}

        reassignments = {}

        for t_cid in truck_clusters:
            t_nodes = cluster_info[t_cid]['nodes']
            t_df = df[df['node'].isin(t_nodes)]

            for idx, row in t_df.iterrows():
                node = row['node']
                demand = row['demand']

                if demand > self.drone_capacity:
                    continue  # cannot assign this node to any drone

                # Try assigning to a drone cluster that has enough remaining capacity
                for d_cid in drone_clusters:
                    d_current_demand = cluster_info[d_cid]['total_demand']
                    if d_current_demand + demand <= self.drone_capacity:
                        # Eligible for reassignment
                        reassignments[node] = {
                            'from_cluster': t_cid,
                            'to_cluster': d_cid,
                            'demand': demand
                        }

                        # Update cluster info
                        cluster_info[d_cid]['nodes'].append(node)
                        cluster_info[d_cid]['total_demand'] += demand

                        cluster_info[t_cid]['nodes'].remove(node)
                        cluster_info[t_cid]['total_demand'] -= demand

                        break  # move to next node

        return reassignments

    def assign_clusters_to_trucks(self, cluster_info):
        # Collect only truck-assigned clusters
        truck_clusters = [cid for cid, info in cluster_info.items() if info['assignment'] == 'truck']
        assigned = {i: [] for i in range(self.num_trucks)}

        # Round robin distribution
        for i, cid in enumerate(truck_clusters):
            assigned[i % self.num_trucks].append(cid)

        return assigned

    def best_drone_route(self, depot, customer_nodes, cost_matrix):
        best_route = []
        min_cost = float('inf')

        for perm in permutations(customer_nodes):
            route = [depot] + list(perm) + [depot]
            cost = sum(cost_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
            if cost < min_cost:
                min_cost = cost
                best_route = route

        return best_route, min_cost

    def group_close_customers_for_drones(self, cluster_nodes, demand_dict, drone_capacity, node_coords):
        """Group customers based on proximity and demand limit"""
        groups = []
        nodes = set(cluster_nodes)

        while nodes:
            current_group = []
            current_load = 0
            seed = nodes.pop()
            current_group.append(seed)
            current_load += demand_dict[seed]

            # Sort remaining nodes by distance to seed
            distances = [(n, self.euclidean(node_coords[seed], node_coords[n])) for n in nodes]
            distances.sort(key=lambda x: x[1])

            for neighbor, _ in distances:
                if current_load + demand_dict[neighbor] <= drone_capacity:
                    current_group.append(neighbor)
                    current_load += demand_dict[neighbor]
            # Remove added nodes
            for n in current_group[1:]:
                nodes.discard(n)

            groups.append(current_group)

        return groups

    def get_nearest_truck_node(self, delivery_node, truck_nodes, cost_matrix):
        return min(truck_nodes, key=lambda tn: cost_matrix[tn][delivery_node])

    def generate_drone_trips(self, cluster_info, df):
        drone_routes = []
        rejected_nodes = []

        # Identify truck-accessible nodes (truck routes will be built from these clusters)
        truck_nodes = []
        for cid, info in cluster_info.items():
            if info['assignment'] == 'truck':
                truck_nodes.extend(info['nodes'])

        # Create cost matrix
        cost_matrix = {i: {j: self.euclidean(self.node_coords[i], self.node_coords[j]) for j in self.node_coords} for i in self.node_coords}

        for cid, info in cluster_info.items():
            if info['assignment'] != 'drone':
                continue

            cluster_nodes = info['nodes']
            cluster_df = df[df['node'].isin(cluster_nodes)]

            # Try to build multiple trips from this cluster
            remaining_nodes = cluster_df.copy()

            while not remaining_nodes.empty:
                # Group customers for drones
                groups = self.group_close_customers_for_drones(
                    list(remaining_nodes['node']),
                    self.demands,
                    self.drone_capacity,
                    self.node_coords
                )

                for group in groups:
                    # Find the nearest truck node for launch and return
                    launch_point = self.get_nearest_truck_node(group[0], truck_nodes, cost_matrix)
                    return_point = self.get_nearest_truck_node(group[-1], truck_nodes, cost_matrix)

                    route, cost = self.best_drone_route(depot=launch_point, customer_nodes=group, cost_matrix=cost_matrix)
                    drone_routes.append({
                        'from': launch_point,
                        'to': return_point,
                        'nodes': route[1:-1],
                        'payload': sum(self.demands[node] for node in route[1:-1]),
                        'distance': cost
                    })
                    remaining_nodes = remaining_nodes[~remaining_nodes['node'].isin(group)]

                if remaining_nodes.empty:
                    break

            if not remaining_nodes.empty:
                rejected_nodes.extend(list(remaining_nodes['node']))

        return drone_routes, rejected_nodes

    def build_truck_route(self, cluster_info):
        truck_nodes = []
        for cid, info in cluster_info.items():
            if info['assignment'] == 'truck':
                truck_nodes.extend(info['nodes'])

        truck_route = [self.depot]
        visited = set([self.depot])
        current = self.depot
        nodes_left = set(truck_nodes)

        while nodes_left:
            next_node = min(nodes_left, key=lambda n: self.euclidean(self.node_coords[current], self.node_coords[n]))
            truck_route.append(next_node)
            visited.add(next_node)
            nodes_left.remove(next_node)
            current = next_node

        truck_route.append(self.depot)  # return to depot
        return truck_route

    def build_all_truck_routes(self, cluster_info, cluster_to_trucks):
        truck_routes = []
        for truck_id in range(self.num_trucks):
            route = [self.depot]
            visited = set([self.depot])

            # Gather nodes in clusters assigned to this truck
            truck_nodes = []
            for cid in cluster_to_trucks[truck_id]:
                truck_nodes.extend(cluster_info[cid]['nodes'])

            current = self.depot
            nodes_left = set(truck_nodes)

            while nodes_left:
                next_node = min(nodes_left, key=lambda n: self.euclidean(self.node_coords[current], self.node_coords[n]))
                route.append(next_node)
                visited.add(next_node)
                nodes_left.remove(next_node)
                current = next_node

            route.append(self.depot)
            truck_routes.append(route)

        return truck_routes

    def euclidean(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def drone_max_distance(self, payload):
        return 14815 / ((65 + payload) ** 1.5)

    def compute_total_cost(self, truck_routes, drone_routes):
        # Truck cost (distance)
        total_truck_cost = 0
        for route in truck_routes:
            for i in range(len(route) - 1):
                total_truck_cost += self.euclidean(self.node_coords[route[i]], self.node_coords[route[i+1]])

        # Drone cost (distance / speed) + waiting cost
        total_drone_cost = 0
        total_waiting_cost = 0
        for route in drone_routes:
            drone_cost = route['distance'] / 1.5
            total_drone_cost += drone_cost

            truck_time = self.euclidean(self.node_coords[route['from']], self.node_coords[route['to']]) / 1.0
            waiting_time = max(0, drone_cost - truck_time)
            total_waiting_cost += waiting_time  # 1 unit cost per unit waiting time

        total_cost = total_truck_cost + total_drone_cost + total_waiting_cost
        return total_truck_cost, total_drone_cost, total_cost

    def plot_all_routes(self, truck_routes, drone_routes):
        plt.figure(figsize=(14, 10))
        colors = plt.cm.get_cmap('tab10', len(truck_routes))

        # Plot truck routes
        for idx, truck_route in enumerate(truck_routes):
            x = [self.node_coords[n][0] for n in truck_route]
            y = [self.node_coords[n][1] for n in truck_route]
            plt.plot(x, y, color=colors(idx), linewidth=2, label=f'Truck {idx+1} Route')
            for n in truck_route:
                plt.annotate(str(n), self.node_coords[n], fontsize=8)

        # Plot drone routes
        for i, route in enumerate(drone_routes):
            x = [self.node_coords[n][0] for n in [route['from']] + route['nodes'] + [route['to']]]
            y = [self.node_coords[n][1] for n in [route['from']] + route['nodes'] + [route['to']]]
            plt.plot(x, y, linestyle='--', linewidth=1, label=f'Drone Trip {i+1}', alpha=0.6)

        # Mark the depot
        depot_x, depot_y = self.node_coords[self.depot]
        plt.scatter(depot_x, depot_y, color='black', s=200, marker='*', label='Depot')

        plt.title("Truck and Drone Routes (Multiple Trucks)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.legend()
        plt.show()

    def print_truck_routes_with_costs(self, truck_routes):
        print("\n=== TRUCK ROUTES ===")
        for idx, route in enumerate(truck_routes):
            print(f"Truck {idx + 1} Route:")
            total_cost = 0
            for i in range(len(route) - 1):
                n1, n2 = route[i], route[i + 1]
                segment_dist = self.euclidean(self.node_coords[n1], self.node_coords[n2])
                total_cost += segment_dist
                print(f"  {n1} -> {n2} | Distance: {segment_dist:.2f}")
            print(f"Total Distance for Truck {idx + 1}: {total_cost:.2f}\n")

    def print_drone_routes_with_costs(self, drone_routes):
        print("\n=== DRONE ROUTES ===")
        for idx, route in enumerate(drone_routes):
            print(f"Drone Trip {idx + 1}:")
            print(f"  Launch from: {route['from']}")
            print(f"  Deliver to nodes: {route['nodes']}")
            print(f"  Return to: {route['to']}")
            print(f"  Payload: {route['payload']}")
            print(f"  Distance: {route['distance']:.2f}")
            drone_cost = route['distance'] / 1.5
            print(f"  Drone Cost: {drone_cost:.2f}")

            # Truck waiting time calculation
            truck_dist = self.euclidean(self.node_coords[route['from']], self.node_coords[route['to']])
            truck_time = truck_dist / 1.0
            waiting_time = max(0, drone_cost - truck_time)
            print(f"  Truck Distance Between Launch and Return: {truck_dist:.2f}")
            print(f"  Truck Travel Time: {truck_time:.2f}")
            print(f"  Drone Travel Time: {drone_cost:.2f}")
            print(f"  🚚 Truck Waiting Time: {waiting_time:.2f}")

            # Print demands for each node in the route
            print("  Node Demands:")
            for node in route['nodes']:
                print(f"    Node {node}: Demand {self.demands[node]}")
            print()

    def plot_clusters(self, df, kmeans):
        plt.figure(figsize=(10, 8))
        colors = plt.cm.get_cmap('tab10', df['cluster'].nunique())

        for cluster_id in df['cluster'].unique():
            cluster_df = df[df['cluster'] == cluster_id]
            plt.scatter(cluster_df['x'], cluster_df['y'], label=f"Cluster {cluster_id}", s=60, color=colors(cluster_id))

            for _, row in cluster_df.iterrows():
                plt.text(row['x'] + 0.5, row['y'] + 0.5, str(row['node']), fontsize=8)

        # Plot depot
        depot_x, depot_y = self.node_coords[self.depot]
        plt.scatter(depot_x, depot_y, c='black', s=150, marker='*', label='Depot')

        plt.title("KMeans Clustering of Nodes")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.show()

    def run(self, n_clusters=10):
        df, kmeans = self.cluster_nodes(n_clusters)

        # Visualize clusters
        self.plot_clusters(df, kmeans)

        cluster_info = self.check_cluster_capacities(df)
        self.find_drone_candidates(cluster_info, df)

        cluster_to_trucks = self.assign_clusters_to_trucks(cluster_info)
        all_truck_routes = self.build_all_truck_routes(cluster_info, cluster_to_trucks)

        # Initial drone assignment
        drone_routes, rejected_nodes = self.generate_drone_trips(cluster_info, df)

        # Handle any missing nodes
        all_routed_nodes = set()
        for route in all_truck_routes:
            all_routed_nodes.update(route[1:-1])  # Exclude depot
        for trip in drone_routes:
            all_routed_nodes.update(trip['nodes'])

        all_node_ids = set(df['node'].tolist())
        missing_nodes = all_node_ids - all_routed_nodes

        # Add missing nodes via drone trips
        if missing_nodes:
            for node in missing_nodes:
                coord = self.node_coords[node]
                drone_routes.append({
                    'from': self.depot,
                    'to': self.depot,
                    'nodes': [node],
                    'payload': self.demands[node],
                    'distance': 2 * self.euclidean(self.node_coords[self.depot], coord)
                })

        # Cost computation
        total_truck_cost, total_drone_cost, total_cost = self.compute_total_cost(all_truck_routes, drone_routes)

        # Print detailed routes
        self.print_truck_routes_with_costs(all_truck_routes)
        self.print_drone_routes_with_costs(drone_routes)

        self.plot_all_routes(all_truck_routes, drone_routes)

        return total_truck_cost, total_drone_cost, total_cost

# Example usage
optimizer = DroneTruckRoutingOptimizer('A-n32-k5.vrp', 'download1.png')
truck_cost, drone_cost, total_cost = optimizer.run()
print(f"Truck Cost: {truck_cost}, Drone Cost: {drone_cost}, Total Cost: {total_cost}")
