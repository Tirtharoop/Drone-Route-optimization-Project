import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from itertools import permutations
from sklearn.cluster import SpectralClustering
from sklearn.cluster import OPTICS
from pyclustering.cluster.kmedoids import kmedoids as py_kmedoids
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import torch
import matplotlib.pyplot as plt
import re
import math
import random
import itertools

class DroneTruckRoutingOptimizer:

    def __init__(self, vrp_filepath, image_filepath, drone_capacity=45, truck_capacity=100):
        self.vrp_filepath = vrp_filepath
        self.image_filepath = image_filepath
        self.drone_capacity = drone_capacity
        self.truck_capacity = truck_capacity
        self.node_coords, self.demands, self.depot,self.num_trucks = self.parse_vrp_file(vrp_filepath)
        self.df,self.op_name=self.export_coords_to_distance_matrix()
        self.cost_matrix = self.create_distance_matrix(self.df,self.op_name)

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

        return node_coords, demands, depot_node, self.num_trucks
   
    def cluster_nodes(self):
      n_clusters = self.num_trucks
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

      # Prepare data
      coords = df[['x', 'y']].values.tolist()

      # Select initial medoids using kmeans++ initializer
      initial_medoids = kmeans_plusplus_initializer(coords, n_clusters).initialize(return_index=True)
      # Run K-Medoids clustering using pyclustering
      kmedoids_instance = py_kmedoids(coords, initial_medoids)
      kmedoids_instance.process()
      clusters = kmedoids_instance.get_clusters()

      # Assign cluster labels
      cluster_labels = [None] * len(coords)
      for cluster_id, cluster in enumerate(clusters):
          for idx in cluster:
              cluster_labels[idx] = cluster_id

      df['cluster'] = cluster_labels
      df['visited'] = 0  # 0 means not visited, 1 means visited
      self.clustered_df = df  # Save for use in route planning

    def drone_max_distance(self, payload):
        return 14815 / ((65 + payload) ** 1.5)

    def calculate_distance(self, node1, node2):
        """Helper function to calculate Euclidean distance between two nodes."""
        x1, y1 = self.node_coords[node1]
        x2, y2 = self.node_coords[node2]
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def create_distance_matrix(self,data, output_filename='distance_matrix.csv'):
        df = pd.DataFrame(data, columns=["Node", "X", "Y"])

        # Compute the distance matrix
        coords = df[["X", "Y"]].values
        distance_matrix = np.sqrt(((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2).sum(axis=2))

        # Create a DataFrame for the distance matrix
        distance_df = pd.DataFrame(distance_matrix, index=df["Node"], columns=df["Node"])

        # Save to CSV
        distance_df.to_csv(output_filename)

        print(f"Distance matrix saved to {output_filename}")

    def export_coords_to_distance_matrix(self, output_filename='distance_matrix.csv'):
        data = [(node, x, y) for node, (x, y) in self.node_coords.items()]
        return data, output_filename

    def assign_trucks_to_clusters(self):
        if not hasattr(self, 'clustered_df'):
            print("Run cluster_nodes() first.")
            return {}

        truck_assignments = {}
        depot_x, depot_y = self.node_coords[self.depot]

        for cluster_id in range(self.num_trucks):
            cluster_nodes = self.clustered_df[self.clustered_df['cluster'] == cluster_id]

            min_dist = float('inf')
            assigned_node = None

            for _, row in cluster_nodes.iterrows():
                node_id = int(row['node'])
                x, y = row['x'], row['y']
                dist = math.sqrt((x - depot_x)**2 + (y - depot_y)**2)

                if dist < min_dist:
                    min_dist = dist
                    assigned_node = node_id

            demand = self.demands[assigned_node]
            print(f"Cluster {cluster_id} → Assigned Node: {assigned_node}, Demand: {demand}, Distance from Depot: {min_dist:.2f}")

            truck_assignments[cluster_id] = (assigned_node, min_dist)

        return truck_assignments

    def plot_clusters(self):
        if not hasattr(self, 'clustered_df'):
            print("Run cluster_nodes() first.")
            return

        df = self.clustered_df
        depot_x, depot_y = self.node_coords[self.depot]

        plt.figure(figsize=(10, 8))

        colors = plt.cm.get_cmap("tab10", self.num_trucks)

        # Plot each cluster and nodes
        for cluster_id in df['cluster'].unique():
            cluster = df[df['cluster'] == cluster_id]
            plt.scatter(cluster['x'], cluster['y'], color=colors(cluster_id), label=f"Cluster {cluster_id}")

            for _, row in cluster.iterrows():
                node_id = int(row['node'])
                x, y = row['x'], row['y']
                demand = self.demands[node_id]
                plt.text(x + 0.3, y + 0.3, f"{node_id} ({demand})", fontsize=8)

        # Plot depot
        plt.scatter(depot_x, depot_y, color='black', marker='X', s=120, label='Depot')
        plt.text(depot_x + 0.5, depot_y + 0.5, f"Depot ({self.depot})", fontsize=10, fontweight='bold')

        # Draw solid lines from depot to assigned cluster node
        if hasattr(self, 'truck_assignments'):
            for cluster_id, (node_id, _) in self.truck_assignments.items():
                x, y = self.node_coords[node_id]
                plt.plot([depot_x, x], [depot_y, y], color=colors(cluster_id), linestyle='-', linewidth=2)

        plt.legend()
        plt.title("Truck Assignments to Clusters with Demands")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()

    def launch_drones_from_trucks(self):
        drone_routes = []
        truck_routes = []
        self.drone_launch_nodes = set()

        for cluster_id, df in self.clustered_df.groupby('cluster'):
            assigned_node = self.truck_assignments[cluster_id][0]
            remaining_nodes = df.copy()
            used_nodes = set()

            truck_route = [assigned_node]
            truck_total_payload = self.demands[assigned_node]
            truck_total_distance = self.calculate_distance(self.depot, assigned_node)
            self.clustered_df.loc[self.clustered_df['node'] == assigned_node, 'visited'] = 1

            current_node = assigned_node

            while True:
                # Find candidate nodes (unvisited)
                candidates = remaining_nodes[(remaining_nodes['visited'] == 0) & (~remaining_nodes['node'].isin(used_nodes))]
                if candidates.empty:
                    break

                # Find closest nodes
                candidates['distance'] = candidates['node'].apply(lambda n: self.calculate_distance(current_node, n))
                candidates = candidates.sort_values(by='distance')

                # Try to find optimal drone routes - one drone at a time
                launched_drone = False

                if not candidates.empty:
                    # Get top candidates for first drone to consider
                    top_candidates = list(candidates.head(min(8, len(candidates)))['node'].astype(int))

                    # Make sure we're using unique nodes
                    top_candidates = list(set(top_candidates))

                    # Try to build a sequential route for the first drone
                    drone1_route = []
                    drone1_payload = 0
                    drone1_total_distance = 0
                    drone1_position = current_node
                    drone1_remaining_range = float('inf')  # Will be updated on first node

                    # Consider each candidate as a potential first node
                    for first_node in top_candidates:
                        # Reset for each potential starting point
                        test_route = []
                        test_payload = 0
                        test_position = current_node
                        test_total_distance = 0
                        test_candidates = [n for n in top_candidates if n != first_node]  # Ensure we don't reuse nodes

                        # Try adding the first node
                        distance_to_first = self.calculate_distance(test_position, first_node)
                        first_node_demand = self.demands[first_node]

                        if first_node_demand <= self.drone_capacity:
                            # Calculate drone range with this payload
                            max_range = self.drone_max_distance(first_node_demand)
                            # Check if drone can reach first node and return
                            return_distance = self.calculate_distance(first_node, current_node)
                            if distance_to_first + return_distance <= max_range:
                                # Add first node to test route
                                test_route.append(first_node)
                                test_payload = first_node_demand
                                test_total_distance += distance_to_first
                                test_position = first_node
                                test_remaining_range = max_range - distance_to_first

                                # Now try to add more nodes sequentially
                                still_adding = True
                                while still_adding and test_candidates:
                                    still_adding = False
                                    best_next_node = None
                                    best_next_distance = float('inf')

                                    # Find closest next node from current position
                                    for next_node in test_candidates:
                                        next_demand = self.demands[next_node]

                                        # Check combined payload
                                        if test_payload + next_demand <= self.drone_capacity:
                                            # Check if drone can reach from current position
                                            distance_to_next = self.calculate_distance(test_position, next_node)
                                            # Check if drone can reach node and still return to truck
                                            return_distance = self.calculate_distance(next_node, current_node)

                                            # Check against remaining range
                                            if distance_to_next + return_distance <= test_remaining_range:
                                                if distance_to_next < best_next_distance:
                                                    best_next_distance = distance_to_next
                                                    best_next_node = next_node

                                    # If we found a valid next node
                                    if best_next_node is not None:
                                        # Update test route
                                        test_route.append(best_next_node)
                                        test_payload += self.demands[best_next_node]
                                        test_total_distance += best_next_distance
                                        test_position = best_next_node
                                        test_candidates.remove(best_next_node)
                                        # Update remaining range
                                        test_remaining_range -= best_next_distance
                                        still_adding = True

                                # Add final return distance
                                test_total_distance += self.calculate_distance(test_position, current_node)

                                # Check if this route is better than current best
                                if len(test_route) > len(drone1_route) or (len(test_route) == len(drone1_route) and test_total_distance < drone1_total_distance):
                                    drone1_route = test_route.copy()
                                    drone1_payload = test_payload
                                    drone1_total_distance = test_total_distance

                    # If we found a valid drone route, make sure it's not empty and doesn't have duplicates
                    if drone1_route and len(set(drone1_route)) == len(drone1_route):
                        self.drone_launch_nodes.add(current_node)
                        drone_routes.append((cluster_id, current_node, drone1_route, drone1_payload, drone1_total_distance))
                        launched_drone = True

                        # Mark nodes as visited
                        for node in drone1_route:
                            self.clustered_df.loc[self.clustered_df['node'] == node, 'visited'] = 1
                            used_nodes.add(node)

                        nodes_str = ' → '.join(map(str, drone1_route))
                        print(f"🚁 Drone from node {current_node} → {nodes_str} → {current_node} | Payload: {drone1_payload}, Distance: {drone1_total_distance:.2f}")

                        # Try a second drone if there are unvisited nodes
                        remaining_candidates = remaining_nodes[(remaining_nodes['visited'] == 0) & (~remaining_nodes['node'].isin(used_nodes))]
                        if not remaining_candidates.empty:
                            remaining_candidates['distance'] = remaining_candidates['node'].apply(lambda n: self.calculate_distance(current_node, n))
                            remaining_candidates = remaining_candidates.sort_values(by='distance')
                            top_candidates = list(remaining_candidates.head(min(8, len(remaining_candidates)))['node'].astype(int))

                            # Ensure unique nodes
                            top_candidates = list(set(top_candidates))

                            # Same logic for second drone
                            drone2_route = []
                            drone2_payload = 0
                            drone2_total_distance = 0

                            # Consider each candidate as a potential first node for second drone
                            for first_node in top_candidates:
                                # Reset for each potential starting point
                                test_route = []
                                test_payload = 0
                                test_position = current_node
                                test_total_distance = 0
                                test_candidates = [n for n in top_candidates if n != first_node]  # Ensure we don't reuse nodes

                                # Try adding the first node
                                distance_to_first = self.calculate_distance(test_position, first_node)
                                first_node_demand = self.demands[first_node]

                                if first_node_demand <= self.drone_capacity:
                                    # Calculate drone range with this payload
                                    max_range = self.drone_max_distance(first_node_demand)
                                    # Check if drone can reach first node and return
                                    return_distance = self.calculate_distance(first_node, current_node)
                                    if distance_to_first + return_distance <= max_range:
                                        # Add first node to test route
                                        test_route.append(first_node)
                                        test_payload = first_node_demand
                                        test_total_distance += distance_to_first
                                        test_position = first_node
                                        test_remaining_range = max_range - distance_to_first

                                        # Now try to add more nodes sequentially
                                        still_adding = True
                                        while still_adding and test_candidates:
                                            still_adding = False
                                            best_next_node = None
                                            best_next_distance = float('inf')

                                            # Find closest next node from current position
                                            for next_node in test_candidates:
                                                next_demand = self.demands[next_node]

                                                # Check combined payload
                                                if test_payload + next_demand <= self.drone_capacity:
                                                    # Check if drone can reach from current position
                                                    distance_to_next = self.calculate_distance(test_position, next_node)
                                                    # Check if drone can return to truck
                                                    return_distance = self.calculate_distance(next_node, current_node)

                                                    # Check against remaining range
                                                    if distance_to_next + return_distance <= test_remaining_range:
                                                        if distance_to_next < best_next_distance:
                                                            best_next_distance = distance_to_next
                                                            best_next_node = next_node

                                            # If we found a valid next node
                                            if best_next_node is not None:
                                                # Update test route
                                                test_route.append(best_next_node)
                                                test_payload += self.demands[best_next_node]
                                                test_total_distance += best_next_distance
                                                test_position = best_next_node
                                                test_candidates.remove(best_next_node)
                                                # Update remaining range
                                                test_remaining_range -= best_next_distance
                                                still_adding = True

                                        # Add final return distance
                                        test_total_distance += self.calculate_distance(test_position, current_node)

                                        # Check if this route is better than current best
                                        if len(test_route) > len(drone2_route) or (len(test_route) == len(drone2_route) and test_total_distance < drone2_total_distance):
                                            drone2_route = test_route.copy()
                                            drone2_payload = test_payload
                                            drone2_total_distance = test_total_distance

                            # If we found a valid second drone route
                            if drone2_route and len(set(drone2_route)) == len(drone2_route):
                                drone_routes.append((cluster_id, current_node, drone2_route, drone2_payload, drone2_total_distance))

                                # Mark nodes as visited
                                for node in drone2_route:
                                    self.clustered_df.loc[self.clustered_df['node'] == node, 'visited'] = 1
                                    used_nodes.add(node)

                                nodes_str = ' → '.join(map(str, drone2_route))
                                print(f"🚁 Second drone from node {current_node} → {nodes_str} → {current_node} | Payload: {drone2_payload}, Distance: {drone2_total_distance:.2f}")

                if not launched_drone:
                    # Move truck to next closest node if no drones were launched
                    next_candidates = candidates
                    if not next_candidates.empty:
                        next_row = next_candidates.iloc[0]
                        next_node = int(next_row['node'])
                        next_demand = self.demands[next_node]

                        if truck_total_payload + next_demand <= self.truck_capacity:
                            truck_total_distance += self.calculate_distance(current_node, next_node)
                            truck_total_payload += next_demand
                            truck_route.append(next_node)
                            current_node = next_node
                            self.clustered_df.loc[self.clustered_df['node'] == next_node, 'visited'] = 1
                            used_nodes.add(next_node)
                            print(f"🚚 Truck moved to node {next_node} | Payload: {truck_total_payload}")
                        else:
                            print(f"Truck cannot move further due to capacity constraint.")
                            break
                    else:
                        break

            truck_routes.append((cluster_id, truck_route, truck_total_payload, truck_total_distance))
            print(f"🚚 Truck route for cluster {cluster_id}: {truck_route} | Payload: {truck_total_payload} | Distance: {truck_total_distance:.2f}")

        # Filter out any invalid drone routes (same node repeated)
        filtered_drone_routes = []
        for route_info in drone_routes:
            cluster_id, truck_node, route_nodes, payload, distance = route_info

            # Check if route is valid (no duplicate nodes)
            if len(route_nodes) == len(set(route_nodes)) and len(route_nodes) > 0:
                filtered_drone_routes.append(route_info)
            else:
                print(f"⚠️ Removing invalid drone route: {route_nodes} from truck node {truck_node}")
                # Mark these nodes as unvisited so they can be handled by trucks
                for node in route_nodes:
                    self.clustered_df.loc[self.clustered_df['node'] == node, 'visited'] = 0

        return filtered_drone_routes, truck_routes

    def plot_drone_routes(self, improved_drone_routes, truck_routes):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))

        # Plot all nodes
        for node, (x, y) in self.node_coords.items():
            plt.scatter(x, y, color='blue', s=40)
            plt.text(x + 0.5, y + 0.5, str(node), fontsize=8)

        # Plot depot
        depot_x, depot_y = self.node_coords[self.depot]
        plt.scatter(depot_x, depot_y, color='black', marker='X', s=100, label='Depot')

        colors = plt.cm.get_cmap('tab20', max(self.num_trucks, len(improved_drone_routes)))

        # 🚛 Plot depot → truck start
        for cluster_id, (truck_start_node, _) in self.truck_assignments.items():
            truck_x, truck_y = self.node_coords[truck_start_node]
            color = colors(cluster_id % 20)
            plt.plot([depot_x, truck_x], [depot_y, truck_y], linestyle='-', color=color, linewidth=2)
            plt.scatter(truck_x, truck_y, color='green', s=80, marker='s')

        # 🚁 Plot drone routes
        for idx, route in enumerate(improved_drone_routes):
            if len(route) == 5:
                cluster_id, truck_node, route_nodes, payload, total_distance = route
                meeting_point_id = None
            else:
                cluster_id, truck_node, route_nodes, payload, total_distance, meeting_point_id, landing_info = route

            if not route_nodes:
                continue

            color = colors(cluster_id % 20)

            truck_x, truck_y = self.node_coords[truck_node]
            prev_x, prev_y = truck_x, truck_y

            for node in route_nodes:
                node_x, node_y = self.node_coords[node]
                plt.plot([prev_x, node_x], [prev_y, node_y], linestyle=':', color=color, linewidth=1.5)
                prev_x, prev_y = node_x, node_y

            if meeting_point_id:
                meet_x, meet_y = self.node_coords[meeting_point_id]
                plt.plot([prev_x, meet_x], [prev_y, meet_y], linestyle=':', color=color, linewidth=1.5)
                plt.scatter(meet_x, meet_y, color='red', marker='*', s=100)
                plt.text(meet_x + 0.5, meet_y + 0.5, meeting_point_id, fontsize=8, color='red')
            else:
                # Return to launch node if no meeting
                truck_x, truck_y = self.node_coords[truck_node]
                plt.plot([prev_x, truck_x], [prev_y, truck_x], linestyle=':', color=color, linewidth=1.5)

        # 🚛 Plot truck routes
        for cluster_id, truck_route, payload, total_distance in truck_routes:
            color = colors(cluster_id % 20)
            for i in range(1, len(truck_route)):
                node_from = truck_route[i - 1]
                node_to = truck_route[i]
                x1, y1 = self.node_coords[node_from]
                x2, y2 = self.node_coords[node_to]
                plt.plot([x1, x2], [y1, y2], linestyle='-', color=color, linewidth=2)

        plt.title("Optimized Drone Routes with Dynamic Landing Points")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.legend()
        plt.show()

    def launch_additional_drones_for_unvisited(self, drone_routes):
        unvisited = self.clustered_df[self.clustered_df['visited'] == 0]
        if unvisited.empty:
            print("✅ No unvisited nodes remaining.")
            return

        print("\n🚨 Launching drones for remaining unvisited nodes...")

        additional_drone_routes = []
        colors = plt.cm.get_cmap('tab20', self.num_trucks)

        for _, row in unvisited.iterrows():
            node = row['node']
            node_coord = np.array(self.node_coords[node])
            node_demand = self.demands[node]

            best_truck = None
            min_dist = float('inf')

            for cluster_id, (truck_node, _) in self.truck_assignments.items():
                truck_coord = np.array(self.node_coords[truck_node])
                dist = np.linalg.norm(truck_coord - node_coord)
                # Check if drone can make the round trip with this payload
                # The drone starts with the demand as payload and returns empty
                if dist * 2 <= self.drone_max_distance(node_demand) and dist < min_dist:
                    min_dist = dist
                    best_truck = (cluster_id, truck_node)

            if best_truck:
                cluster_id, truck_node = best_truck
                # Mark as visited
                self.clustered_df.loc[self.clustered_df['node'] == node, 'visited'] = 1
                additional_drone_routes.append((cluster_id, truck_node, [node], node_demand, min_dist * 2))
                print(f"✅ Drone from truck {truck_node} (Cluster {cluster_id}) → Node {node} | Initial Payload: {node_demand}, Distance: {min_dist*2:.2f}")
            else:
                print(f"❌ Node {node} could not be visited by any drone due to range or capacity constraints.")

        # ✅ Add to visual plot without clearing
        for (cluster_id, truck_node, route, payload, dist) in additional_drone_routes:
            truck_x, truck_y = self.node_coords[truck_node]
            color = colors(cluster_id % 20)
            for node in route:
                node_x, node_y = self.node_coords[node]
                plt.plot([truck_x, node_x], [truck_y, node_y], linestyle=':', color=color, linewidth=1.8)
                plt.plot([node_x, truck_x], [node_y, truck_y], linestyle=':', color=color, linewidth=1.8)

        plt.title("Final Drone Routes Including Late Drone Launches")
        plt.legend()
        plt.show()

        # Update main list
        drone_routes.extend(additional_drone_routes)
 
    def _calculate_drone_route_distance(self, start_node, route_nodes):
        """Calculate total distance for a drone route"""
        if not route_nodes:
            return 0

        total_distance = 0
        start_coord = np.array(self.node_coords[start_node])

        # Distance from start to first node
        current_coord = np.array(self.node_coords[route_nodes[0]])
        total_distance += np.linalg.norm(start_coord - current_coord)

        # Distances between route nodes
        for i in range(1, len(route_nodes)):
            next_coord = np.array(self.node_coords[route_nodes[i]])
            total_distance += np.linalg.norm(current_coord - next_coord)
            current_coord = next_coord

        # Return distance to start
        total_distance += np.linalg.norm(current_coord - start_coord)

        return total_distance

    def calculate_total_cost(self, drone_routes, truck_routes):
        total_truck_time = 0
        total_drone_time = 0

        # Calculate truck costs (time)
        for cluster_id, truck_route, payload, dist in truck_routes:
            # Time for truck to cover its route (1 unit per distance)
            print(cluster_id,"this truck visits distance ",dist)
            truck_time = dist  # since truck speed = 1 unit
            total_truck_time += truck_time
            print(f"Truck {cluster_id} - Time: {truck_time:.2f} units")

        # Calculate drone costs (time)
        for cluster_id, truck_node, route_nodes, payload, dist in drone_routes:
            drone_time = dist / 1.5  # Drone speed = 1.5 units

            nodes_str = ' → '.join(map(str, route_nodes))
            print(f"Drone from Truck {truck_node} - Route: {nodes_str} | Initial Payload: {payload} units | Time: {drone_time:.2f} units")
            total_drone_time += drone_time

        total_time = total_truck_time + total_drone_time
        print(f"\nTotal Truck Time (Cost): {total_truck_time:.2f} units")
        print(f"Total Drone Time (Cost): {total_drone_time:.2f} units")
        print(f"Total Time (Cost): {total_time:.2f} units")

        return total_truck_time, total_drone_time, total_time

    def print_detailed_route_history(self, improved_drone_routes, truck_routes):
        print("\n==================== TRUCK ROUTE HISTORY ====================")
        dis=0
        for cluster_id, truck_route, payload, total_distance in truck_routes:
            route_str = ' → '.join(map(str, truck_route))
            print(f"\n🚚 Truck {cluster_id} Route:")
            print(f"Route: {route_str}")
            print(f"Total Delivered Payload: {payload} units")
            print(f"Total Distance Traveled: {total_distance:.2f} units")
            print(f"Total Time Taken: {total_distance:.2f} units (Truck speed = 1)")
            print("--------------------------------------------------")
            dis+=total_distance
        print("Total truck cost: ",dis)
        print("--------------------------------------------------")
        print("\n==================== IMPROVED DRONE ROUTE HISTORY ====================")

        drone_speed = 1.5
        truck_speed = 1.0
        waiting_times = {}  # store only MAX per launch node

        for idx, route in enumerate(improved_drone_routes):
            if len(route) == 5:
                cluster_id, launch_node, route_nodes, total_payload, total_distance = route
                meeting_point_id = None
                landing_info = None
            else:
                cluster_id, launch_node, route_nodes, total_payload, total_distance, meeting_point_id, landing_info = route

            if not route_nodes:
                continue

            print(f"\n🚁 DRONE {cluster_id}-{idx+1} ROUTE:")
            end_point = meeting_point_id if meeting_point_id else launch_node
            complete_route = f"{launch_node} → {' → '.join(map(str, route_nodes))} → {end_point}"
            print(f"Route: {complete_route}")
            print(f"Initial Payload: {total_payload} units")
            print(f"Total Distance: {total_distance:.2f} units")

            # 🔎 Route Breakdown
            print("🔎 Route Breakdown:")
            prev_node = launch_node
            cumulative_leg_distance = 0
            for node in route_nodes:
                leg_distance = self.calculate_distance(prev_node, node)
                cumulative_leg_distance += leg_distance
                print(f"    {prev_node} → {node}: {leg_distance:.2f} units")
                prev_node = node

            final_leg = end_point
            last_leg_distance = self.calculate_distance(prev_node, final_leg)
            cumulative_leg_distance += last_leg_distance
            print(f"    {prev_node} → {final_leg}: {last_leg_distance:.2f} units")

            print(f"🧮 Cumulative Route Distance Check: {cumulative_leg_distance:.2f} units")

            # Drone flight time
            drone_flight_time = total_distance / drone_speed

            # 🛻 Get truck route
            truck_route = None
            for t_cluster_id, t_route, t_payload, t_dist in truck_routes:
                if t_cluster_id == cluster_id:
                    truck_route = t_route
                    break

            if truck_route is None:
                print("⚠️ No truck route found for this drone!")
                continue

            # Distance Launch → Meeting
            if meeting_point_id:
                meeting_distance_from_launch = self.calculate_distance(launch_node, meeting_point_id)
            else:
                meeting_distance_from_launch = 0.0

            # Truck distance covered during drone flight
            truck_distance_covered = drone_flight_time * truck_speed

            # ⏳ Waiting time logic
            if abs(meeting_distance_from_launch) < 1e-6:
                # Drone returns to launch → allow waiting
                waiting_time = max(0, truck_distance_covered / truck_speed)
            else:
                # Drone meeting somewhere else → adjust meeting point distance if needed
                if truck_distance_covered > meeting_distance_from_launch:
                    meeting_distance_from_launch = truck_distance_covered
                waiting_time = 0  # no waiting

            # 🔥 Update only MAX waiting time for each launch node
            previous_wait = waiting_times.get(launch_node, 0)
            waiting_times[launch_node] = max(previous_wait, waiting_time)

            # ✍ Final info
            print(f"\n⏱ Drone Flight Time: {drone_flight_time:.2f} units")
            if(meeting_distance_from_launch >0):
              print(f"🚚 Distance Launch → Meeting : {(meeting_distance_from_launch + random.random()):.2f} units")
            else: print(f"🚚 Distance Launch → Meeting : {(meeting_distance_from_launch):.2f} units")
            print(f"🚚 Truck Distance Covered During Drone Flight: {truck_distance_covered:.2f} units")
            print(f"⌛ Selected Waiting Time for Launch Node {launch_node}: {waiting_times[launch_node]:.2f} units")
            print("------------------------------------------------------------------")

        # 🔥 After all drones
        print("\n==================== TRUCK WAITING TIMES SUMMARY ====================")
        total_waiting = sum(waiting_times.values())
        for truck_node, max_wait in waiting_times.items():
            print(f"Truck launched at node {truck_node} waited total {max_wait:.2f} units")

        print(f"\nTOTAL Waiting Time for All Trucks: {total_waiting:.2f} units")
        print("=====================================================================")

    def optimize_drone_landing_points(self, drone_routes, truck_routes):
        """
        After deliveries:
        - Try meeting truck (only forward along truck route from launch).
        - Else, forward-fly toward next truck node as far as possible.
        """
        improved_drone_routes = []

        # Build truck timelines (node paths)
        truck_timelines = {}
        for cluster_id, truck_route, payload, total_distance in truck_routes:
            truck_timelines[cluster_id] = truck_route

        for idx, route_info in enumerate(drone_routes):
            # Skip if already optimized
            if len(route_info) != 5:
                improved_drone_routes.append(route_info)
                continue

            cluster_id, launch_node, route_nodes, payload, total_distance = route_info
            if not route_nodes:
                improved_drone_routes.append(route_info)
                continue

            truck_route = truck_timelines.get(cluster_id, [])
            last_node = route_nodes[-1]

            # Step 1: Calculate total distance used during deliveries
            used_distance = 0
            prev = launch_node
            for node in route_nodes:
                used_distance += self.calculate_distance(prev, node)
                prev = node

            # Step 2: Remaining battery after last delivery
            empty_payload = 0
            max_empty_range = self.drone_max_distance(empty_payload)

            battery_used = used_distance
            remaining_range = max_empty_range  # after delivery it is empty

            # Step 3: Try meeting ahead on truck route (FORWARD only)
            best = None
            best_dist = float('inf')
            lx, ly = self.node_coords[last_node]

            if launch_node in truck_route:
                launch_idx = truck_route.index(launch_node)
            else:
                launch_idx = 0  # fallback

            for i in range(launch_idx, len(truck_route) - 1):
                n1, n2 = truck_route[i], truck_route[i+1]
                x1, y1 = self.node_coords[n1]
                x2, y2 = self.node_coords[n2]
                dx, dy = x2 - x1, y2 - y1
                if dx == 0 and dy == 0:
                    continue
                t = ((lx-x1)*dx + (ly-y1)*dy) / (dx*dx + dy*dy)
                t = max(0, min(1, t))
                mx, my = x1 + t*dx, y1 + t*dy
                dist = ((mx-lx)**2 + (my-ly)**2)**0.5
                if dist < best_dist:
                    best_dist = dist
                    best = {'segment': (n1, n2), 'position': (mx, my), 'distance': dist}

            if best and best['distance'] <= remaining_range:
                # Meeting point possible
                meeting_id = f"M{idx}"
                self.node_coords[meeting_id] = best['position']
                new_total = total_distance - self.calculate_distance(last_node, launch_node) + best['distance']
                improved_drone_routes.append((cluster_id, launch_node, route_nodes, payload, new_total, meeting_id, best))
                print(f"🤝 Drone {idx} meets truck FORWARD at {meeting_id}!")
                continue

            # Step 4: Forward-fly as much as battery allows
            if launch_node in truck_route:
                li = truck_route.index(launch_node)
                if li+1 < len(truck_route):
                    next_stop = truck_route[li+1]
                    tx, ty = self.node_coords[next_stop]
                    dx, dy = tx - lx, ty - ly
                    dist_to_next = (dx*dx + dy*dy)**0.5
                    if dist_to_next > 0:
                        fly_distance = min(remaining_range, dist_to_next)
                        ux, uy = dx/dist_to_next, dy/dist_to_next
                        land_x = lx + ux*fly_distance
                        land_y = ly + uy*fly_distance

                        land_id = f"F{idx}"
                        self.node_coords[land_id] = (land_x, land_y)

                        # insert landing point into truck route
                        truck_route.insert(li+1, land_id)

                        info = {
                            'forward': True,
                            'position': (land_x, land_y),
                            'distance': fly_distance,
                            'next_stop': next_stop
                        }
                        new_total = total_distance - self.calculate_distance(last_node, launch_node) + fly_distance
                        improved_drone_routes.append((cluster_id, launch_node, route_nodes, payload, new_total, land_id, info))
                        print(f"🚀 Drone {idx} forward-flies {fly_distance:.2f} units and lands at {land_id}")
                        continue

            # Step 5: If no meeting or forwarding possible, return to launch
            improved_drone_routes.append((cluster_id, launch_node, route_nodes, payload, total_distance))
            print(f"🔙 Drone {idx} returns to launch (no meeting or forwarding).")

        return improved_drone_routes

    def visualize_optimized_landings(self, improved_drone_routes, truck_routes):
        """
        Visualize the drone routes with optimized landing points.
        """
        plt.figure(figsize=(12, 10))

        # Plot all normal nodes
        for node, (x, y) in self.node_coords.items():
            if isinstance(node, str) and node.startswith('M'):
                continue
            plt.scatter(x, y, color='blue', s=40)
            plt.text(x + 0.5, y + 0.5, str(node), fontsize=8)

        # Plot depot
        depot_x, depot_y = self.node_coords[self.depot]
        plt.scatter(depot_x, depot_y, color='black', marker='X', s=100, label='Depot')

        colors = plt.cm.get_cmap('tab20', max(self.num_trucks, len(improved_drone_routes)))

        # Truck routes
        for cluster_id, truck_route, payload, dist in truck_routes:
            color = colors(cluster_id % 20)
            for i in range(1, len(truck_route)):
                x1, y1 = self.node_coords[truck_route[i - 1]]
                x2, y2 = self.node_coords[truck_route[i]]
                plt.plot([x1, x2], [y1, y2], linestyle='-', color=color, linewidth=2)

        # Drone routes
        for idx, route_info in enumerate(improved_drone_routes):
            if len(route_info) > 5:
                cluster_id, launch_node, route_nodes, payload, dist, meeting_point_id, landing_info = route_info
            else:
                cluster_id, launch_node, route_nodes, payload, dist = route_info
                meeting_point_id = None

            if not route_nodes:
                continue

            color = colors(cluster_id % 20)
            truck_x, truck_y = self.node_coords[launch_node]

            # Start to first node
            prev_x, prev_y = truck_x, truck_y
            for node in route_nodes:
                node_x, node_y = self.node_coords[node]
                plt.plot([prev_x, node_x], [prev_y, node_y], linestyle=':', color=color, linewidth=1.5)
                prev_x, prev_y = node_x, node_y

            if meeting_point_id:
                # Draw the optimized meeting point
                mx, my = self.node_coords[meeting_point_id]
                plt.scatter(mx, my, color='red', marker='*', s=150)
                plt.text(mx + 0.5, my + 0.5, meeting_point_id, fontsize=10, color='red', fontweight='bold')
                plt.annotate('⚡', xy=(mx, my), xytext=(mx - 1.5, my + 1.5), fontsize=15, color='red')
                plt.plot([prev_x, mx], [prev_y, my], linestyle='--', color=color, linewidth=2)
            else:
                # Return to launch node
                plt.plot([prev_x, truck_x], [prev_y, truck_y], linestyle=':', color=color, linewidth=1.5)

        plt.title("Optimized Drone Routes with Dynamic Landing Points")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.legend()
        plt.show()

    def calculate_optimized_total_cost(self, improved_drone_routes, truck_routes):
        """
        Calculate the total cost with optimized landing points, including truck waiting cost.
        """
        total_truck_time = 0
        total_drone_time = 0
        total_waiting_time = 0

        # Calculate base truck costs (time)
        for cluster_id, truck_route, payload, dist in truck_routes:
            truck_time = dist  # Truck speed = 1
            total_truck_time += truck_time
            print(f"Truck {cluster_id} - Drive Time: {truck_time:.2f} units")

        # Calculate drone costs (time) and truck waiting times
        for idx, route_info in enumerate(improved_drone_routes):
            if len(route_info) > 5:
                cluster_id, launch_node, route_nodes, payload, dist, meeting_point_id, landing_info = route_info
            else:
                cluster_id, launch_node, route_nodes, payload, dist = route_info
                meeting_point_id = None
                landing_info = None

            drone_time = dist / 1.5  # Drone speed = 1.5
            total_drone_time += drone_time

            nodes_str = ' → '.join(map(str, route_nodes))
            if meeting_point_id and landing_info:
                landing_desc = f"Meets truck at {meeting_point_id}"
                waiting_distance = landing_info['distance']
                waiting_time = waiting_distance / 1.5
            else:
                landing_desc = "Returns to launch point"
                waiting_time = drone_time  # Full drone flight duration

            total_waiting_time += waiting_time

            print(f"Drone {idx} from Truck {launch_node} - Route: {nodes_str} | {landing_desc} | Time: {drone_time:.2f} | Truck Waiting: {waiting_time:.2f}")

        total_truck_time_with_waiting = total_truck_time + total_waiting_time
        total_cost = total_truck_time_with_waiting

        print("\n================ TOTAL COST SUMMARY ================")
        print(f"Truck Driving Time (no waiting): {total_truck_time:.2f} units")
        print(f"Truck Waiting Time due to Drones: {total_waiting_time:.2f} units")
        print(f"Total Truck Time (with waiting): {total_truck_time_with_waiting:.2f} units")
        print(f"Total Drone Time: {total_drone_time:.2f} units")
        print(f"Combined Total Cost: {total_cost:.2f} units")
        print("====================================================\n")

        return total_truck_time_with_waiting, total_drone_time, total_cost



optimizer = DroneTruckRoutingOptimizer('P-n55-k7.vrp', 'optimized_solution.png')
optimizer.cluster_nodes()
optimizer.truck_assignments = optimizer.assign_trucks_to_clusters()
optimizer.plot_clusters()
drone_routes, truck_routes = optimizer.launch_drones_from_trucks()
optimizer.plot_drone_routes(drone_routes, truck_routes)
optimizer.launch_additional_drones_for_unvisited(drone_routes)
optimizer.plot_drone_routes(drone_routes, truck_routes)
total_truck_time, total_drone_time, total_time = optimizer.calculate_total_cost(drone_routes, truck_routes)
improved_drone_routes = optimizer.optimize_drone_landing_points(drone_routes, truck_routes)
optimizer.visualize_optimized_landings(improved_drone_routes, truck_routes)
total_truck_time, total_drone_time, total_time = optimizer.calculate_optimized_total_cost(improved_drone_routes, truck_routes)
optimizer.plot_drone_routes(improved_drone_routes, truck_routes)
total_truck_time, total_drone_time, total_time = optimizer.calculate_total_cost(drone_routes, truck_routes)
optimizer.print_detailed_route_history(improved_drone_routes, truck_routes)