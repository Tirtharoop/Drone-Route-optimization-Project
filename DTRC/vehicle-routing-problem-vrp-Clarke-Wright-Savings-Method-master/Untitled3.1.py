import numpy as np
from typing import List, Tuple
import math
import time as time_lib

class Node:
    def __init__(self, id: int, x: float, y: float, demand: float):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand

class DTRC:
    def __init__(self, nodes: List[Node], capacity: float, initial_routes: List[List[int]], 
                 num_drones: int = 2, drone_capacity: float = 25, battery_life: float = 30):
        self.nodes = nodes
        self.capacity = capacity
        self.initial_routes = initial_routes
        self.num_drones = num_drones
        self.drone_capacity = drone_capacity
        self.battery_life = battery_life
        self.distances = self._calculate_distances()
        
    def _calculate_distances(self) -> np.ndarray:
        """Calculate Euclidean distances between all nodes."""
        n = len(self.nodes)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                xi, yi = self.nodes[i].x, self.nodes[i].y
                xj, yj = self.nodes[j].x, self.nodes[j].y
                distances[i][j] = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
        return distances

    def apply_drone_routing(self, route: List[int]) -> Tuple[List[int], float]:
        remain_nodes = set(route)
        available_drones = set(range(self.num_drones))
        drone_positions = {i: route[0] for i in range(self.num_drones)}
        drone_batteries = {i: self.battery_life for i in range(self.num_drones)}
        drone_loads = {i: 0 for i in range(self.num_drones)}
        
        optimized_route = []
        total_cost = 0
        landing_nodes = set()  # Track landing locations of drones
        
        while remain_nodes:
            if available_drones:
                drone = min(available_drones)
                available_drones.remove(drone)
                
                best_node = None
                min_ratio = float('inf')
                
                for node in remain_nodes:
                    if self.nodes[node-1].demand <= self.drone_capacity and self.distances[drone_positions[drone]-1][node-1] <= drone_batteries[drone]:
                        D = self.distances[drone_positions[drone]-1][node-1]
                        W = drone_loads[drone] + self.nodes[node-1].demand
                        ratio = D / W if W > 0 else float('inf')  
                        
                        if ratio < min_ratio:
                            min_ratio = ratio
                            best_node = node
                
                if best_node:
                    remain_nodes.remove(best_node)
                    drone_batteries[drone] -= self.distances[drone_positions[drone]-1][best_node-1]
                    drone_loads[drone] += self.nodes[best_node-1].demand
                    drone_positions[drone] = best_node
                    total_cost += self.distances[drone_positions[drone]-1][best_node-1]
                    landing_nodes.add(best_node)  # Store the landing node
        
            if remain_nodes:
                current_pos = optimized_route[-1] if optimized_route else route[0]
                best_node = None
                min_cost = float('inf')
                
                for node in remain_nodes:
                    cost = self.distances[current_pos-1][node-1]
                    if cost < min_cost:
                        min_cost = cost
                        best_node = node
                
                if best_node:
                    remain_nodes.remove(best_node)
                    optimized_route.append(best_node)
                    total_cost += min_cost
                    
                    if best_node in landing_nodes:
                        for drone in range(self.num_drones):
                            if drone_positions[drone] == best_node:
                                available_drones.add(drone)
                                drone_batteries[drone] = self.battery_life
                                drone_loads[drone] = 0
                                landing_nodes.remove(best_node)  # Reset landing node
        
        return optimized_route, total_cost

    def solve(self) -> Tuple[List[List[int]], float]:
        final_routes = []
        total_cost = 0
        
        for route in self.initial_routes:
            optimized_route, route_cost = self.apply_drone_routing(route)
            final_routes.append(optimized_route)
            total_cost += route_cost
            
        return final_routes, total_cost

def main():
    nodes_data = [
        (1, 42, 68, 0), (2, 77, 97, 5), (3, 28, 64, 23), (4, 77, 39, 14),
        (5, 32, 33, 13), (6, 32, 8, 8), (7, 42, 92, 18), (8, 8, 3, 19),
        (9, 7, 14, 10), (10, 82, 17, 18), (11, 48, 13, 20), (12, 53, 82, 5),
        (13, 39, 27, 9), (14, 7, 24, 23), (15, 67, 98, 9), (16, 54, 52, 18),
        (17, 72, 43, 10), (18, 73, 3, 24), (19, 59, 77, 13), (20, 58, 97, 14),
        (21, 23, 43, 8), (22, 68, 98, 10), (23, 47, 62, 19), (24, 52, 72, 14),
        (25, 32, 88, 13), (26, 39, 7, 14), (27, 17, 8, 2), (28, 38, 7, 23),
        (29, 58, 74, 15), (30, 82, 67, 8), (31, 42, 7, 20), (32, 68, 82, 24),
        (33, 7, 48, 3)
    ]
    
    nodes = [Node(id, x, y, demand) for id, x, y, demand in nodes_data]
    capacity = 100
    
    initial_routes = [
        [15, 17, 9, 3, 16, 29],
        [12, 5, 26, 7, 8, 13, 32, 2],
        [20, 4, 27, 25, 30, 10],
        [23, 28, 18, 22],
        [24, 6, 19, 14, 21, 1, 31, 11]
    ]
    
    optimal_value = 661  # Given target cost
    costs = []
    times = []
    
    # Run algorithm 10 times
    for _ in range(10):
        start_time = time_lib.time()
        solver = DTRC(nodes, capacity, initial_routes)
        _, cost = solver.solve()
        end_time = time_lib.time()
        
        costs.append(cost)
        times.append(end_time - start_time)
    
    # Calculate statistics
    best_cost = min(costs)
    average_cost = sum(costs) / len(costs)
    average_time = sum(times) / len(times)
    gap = ((best_cost - optimal_value) / optimal_value) * 100
    
    print(f"{gap:.2f}\t{best_cost:.0f}\t{average_cost:.1f}\t{average_time:.2f}")

if __name__ == "__main__":
    main()
