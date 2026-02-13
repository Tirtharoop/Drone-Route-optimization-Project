import copy
import random
import math
import numpy as np
import pandas as pd
import sys
from truck import Truck
from drone import Drone
from node import Node
from PaperAlgos.route import *

class LNS:
    def __init__(
                    self, truck_routes, drone_routes, demands, distance_matrix,
                    drone_capacity=99999, truck_capacity=100, 
                    max_time=10, max_iterations=1000, p1=0.3, p2=0.3, p3=0.3
                 ) -> None:
        self.nodes : list[Node] = self.initialize_nodes(demands, distance_matrix)
        self.trucks :list[Truck] = self.initialize_from_dtrc_solution(truck_routes, drone_routes, demands, distance_matrix, truck_capacity, drone_capacity)

    def initialize_nodes(self, demands, distance_matrix) -> list[Node]:
        nodes = []
        for idx, demand in enumerate(demands):
            nodes.append( Node(idx, demand[idx+1], distance_matrix[idx] ) )
        return nodes

    def initialize_from_dtrc_solution(self, truck_routes, drone_routes, demands, 
                                      distance_matrix, truck_capacity, drone_capacity):
        trucks = []
        for t_idx, t_route in enumerate(truck_routes):
            drones = []
            truck_update_list = []
            for d_idx, drone in enumerate(drone_routes[t_idx]):
                routes = []
                drone_update_list = []
                for d_routes in drone:
                    route = None
                    for d_route in d_routes:
                        nodes = []
                        for d_node in d_route:
                            node = next(n for n in self.nodes if n.id == d_node)
                            nodes.append(node)
                            drone_update_list.append(node)
                        route = DroneRoute(nodes)
                    routes.append(route)
                drone = Drone(d_idx+1, drone_capacity, routes)
                for node in drone_update_list:
                    node.update_assigned_drone(drone)
                drones.append(drone)
            nodes = []
            for t_node in t_route:
                node = next(n for n in self.nodes if n.id == t_node)
                nodes.append(node)
                truck_update_list.append(node)
            truck = Truck(t_idx+1, drones, TruckRoute(nodes), truck_capacity)
            for node in truck_update_list:
                node.update_assigned_truck(truck)
            trucks.append(truck)
        return trucks
        
    def print_routes(self, trucks):
        for truck in trucks:
            print(truck)
                    

                    
                