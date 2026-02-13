import copy
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

class LNS:
    def __init__(self, truck_routes, drone_routes, demands,
                 euclidean_distance_matrix, max_time=10,
                 max_iterations=1000, p1=0.3, p2=0.3, p3=0.3,
                 drone_capacity=99999, truck_capacity=100,
                 #truck_speed=1, drone_speed
                ):
        """
        Initialize the Truck-Drone Routing problem with LNS algorithm

        Args:
            truck_routes: List of truck routes, each route is a list of customer indices
            drone_routes: List of drone routes for each truck, where each truck has up to 2 drones
            demands: Dictionary of node demands (node_id -> demand value)
	    euclidean_distance_matrix: Matrix of Euclidean distances between nodes for drones
            max_iterations: Maximum number of iterations for the LNS algorithm
            max_time: Maximum number of trials for improvement
            p1: Percentage of drone-only nodes to remove
            p2: Percentage of truck-only nodes to remove
            p3: Percentage of sub-drone routes to remove
            drone_capacity: Maximum capacity of drones
            drone_cost_factor: Cost multiplier for drone routes (< 1 makes drones cheaper)
            truck_speed: Speed of trucks in km/min
            drone_speed: Speed of drones in km/min
            time_weight: Weight for time optimization (1-time_weight is cost weight)
            force_use_drones: Whether to force the solution to use drones
        """
        self.truck_routes = truck_routes
        self.drone_routes = drone_routes
        self.demands = demands
        #self.euclidean_distance_matrix = euclidean_distance_matrix
        self.euclidean_distance_matrix = euclidean_distance_matrix
        self.max_times = max_time
        self.max_iterations = max_iterations
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.drone_capacity = drone_capacity
        # self.drone_cost_factor = drone_cost_factor
        self.truck_capacity = truck_capacity
        self.truck_speed = 1  # km/h
        self.drone_speed = 1.5  # km/h
        # self.time_weight = time_weight  # weight for time optimization
        # self.force_use_drones = force_use_drones
        self.best_solution = None
        self.best_objective = float('inf')
        self.current_solution = None
        self.current_objective = float('inf')
        self.logger = open("PaperAlgos/LNS.log", "w")
        self.log_indentation = ""
        # Initialize node locations for visualization
        self.num_nodes = max(demands.keys()) + 1
        # Use a fixed seed for reproducibility
        np.random.seed(42)
        # self.node_locations = np.random.rand(self.num_nodes, 2) * 100    def get_cost(self, truck_routes):
        cost = 0
        for route in truck_routes:
            for i in range(1, len(route)):
                cost += self.euclidean_distance_matrix[route[i-1]][route[i]] / self.truck_speed
        
        # Add 3 * number of delivery points truck delivery (excluding depot)
        truck_delivery_points = 0
        for route in truck_routes:
            # Count delivery points (exclude depot at start and end)
            truck_delivery_points += len(route) - 2  # -2 for start and end depot
        
        # Add 3 times the number of truck delivery points to cost
        truck_delivery_cost = 3 * truck_delivery_points
        cost += truck_delivery_cost
        
        # Add 1 * number of delivery points by drone (excluding takeoff and landing)
        drone_delivery_points = 0
        for truck_drone_routes in self.drone_routes:
            for drone_route_list in truck_drone_routes:
                for route in drone_route_list:
                    # Count delivery points (exclude takeoff and landing nodes)
                    drone_delivery_points += len(route) - 2  # -2 for takeoff and landing nodes
        
        # Add 1 times the number of drone delivery points to cost
        drone_delivery_cost = 1 * drone_delivery_points
        cost += drone_delivery_cost
        
        return cost

    def get_best_solution(self, s1, s2, s3):
        self.log_function("get_best_solution")
        c1 = math.inf if s1 is None else self.get_cost(s1[0])
        c2 = self.get_cost(s2[0])
        c3 = math.inf if s3 is None else self.get_cost(s3[0])
        cost_str = f"Drone Insert Cost : {c1}. Truck Insert Cost : {c2}. Sub route Insert Cost : {c3}.\n"
        self.log_message(cost_str)
        if c1 <= c2 and c1 <= c3:
            self.log_function("get_best_solution", False)
            return s1, cost_str 
        elif c2 <= c1 and c2 <= c3:
            self.log_function("get_best_solution", False)
            return s2, cost_str 
        elif c3 <= c1 and c3 <= c2:
            self.log_function("get_best_solution", False)
            return s3, cost_str 
        else:
            self.log_function("get_best_solution", False)
            return None, cost_str 

    def insertion_cost(self, route, i, node):
        return self.euclidean_distance_matrix[route[i-1]][node] + self.euclidean_distance_matrix[node][route[i]] - self.euclidean_distance_matrix[route[i-1]][route[i]]

    def get_ordered_insertion_points(self, drone_routes, node):
        self.log_function("get_ordered_insertion_points")
        insertion_points = []
        for idx, route in enumerate(drone_routes):
            for i in range(1, len(route['route'])):
                insertion_points.append({'route_idx': idx, 'insert_idx': i, 'cost': self.insertion_cost(route['route'], i, node)})
                self.log_message(f"Added Insertion Point : {insertion_points[-1]}")
        insertion_points.sort(key=lambda x: x['cost'])
        self.log_message(f"Sorted Insertion Points : {insertion_points}")
        self.log_function("get_ordered_insertion_points", False)
        return insertion_points
    
    def get_drone_payload(self, route):
        payload = 0
        for node in route[1:-1]:
            payload += self.demands[node+1]
        return payload

    def drone_payload_feasibility(self, route, node):
        self.log_function("drone_payload_feasibility")
        weight = self.demands[node+1] + self.get_drone_payload(route)
        # if weight <= self.drone_capacity:
        #     print("Drone payload feasible.")
        # else:
        #     print("Drone payload infeasible.")
        self.log_message(f"Drone Payload : {weight}. Drone Capacity : {self.drone_capacity}")
        self.log_function("drone_payload_feasibility", False)
        return weight <= self.drone_capacity
    
    def battery_feasibility(self, route, insert_idx, node):
        self.log_function("battery_feasibility")
        route.insert(insert_idx, node)
        payload = self.get_drone_payload(route)
        battery = 100
        self.log_message(f"Drone Payload at launch : {payload}. Battery : {battery}%.")
        for i in range(1, len(route)):
            distance = self.euclidean_distance_matrix[route[i-1]][route[i]]
            battery -= (distance * ((200 + payload)**1.5)) / 1558.85
            # print("Remaining Battery : ", battery)
            payload -= self.demands[route[i]+1]
            self.log_message(f"Distance Traversed : {distance:.2f}. Remaining Battery : {battery:.2f}. Remaining Payload : {payload}")
            if battery < 0:
                self.log_message("Battery failed for node #{node} in route {route}.")
                break            
        route.remove(node)
        # if battery >= 0:
        #     print("Battery capacity feasible.")
        # else:
        #     print("Battery capacity infeasible.")
        self.log_function("battery_feasibility")
        return battery >= 0


    def drone_node_insertion_feasible(self, point, drone_routes, node, truck_payloads):
        self.log_function("drone_node_insertion_feasible")
        selected_route = drone_routes[point['route_idx']]
        self.log_message(f"Selected drone route : {selected_route}. Truck : {selected_route['truck_idx']+1}. Truck Payload : {truck_payloads[selected_route['truck_idx']]}. Node demand : {self.demands[node+1]}. Truck Capacity : {self.truck_capacity}.")
        return ( 
            truck_payloads[selected_route['truck_idx']] + self.demands[node+1] < self.truck_capacity
            and self.drone_payload_feasibility(selected_route['route'], node) 
            and self.battery_feasibility(selected_route['route'], point['insert_idx'], node)
            and (self.log_function("drone_node_insertion_feasible", False) or True)
            or (self.log_function("drone_node_insertion_feasible", False) or False)
            )

    def perform_cheapest_insertion(self, drone_routes, node, turck_payloads):
        self.log_function("perform_cheapest_insertion")
        insertion_points = self.get_ordered_insertion_points(drone_routes, node)
        for point in insertion_points:
            self.log_message(f"Trying insertion at point : {point}")
            if self.drone_node_insertion_feasible(point, drone_routes, node, turck_payloads):
                self.log_message(f"Inserting drone node #{node} for Drone #{drone_routes[point['route_idx']]['drone_idx']+1} of Truck #{drone_routes[point['route_idx']]['truck_idx']+1}.")
                self.log_message(f"Drone route before insertion at index {point['route_idx']}: {drone_routes[point['route_idx']]['route']}")
                drone_routes[point['route_idx']]['route'].insert(point['insert_idx'], node)
                self.log_message(f"Drone route after insertion : {drone_routes[point['route_idx']]['route']}")
                self.log_function("perform_cheapest_insertion", False)
                return True
        self.log_function("perform_cheapest_insertion", False)
        return False
    
    def insert_drone_nodes(self, s_temp, node):
        self.log_function("insert_drone_nodes")
        truck_payloads = [ ]
        for t_idx, route in enumerate(s_temp[0]):
            truck_payloads.append(self.get_truck_payload(route, s_temp[1][t_idx]))
            self.log_message(f"Truck #{t_idx+1} - Route : {route}, Payload : {truck_payloads[-1]}")
        drone_routes = self.get_sub_drone_routes(s_temp[1])
        if not drone_routes:
            self.log_function("insert_drone_nodes", False)
            return None
        if self.perform_cheapest_insertion(drone_routes, node, truck_payloads):
            self.log_function("insert_drone_nodes", False)
            return s_temp
        else:
            self.log_function("insert_drone_nodes", False)
            return None
        
    def get_truck_payload(self, route, carrying_drones):
        payload = self.get_drone_payload_of_truck(carrying_drones)
        for node in route[1:-1]:
            payload += self.demands[node+1]
        return payload
    
    def get_drone_payload_of_truck(self, drones):
        payload = 0
        for drone_routes in drones:
            for route in drone_routes:
                payload += self.get_drone_payload(route)
        return payload

    def truck_payload_feasibility(self, truck_route, drones_of_truck, node):
        self.log_function("truck_payload_feasibility")
        payload = self.get_truck_payload(truck_route, drones_of_truck) + self.demands[node+1]
        self.log_message(f"Truck Payload : {payload}. Truck Capacity : {self.truck_capacity}")
        self.log_function("truck_payload_feasibility", False)
        return  payload<= self.truck_capacity
    
    def get_min_cost_insert_index(self, route, node):
        self.log_function("get_min_cost_insert_index")
        min_cost = math.inf
        min_index = None
        for i in range(1, len(route)):
            cost  = self.insertion_cost(route, i, node)
            if cost < min_cost:
                min_cost = cost
                min_index = i
        self.log_message(f"Best index to insert node #{node} in route {route} is {min_index}. Cost: {min_cost:2}")
        self.log_function("get_min_cost_insert_index")
        return min_index, min_cost

    def insert_truck_nodes(self, s_temp, node):
        self.log_function("insert_truck_nodes")
        truck_routes = s_temp[0]
        drone_routes = s_temp[1]
        route_insert_index = None
        node_insert_index = None
        min_insert_cost = math.inf
        for truck_idx, route in enumerate(truck_routes):
            if self.truck_payload_feasibility(route, drone_routes[truck_idx], node):
                self.log_message(f"Inserting node #{node} in Truck #{truck_idx+1} route is feasible.")
                idx, cost = self.get_min_cost_insert_index(route, node)
                self.log_message(f"Current min. cost : {min_insert_cost}. New cost : {cost}")
                if cost < min_insert_cost:
                    min_insert_cost = cost
                    node_insert_index = idx
                    route_insert_index = truck_idx
                    self.log_message(f"Best truck route for insertion updated. Truck #{truck_idx}.")
            else:
                self.log_message(f"Inserting node #{node} in Truck #{truck_idx+1} route is not feasible.")
        if route_insert_index is None:
            truck_routes.append([0, node, 0])   #add a new truck
            self.log_message(f"New truck route added : {truck_routes[-1]}.")
            s_temp[1].append([[],[]])           #add drones for the new truck
            self.log_message(f"Drones to the new truck added. {s_temp[-1]}.")
        else:
            self.log_message(f"Truck #{route_insert_index+1} route before node #{node} insert : {truck_routes[route_insert_index]}")
            truck_routes[route_insert_index].insert(node_insert_index, node)
            self.log_message(f"Truck #{route_insert_index+1} route after node #{node} insert : {truck_routes[route_insert_index]}")
        self.log_function("insert_truck_nodes", False)
        return s_temp

    def get_drone_availability(self, truck_route, terminal_nodes):
        self.log_function("get_drone_availability")
        self.log_message(f"Considered Truck route : {truck_route}")
        self.log_message(f"Terminals of Drone route : {terminal_nodes}")
        launch_nodes = []
        landing_nodes = []
        drone_availability = [False]    # Drone cannot be launched from depot. Hence, unavailable at depot.
        for node in terminal_nodes:
            launch_nodes.append(node[0])
            landing_nodes.append(node[1])
        available = True
        for node in truck_route[1:-1]:
            if node in launch_nodes:
                available = not available
            drone_availability.append(available)
            if node in landing_nodes:
                available = not available
        drone_availability.append(False) # Drone cannot land at depot. Hence, unavailable at depot.
        self.log_message(f"Drone Availability across Truck Route : {drone_availability}")
        self.log_function("get_drone_availability", False)
        return drone_availability
    

    def ensure_single_launch_or_land(self, drone_availability, drone_terminals, truck_route):
        for launch_node, land_node in drone_terminals:
            drone_availability[truck_route.index(launch_node)] = False
            drone_availability[truck_route.index(land_node)] = False

    def get_availability_of_drones(self,truck_route, drone_routes):
        self.log_function("get_availability_of_drones")
        drone_1_terminal_nodes = [(route[0], route[-1]) for route in drone_routes[0]]
        drone_2_terminal_nodes = [(route[0], route[-1]) for route in drone_routes[1]]
        drone_1_availability = self.get_drone_availability(truck_route, drone_1_terminal_nodes)
        drone_2_availability = self.get_drone_availability(truck_route, drone_2_terminal_nodes)
        self.ensure_single_launch_or_land(drone_1_availability, drone_2_terminal_nodes, truck_route)
        self.ensure_single_launch_or_land(drone_2_availability, drone_1_terminal_nodes, truck_route)
        self.log_message(f"Truck Route : {truck_route}")
        self.log_message(f"Drone #1 Route : {drone_routes[0]}")
        self.log_message(f"Drone #2 Route : {drone_routes[1]}")
        self.log_message(f"Drone #1 Availability : {drone_1_availability}")
        self.log_message(f"Drone #2 Availability : {drone_2_availability}")
        self.log_function("get_availability_of_drones", False)
        return [drone_1_availability, drone_2_availability]

    def drone_available_till_next_node(self, availability, i):
            # check if either of the drones is available
            return availability[0][i] and availability[0][i+1] or availability[1][i] and availability[1][i+1]
    
    def drone_subroute_creation_feasibility(self, truck_route, drones_of_truck, subroute, node):
        self.log_function("drone_subroute_creation_feasibility")
        return ( 
                    self.truck_payload_feasibility(truck_route, drones_of_truck, node)
                and self.drone_payload_feasibility(subroute, node)
                and self.battery_feasibility(subroute, 1, node) 
                and (self.log_function("drone_subroute_creation_feasibility") or True)
                or (self.log_function("drone_subroute_creation_feasibility") or False)
        )

    def get_cheapest_launch_node_in_route(self, truck_route, drones_of_truck, node):
        self.log_function("get_cheapest_launch_node_in_route")
        cheapest_cost = math.inf
        cheapest_launch = None
        assigned_drone = None
        # get the availability of all the drones for each truck node in the truck route
        drone_availability = self.get_availability_of_drones(truck_route, drones_of_truck)
        for i in range(2, len(truck_route)-1): # exclude the depot at both end
            cost = self.euclidean_distance_matrix[truck_route[i-1]][node] + self.euclidean_distance_matrix[node][truck_route[i]]            
            if cost < cheapest_cost and self.drone_available_till_next_node(drone_availability, i-1):
                sub_route = [truck_route[i-1], truck_route[i]]
                if  self.drone_subroute_creation_feasibility(truck_route, drones_of_truck, sub_route, node):
                    cheapest_cost = cost
                    cheapest_launch = i-1
                    preferred_drone = 1 if drone_availability[1].count(True) > drone_availability[0].count(True) else 0
                    if drone_availability[preferred_drone][i-1] and drone_availability[preferred_drone][i]:
                        assigned_drone = preferred_drone
                    elif drone_availability[(preferred_drone+1)%2][i-1] and drone_availability[(preferred_drone+1)%2][i]:
                        assigned_drone = (preferred_drone+1)%2
                    else:
                        assigned_drone = None
        self.log_message(f"Cheapest launch node in route {truck_route} is {cheapest_launch} for Drone #{assigned_drone if assigned_drone is None else assigned_drone+1}. Cost : {cheapest_cost}.")
        self.log_function("get_cheapest_launch_node_in_route", False)
        return cheapest_launch, cheapest_cost, assigned_drone

    def insert_drone_routes(self, s_temp, node):
        self.log_function("insert_drone_routes")
        truck_routes = s_temp[0]
        drone_routes = s_temp[1]
        # initialize the best truck route, best launch node in the route, drone of the truck and best cost
        cheapest_launch_node = None
        cheapest_cost = math.inf
        assigned_drone_idx = None
        truck_route_idx = None

        for idx, truck_route in enumerate(truck_routes):
            # get the cheapest launch node, selected drone nodes and best cost for sub drone route creation in the route
            route_launch_node, route_cost, route_drone_idx = self.get_cheapest_launch_node_in_route(truck_route, drone_routes[idx], node)
            self.log_message(f"Current Best Cost : {cheapest_cost}. New Cost : {route_cost}")
            # if the new cost is better
            if route_cost < cheapest_cost:
                # update the truck route, launch node, drone and cost 
                truck_route_idx, cheapest_launch_node, cheapest_cost, assigned_drone_idx = idx, route_launch_node, route_cost, route_drone_idx
                self.log_message(f"New best route is updated to Truck #{truck_route_idx+1}, Drone #{assigned_drone_idx if assigned_drone_idx is None else assigned_drone_idx+1}.")                
        # if a launch node is found
        if cheapest_launch_node is not None:
            # create the new sub drone route
            new_sub_route = [truck_routes[truck_route_idx][cheapest_launch_node], node, truck_routes[truck_route_idx][cheapest_launch_node + 1]]
            self.log_message(f"Created Sub-drone route : {new_sub_route}")
            drone_routes[truck_route_idx][assigned_drone_idx].append(new_sub_route)
            # reorder the new sub drone route following the visit sequence of its truck
            try:
                drone_routes[truck_route_idx][assigned_drone_idx].sort(key=lambda x: truck_routes[truck_route_idx].index(x[0]))
                self.log_message(f"Sub-drone route assigned to Drone #{assigned_drone_idx+1} of Truck #{truck_route_idx+1}.")
                self.log_message(f"Assigned Truck Route : {truck_routes[truck_route_idx]}")       
                self.log_message(f"Assigned Drone Route : {drone_routes[truck_route_idx][assigned_drone_idx]}")                    
            except (IndexError, ValueError) as error:
                print(f"Error : {error}")
                print(f"Truck Route Index : {truck_route_idx}")
                print(f"Drone Routes at Truck Route Index : {drone_routes[truck_route_idx]}")
                print(f"Assigned Drone Index : {assigned_drone_idx}")
                print(f"Routes at Assigned Drone Index : {drone_routes[truck_route_idx][assigned_drone_idx]}")
                print(f"Truck Route at Truck Route Index : {truck_routes[truck_route_idx]}")
                sys.exit(0)
            self.log_function("insert_drone_routes", False)
            return s_temp
        return None
    
    def repair(self, s_temp, removed_nodes):
        # the repair operation
        self.log_function("repair")
        debug_str = ""
        while removed_nodes:
            # select a node from removed nodes by destroy operation
            self.log_message(f"Nodes remaining for insertion : {[removed_nodes]}")
            selected_node = removed_nodes.pop()
            self.log_message(f"Node selected for insertion : {selected_node}")
            debug_str += f"Inserting Node : {selected_node}.\n"
            # insert the selected node as drone node
            s_drone_node_inserted = self.insert_drone_nodes(copy.deepcopy(s_temp), selected_node)
            # insert the selected node as truck node
            s_truck_node_inserted = self.insert_truck_nodes(copy.deepcopy(s_temp), selected_node)
            # insert the selected node as new sub drone route
            s_drone_route_inserted = self.insert_drone_routes(copy.deepcopy(s_temp), selected_node)
            # get the best solution among the above three options
            s_temp, cost_str = self.get_best_solution(s_drone_node_inserted, s_truck_node_inserted, s_drone_route_inserted)
            debug_str += cost_str
            if s_temp is s_drone_node_inserted:
                debug_str += f"Drone Node Inserted for Node #{selected_node}.\n"
            elif s_temp is s_truck_node_inserted:
                debug_str += f"Truck Node Inserted for Node #{selected_node}.\n"
            elif s_temp is s_drone_route_inserted:
                debug_str += f"Sub Drone Route Inserted for Node #{selected_node}.\n" 
            else:
                debug_str += f"Fault Insertion for Node #{selected_node}.\n"
        self.log_function("repair", False)
        return s_temp, debug_str
    
    def get_drone_nodes(self, drone_routes):
        # get the drone nodes and the launch and land nodes
        self.log_function("get_drone_nodes")
        self.log_heading("Current Drone Routes")
        drone_nodes = []
        launch_land_nodes = []
        for truck_drones in drone_routes:
            for drone in truck_drones:
                for drone_route in drone:
                    if drone_route:
                        # A drone route should have atleast 3 nodes including  launching and landing nodes
                        self.log_message(f"{drone_route}")
                        if len(drone_route) < 3:    
                            raise ValueError(drone_route)
                        drone_nodes.extend(drone_route[1:-1])        #exclude landing and launching nodes
                        launch_land_nodes.append((drone_route[0], drone_route[-1]))
        self.log_message(f"All Drone Nodes : {drone_nodes}")
        self.log_message(f"Landing and Launching Node : {launch_land_nodes}")
        self.log_function("get_drone_nodes", False)
        return drone_nodes, launch_land_nodes


    def remove_nodes_from_drone_route(self, drone_routes, nodes_for_removal, launch_land_nodes):
        # remove the selected drone nodes from drone routes
        self.log_function("remove_nodes_from_drone_route")
        self.log_message(f"Drone nodes to remove : {nodes_for_removal}")
        for node in nodes_for_removal:
            removed = False
            for truck_drones in drone_routes:
                for drone in truck_drones:
                    for drone_route in drone:
                        if node in drone_route:
                            self.log_message(f"Drone Node #{node} found in route {drone_route}.")
                            self.log_message(f"All routes of this drone before removal : {drone}.")
                            if len(drone_route) == 3:
                                try: # only launching and landing node remains after removal
                                    # remove them from launching and landing node
                                    self.log_message(f"Releasing Launch and Landing Node : {(drone_route[0], drone_route[-1])}")
                                    launch_land_nodes.remove((drone_route[0], drone_route[-1]))
                                except ValueError as error:
                                    self.log_message(f"Error : {error}")
                                    print(f"Error : {error}")
                                    print(f"Nodes for Removal : {nodes_for_removal}")
                                    print(f"Drone Removed : {node}")
                                    print(f"Drone Route after removal: {drone_route}")
                                    print(f"Landing and Launching Nodes : {launch_land_nodes}")
                                    sys.exit(0)
                                drone.remove(drone_route)
                            else:
                                drone_route.remove(node)
                            self.log_message(f"All routes of this drone after removal : {drone}.")
                            removed = True
                            break
                    if removed:
                        break
                if removed:
                    break
        self.log_function("remove_nodes_from_drone_route", False)

    def remove_drone_nodes(self, s_current):
        self.log_function("remove_drone_nodes")
        # get the drone nodes and launch and land nodes
        drone_nodes, launch_land_nodes = self.get_drone_nodes(s_current[1])
        # select p1 fraction of drone node to be removed
        nodes_for_removal = random.sample(population=drone_nodes, k=int(len(drone_nodes)*self.p1))
        self.log_message(f"Drone Nodes for Removal : {nodes_for_removal}")
        # remove the selected drone nodes from drone routes
        self.remove_nodes_from_drone_route(s_current[1], nodes_for_removal, launch_land_nodes)
        # self.log_heading("Return Values")
        # self.log_message(f"s_current : {s_current}")
        # self.log_message(f"node_for_removal : {node_for_removal}")
        self.log_function("remove_drone_nodes", False)
        return s_current, nodes_for_removal, launch_land_nodes

    def remove_nodes_from_truck_route(self, truck_routes, nodes_for_removal):
        # remove the selected truck nodes from truck routes
        self.log_function("remove_nodes_from_truck_route")
        for node in nodes_for_removal:
            for route in truck_routes:
                if node in route:
                    self.log_message(f"Removing {node} from {route}.")
                    route.remove(node)
                    self.log_message(f"Route after node removal : {route}")
                    break
        self.log_function("remove_nodes_from_truck_route", False)

    def get_truck_nodes(self, truck_routes, launch_land_nodes):
        # get the truck nodes in an array except the launch and land nodes
        self.log_function("get_truck_nodes")
        truck_nodes = []
        launch_nodes = [x for (x, y) in launch_land_nodes]
        self.log_message(f"All launch nodes : {launch_nodes}")
        land_nodes = [y for (x, y) in launch_land_nodes]
        self.log_message(f"All landing nodes : {land_nodes}")
        for route in truck_routes:
            self.log_message(f"Truck Route under consideration : {route}")
            for node in route[1:-1]:
                if node not in launch_nodes and node not in land_nodes:
                    truck_nodes.append(node)
        self.log_message(f"All truck nodes : {truck_nodes}")
        self.log_function("get_truck_nodes", False)
        return truck_nodes

    def remove_truck_nodes(self, s_drone_removed, launch_land_nodes):
        self.log_function("remove_truck_nodes")
        # remove the truck nodes except the launch and land nodes
        truck_nodes = self.get_truck_nodes(s_drone_removed[0], launch_land_nodes)
        # select p2 fraction of truck nodes to be removed
        nodes_for_removal = random.sample(population=truck_nodes, k=int(len(truck_nodes)*self.p2))
        self.log_message(f"Truck Nodes for Removal : {nodes_for_removal}")
        # remove the selected nodes
        self.remove_nodes_from_truck_route(s_drone_removed[0], nodes_for_removal)
        self.log_function("remove_truck_nodes", False)
        return s_drone_removed, nodes_for_removal
    
    def get_sub_drone_routes(self, drone_routes):
        # get all the drone routes in a single array
        self.log_function("get_sub_drone_routes")
        sub_drone_routes = []
        for t_idx, truck_drones in enumerate(drone_routes):
            for d_idx, drone in enumerate(truck_drones):
                for r_idx, drone_route in enumerate(drone):
                    if drone_route:
                        sub_drone_routes.append({'truck_idx': t_idx, 'drone_idx': d_idx, 'route_idx': r_idx, 'route': drone_route})
                        self.log_message(f"Sub-drone route found : {sub_drone_routes[-1]}")                        
        self.log_function("get_sub_drone_routes", False)
        return sub_drone_routes

    def remove_routes_from_drone_routes(self, drone_routes, routes_for_removal):
        # remove the selected sub drone routes
        self.log_function("remove_routes_from_drone_routes")
        for route in routes_for_removal:
            removed = False
            self.log_message(f"Removing sub-drone route : {route}")
            for truck_drones in drone_routes:
                for drone in truck_drones:
                    if route in drone:
                        self.log_message(f"Drone routes before removal : {drone}")
                        drone.remove(route)
                        removed = True
                        self.log_message(f"Drone routes after removal : {drone}")
                        break
                if removed:
                    break
        self.log_function("remove_routes_from_drone_routes", False)

    def remove_sub_drone_route(self, s_truck_node_removed):
        self.log_function("remove_sub_drone_route")
        # get all sub drone routes
        sub_drone_routes = self.get_sub_drone_routes(s_truck_node_removed[1])
        # determine p3 fraction of sub drone routes to be removed
        routes_for_removal = random.sample(population=sub_drone_routes, k=int(len(sub_drone_routes)*self.p3))
        # remove the selected sub drone routes
        self.log_message(f"Sub-drone routes for removal : {sub_drone_routes}")
        self.remove_routes_from_drone_routes(s_truck_node_removed[1], routes_for_removal)

        removed_nodes = []
        # save the routes removed by sub drone route removal
        for route in routes_for_removal:
            removed_nodes.extend(route['route'][1:-1])
        self.log_message(f"Nodes removed during route removal : {removed_nodes}")
        self.log_function("remove_sub_drone_route", False)
        return s_truck_node_removed, removed_nodes

    def destroy(self, s_current): 
        self.log_function("destroy")
        removed_nodes = [] # holds the removed nodes by destroy operators
        # remove drone nodes
        s_drone_node_removed, nodes_removed, launch_land_nodes = self.remove_drone_nodes(s_current)
        # save the removed nodes
        removed_nodes.extend(nodes_removed)
        self.log_message(f"All removed nodes after drone node removal : {removed_nodes}")
        # remove truck nodes
        s_truck_node_removed, nodes_removed = self.remove_truck_nodes(s_drone_node_removed, launch_land_nodes)
        # save the removed nodes
        removed_nodes.extend(nodes_removed)
        self.log_message(f"All removed nodes after truck node removal : {removed_nodes}")
        # remove sub drone routes
        s_drone_route_removed, nodes_removed = self.remove_sub_drone_route(s_truck_node_removed)
        # save the removed nodes
        removed_nodes.extend(nodes_removed)
        self.log_message(f"All removed nodes after sub-drone route removal : {removed_nodes}")
        # print("Removed Nodes : ", removed_nodes)
        return s_drone_route_removed, removed_nodes
    
    def log_truck_routes(self, truck_routes):
        self.log_heading("Truck Routes")
        for idx, route in enumerate(truck_routes):
            self.logger.write(f"\nTruck #{idx+1} : {route}")
        self.logger.write("\n")
    
    def log_drone_routes(self, drone_routes):
        self.log_heading("Drone Routes")
        for t_idx, drones in enumerate(drone_routes):
            for d_idx, drone in enumerate(drones):
                self.logger.write(f"\nTruck #{t_idx+1} - Drone #{d_idx+1} : {drone}")
        self.logger.write("\n")

    def log_heading(self, str):
        self.logger.write("\n"+self.log_indentation+str)
        self.logger.write("\n"+self.log_indentation+("-"*len(str))+"\n")

    def log_message(self, str, enable_log = False):
        enable_log and self.logger.write("\n"+self.log_indentation+str)
    
    def log_function(self, function_name, start = True):
        if start:
            self.log_indentation += "\t"
        self.log_message(f"Function :: {function_name} -- {'Start' if start else 'End'}", True)
        if not start:
            self.log_indentation = self.log_indentation[:-1]

    def run(self):
        # Initialize the solution provided by DTRC as best solution
        s_best = (self.truck_routes, self.drone_routes)
        self.log_heading("DTRC Solution")
        self.log_truck_routes(s_best[0])
        self.log_drone_routes(s_best[1])
        # Initialize the solution provided by DTRC as current solution
        s_current = copy.deepcopy(s_best)
        # Count of times improvement will be tried
        time = 0
        # Count of iteration
        i = 0
        while time <= self.max_times:
            while i <= self.max_iterations:
                self.log_heading(f"Trial #{time}")
                # Enter destroy phase
                self.log_heading(f"Iteration {i}:")
                self.log_message("Entering Destroy Phase.")
                s_destroyed, removed_nodes = self.destroy(copy.deepcopy(s_current))
                # Enter repair phase
                self.log_message("Entering Repair Phase.")
                s_repired, debug_str = self.repair(s_destroyed, removed_nodes)
                # Update current solution by repaired solution if the later one is better
                if self.get_cost(s_repired[0]) < self.get_cost(s_current[0]):
                    s_current = s_repired
                    self.log_message("Current Solution Updated. Iteration will reset.")
                    self.log_truck_routes(s_repired[0])
                    self.log_drone_routes(s_repired[1])
                    # reset iteration count if current solution is updated
                    i = 0
                    print("Current Updated. Iteration Reset. Current Cost: ",self.get_cost(s_current[0]))
                    print(debug_str)
                else:
                    i += 1
            # if the current solution is better than best solution        
            print("Current Cost : ", self.get_cost(s_current[0]))
            print("Best Cost : ", self.get_cost(s_best[0]))
            if self.get_cost(s_current[0]) < self.get_cost(s_best[0]):
                # update best solution
                s_best = s_current
                # reset iteration count
                self.log_message("Best Solution Updated. Iteration will reset.")
                self.log_truck_routes(s_best[0])
                self.log_drone_routes(s_best[1])
                i = 0
                print("Best Updated.")
                print(debug_str)
            else:
                # set current solution as provied by DTRC algorithm
                s_current = copy.deepcopy((self.truck_routes, self.drone_routes))
                # reset iteration count
                i = 0
            # try for next time
            time += 1
            print("Next time : ",time )
        return s_best
