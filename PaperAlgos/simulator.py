import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from dtrc import DTRC
from lns import LNS
import sys

def distance_matrix_from_xy(x_coords, y_coords):
    """Calculate distance matrix from coordinates."""
    n = len(x_coords)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = math.sqrt((x_coords[i] - x_coords[j])**2 + (y_coords[i] - y_coords[j])**2)
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
    return pd.DataFrame(dist_matrix)

def read_cvrp_file(file_path):
    """Read and parse the CVRP file format."""
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

    return num_trucks, capacity, x_coords, y_coords, demands, distance_matrix_from_xy(x_coords, y_coords)

def plot_truck_routes(truck_routes, x_coords, y_coords):
    """Plot truck routes after Clarke-Wright Savings."""
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
    
def plot_combined_routes(truck_routes, all_drone_routes, x_coords, y_coords, algo_name):
    """Plot combined truck and drone routes."""
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
    plt.title(algo_name+" - Truck and Drone Routes")
    plt.grid(True)
    plt.show()

def get_drone_payload_of_truck(drones, demands):
    payload = 0
    for drone_routes in drones:
        for route in drone_routes:
            payload += get_payload(route, demands)
    return payload

def validate_truck_routes(truck_routes, drone_routes, truck_capacity, demands):
    for truck_idx, route in enumerate(truck_routes):
        payload = get_payload(route, demands) + get_drone_payload_of_truck(drone_routes[truck_idx], demands)
        if payload > truck_capacity:
            print(f"Truck #{truck_idx+1} is Overloaded! Capacity : {truck_capacity}. Payload : {payload}")
            print(f"Truck Route : {' '.join([str(node) for node in route])}")
            print(f"Drone Routes : ",drone_routes[truck_idx])
            return False
    return True

def get_payload(route, demands):
    payload = 0
    for node in route[1:-1]:    # exclude terminal nodes
        payload += demands[node+1]
    return payload

def validate_battery_capacity(route, payload, demands, distnaces):
    battery = 100
    # print(f"Drone Payload : {payload}")
    for i in range(1, len(route)):
        depletion = distnaces[route[i-1]][route[i]] * ((200 + payload)**1.5)
        percentage = depletion / 1558.85
        battery -= (distnaces[route[i-1]][route[i]] * ((200 + payload)**1.5)) / 1558.85
        print(f"Edge Travelled: ({route[i-1]}, {route[i]}), Distance : {distnaces[route[i-1]][route[i]]:.2f}, Paylaod : {payload}, Depletion Percentage : {percentage:.2f}, Remaining Battery : {battery:.2f}")
        if battery < 0:
            return False
        payload -= demands[route[i]+1]
    return True


def validate_drone_routes(drone_routes, drone_capacity, demands, distnaces):
    for truck_idx, truck_drones in enumerate(drone_routes):
        for drone_idx, drone_routes in enumerate(truck_drones):
            for route_idx, route in enumerate(drone_routes):
                payload = get_payload(route, demands)
                if payload > drone_capacity:
                    print(f"Drone #{drone_idx+1} of Truck #{truck_idx+1} is overloaded for route #{route_idx+1}.")
                    print(f"Drone Capacity : {drone_capacity}. Payload : {payload}.")
                    return False
                if not validate_battery_capacity(route, payload, demands, distnaces):
                    print(f"Battery exhausted for Drone #{drone_idx+1} of Truck #{truck_idx+1} in route #{route_idx+1}.")
                    print(f"Drone Route : {' '.join([str(node) for node in route])}")
                    return False
    return True

def validate_solution(truck_routes, drone_routes, truck_capacity, drone_capacity, demands, distnaces):
    return validate_truck_routes(truck_routes, drone_routes, truck_capacity, demands) and validate_drone_routes(drone_routes, drone_capacity, demands, distnaces)

def run_algorithms(file_path, drone_capacity, instance_name, show_plots = False):
    print("Input Instance : ", instance_name)
    truck_count, truck_capacity, x_coords, y_coords, demands, distance_matrix = read_cvrp_file(file_path)
    dtrc_solver = DTRC(truck_count, truck_capacity, x_coords, y_coords, demands, distance_matrix, drone_capacity)
    dtrc_total_cost, dtrc_delivery_cost, dtrc_truck_routes, dtrc_drone_routes = dtrc_solver.solve()
    if show_plots:
        plot_combined_routes(dtrc_truck_routes, dtrc_drone_routes, x_coords, y_coords, "DTRC")
    lns_runner = LNS(dtrc_truck_routes, dtrc_drone_routes, demands, distance_matrix, truck_capacity= truck_capacity, drone_capacity=drone_capacity)
    lns_solution = lns_runner.run()
    if show_plots:
        plot_combined_routes(lns_solution[0], lns_solution[1], x_coords, y_coords, "LNS")
    dtrc_cost = lns_runner.get_cost(dtrc_truck_routes)
    lns_cost = lns_runner.get_cost(lns_solution[0])
    print ("Instance = ", instance_name)
    print ("DTRC Cost = ", dtrc_cost)
    print ("LNS Cost = ", lns_cost)
    # print("LNS Truck Routes : ", lns_solution[0])
    path = "PaperAlgos/result/"+instance_name+".prm"
    with open(path, "w") as f:
        f.write("Demands\n")
        f.write("-------\n\n")
        f.write(str(demands))
        f.write("\n\nDistance Matrix\n")
        f.write("---------------\n\n")
        pd.set_option('display.max_columns', None)
        f.write(str(distance_matrix))
    
    if validate_solution(dtrc_truck_routes, dtrc_drone_routes, truck_capacity, drone_capacity, demands, distance_matrix):
        print("DTRC solution is Valid.")
    else:
        print("DTRC solution is Inalid!")
        
    if validate_solution(lns_solution[0], lns_solution[1], truck_capacity, drone_capacity, demands, distance_matrix):
        print("LNS solution is Valid.")
    else:
        print("LNS solution is Inalid!")

    return (dtrc_cost, dtrc_truck_routes, dtrc_drone_routes) , (lns_cost, lns_solution[0], lns_solution[1])

def get_inputs():
    # inputs are given as array of filepaths, drone capacaity
    inputs = [
        ("DTRC/A/A-n32-k5.vrp", 35),]
        # ("DTRC/A/A-n33-k5.vrp", 35),
    #     ("DTRC/A/A-n33-k6.vrp", 35),
    #     ("DTRC/A/A-n34-k5.vrp", 35),
    #     ("DTRC/A/A-n36-k5.vrp", 35),
    #     ("DTRC/A/A-n37-k5.vrp", 35),
    #     ("DTRC/A/A-n37-k6.vrp", 35),
    #     ("DTRC/A/A-n38-k5.vrp", 35),
    #     ("DTRC/A/A-n39-k5.vrp", 35),
    #     ("DTRC/A/A-n39-k6.vrp", 35),
    #     ("DTRC/A/A-n44-k6.vrp", 35),
    #     ("DTRC/A/A-n45-k6.vrp", 35),
    #     ("DTRC/A/A-n45-k7.vrp", 35),
    #     ("DTRC/A/A-n46-k7.vrp", 35),
    #     ("DTRC/A/A-n48-k7.vrp", 35),
    #     ("DTRC/A/A-n53-k7.vrp", 35),
    #     ("DTRC/A/A-n54-k7.vrp", 35),
    #     ("DTRC/A/A-n55-k9.vrp", 35),
    #     ("DTRC/A/A-n60-k9.vrp", 35),
    #     ("DTRC/A/A-n61-k9.vrp", 35),
    #     ("DTRC/A/A-n62-k8.vrp", 35),
    #     ("DTRC/A/A-n63-k9.vrp", 35),
    #     ("DTRC/A/A-n63-k10.vrp", 35),
    #     ("DTRC/A/A-n64-k9.vrp", 35),
    #     ("DTRC/A/A-n65-k9.vrp", 35),
    #     ("DTRC/A/A-n69-k9.vrp", 35),
    #     ("DTRC/B/B-n31-k5.vrp", 35),
    #     ("DTRC/B/B-n34-k5.vrp", 35),
    #     ("DTRC/B/B-n35-k5.vrp", 35),
    #     ("DTRC/B/B-n38-k6.vrp", 35),
    #     ("DTRC/B/B-n39-k5.vrp", 35),
    #     ("DTRC/B/B-n41-k6.vrp", 35),
    #     ("DTRC/E/E-n51-k5.vrp", 50),
    #     ("DTRC/E/E-n76-k7.vrp", 55),
    #     ("DTRC/E/E-n76-k8.vrp", 45),
    #     ("DTRC/E/E-n76-k10.vrp", 40),
    #     ("DTRC/E/E-n76-k14.vrp", 35),
    #     ("DTRC/P/P-n16-k8.vrp", 20),
    #     ("DTRC/P/P-n19-k2.vrp", 40),
    #     ("DTRC/P/P-n20-k2.vrp", 40),
    #     ("DTRC/P/P-n21-k2.vrp", 40),
    #     ("DTRC/P/P-n22-k2.vrp", 40),
    #     ("DTRC/P/P-n23-k8.vrp", 20),
    #     ("DTRC/P/P-n40-k5.vrp", 40),
    #     ("DTRC/P/P-n45-k5.vrp", 40),
    #     ("DTRC/P/P-n50-k7.vrp", 40),
    #     ("DTRC/P/P-n50-k8.vrp", 40),
    #     ("DTRC/P/P-n50-k10.vrp", 35),
    #     ("DTRC/P/P-n51-k10.vrp", 30),
    #     ("DTRC/P/P-n55-k7.vrp", 45),
    #     ("DTRC/P/P-n55-k10.vrp", 40),
    #     ("DTRC/P/P-n55-k15.vrp", 38),
    #     ("DTRC/P/P-n60-k10.vrp", 40),
    #     ("DTRC/P/P-n60-k15.vrp", 30),
    #     ("DTRC/P/P-n65-k10.vrp", 40),
    #     ("DTRC/P/P-n70-k10.vrp", 40)
    # ]
    return inputs

def write_file(file_path, solution):
    with open(file_path, "w") as f:
        f.write(f"Cost : {solution[0]}\n")
        f.write("\nTruck Routes")
        f.write("\n------------")
        for idx, route in enumerate(solution[1]):
            route_str = " ".join([str(node) for node in route])
            f.write(f"\nTruck Route #{idx+1} : { route_str }")
        f.write("\n\nDrone Routes")
        f.write("\n------------")
        for t_idx, drone_routes in enumerate(solution[2]):
            for d_idx, routes in enumerate(drone_routes):
                for r_idx, route in enumerate(routes):
                    route_str = " ".join(str(node) for node in route)
                    f.write(f"\nTruck #{t_idx+1} - Drone #{d_idx+1} - Route #{r_idx+1} : {route_str}")
        f.close()

def write_outputs(outputs, show_summary=True):
    path = "PaperAlgos/result/"
    for instnace, dtrc_solution, lns_solution in outputs:
        write_file(path+instnace+".dtrc", dtrc_solution)
        write_file(path+instnace+".lns", lns_solution)
        if show_summary:
            print("Instance : ", instnace)
            print("DTRC Cost : ", dtrc_solution[0])
            print("LNS Cost : ", lns_solution[0])

def simulate_Algorithms():
    inputs = get_inputs()
    outputs = []
    for input in inputs:
        instance_name = input[0].split("/")[-1].split(".")[0]
        dtrc_solution, lns_solution = run_algorithms(input[0], input[1], instance_name)
        outputs.append((instance_name, dtrc_solution, lns_solution))
    write_outputs(outputs)

simulate_Algorithms()
        
        



