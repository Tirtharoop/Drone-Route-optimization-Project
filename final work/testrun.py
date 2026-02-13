


import traci
import random
import os
import xml.etree.ElementTree as ET
import pandas as pd

# Step 1: Define helper functions
def get_buildings_from_net(net_file):
    """
    Extract building polygons from the SUMO network file.
    """
    print("Extracting buildings from the network file...")
    if not os.path.exists(net_file):
        print(f"Error: Network file '{net_file}' not found.")
        return []

    tree = ET.parse(net_file)
    root = tree.getroot()
    buildings = []
    for poly in root.findall("poly"):
        if "building" in poly.attrib.get("type", ""):
            buildings.append({
                "id": poly.attrib["id"],
                "coords": poly.attrib["shape"]
            })
    print(f"Found {len(buildings)} buildings in the network file.")
    return buildings


def select_valid_buildings(buildings):
    """
    Select two random buildings with valid edges.
    """
    print("Selecting valid building pairs...")
    valid_pairs = []
    for i in range(len(buildings)):
        for j in range(i + 1, len(buildings)):
            start_coords = get_building_coords(buildings[i])
            end_coords = get_building_coords(buildings[j])
            start_edge = find_valid_edge(start_coords)
            end_edge = find_valid_edge(end_coords)
            if start_edge and end_edge:
                valid_pairs.append((buildings[i], buildings[j]))
    if not valid_pairs:
        print("No valid building pairs found.")
        raise ValueError("No valid building pairs found in the network.")
    print(f"Found {len(valid_pairs)} valid building pairs.")
    return random.choice(valid_pairs)


def calculate_route(start_coords, end_coords):
    """
    Use SUMO TraCI to calculate the shortest route between two coordinates.
    """
    print("Calculating route between selected buildings...")
    start_edge = find_valid_edge(start_coords)
    end_edge = find_valid_edge(end_coords)

    if not start_edge or not end_edge:
        raise ValueError(f"Invalid edge(s): start_edge={start_edge}, end_edge={end_edge}")

    return traci.simulation.findRoute(start_edge, end_edge)


def find_valid_edge(coords):
    """
    Validate and find the closest edge for given coordinates.
    """
    try:
        edge = traci.simulation.convertRoad(coords[0], coords[1])
        if edge:
            return edge
    except traci.exceptions.TraCIException:
        print(f"Invalid coordinates {coords}: No edge found.")
    return None


def get_building_coords(building):
    """
    Extract the first coordinate from a building's shape attribute.
    """
    shape = building["coords"]
    first_coord = shape.split()[0]  # Take the first pair of coordinates
    return tuple(map(float, first_coord.split(",")))  # Convert to float tuple


# Step 2: Initialize and run the simulation
def run_simulation():
    print("Starting SUMO simulation...")
    sumo_cmd = ["sumo", "-c", "osm.sumocfg"]  # Replace 'sumo' with 'sumo-gui' for GUI mode

    # Check if SUMO configuration file exists
    if not os.path.exists("osm.sumocfg"):
        print("Error: SUMO configuration file 'osm.sumocfg' not found.")
        return

    try:
        traci.start(sumo_cmd)
    except Exception as e:
        print(f"Error starting SUMO simulation: {e}")
        return

    net_file = "osm.net.xml"
    buildings = get_buildings_from_net(net_file)
   
    if len(buildings) < 2:
        print("Not enough buildings found in the map.")
        traci.close()
        return

    try:
        # Select two valid buildings with valid edges
        start_building, end_building = select_valid_buildings(buildings)
    except ValueError as e:
        print(f"Error: {e}")
        traci.close()
        return

    start_coords = get_building_coords(start_building)
    end_coords = get_building_coords(end_building)

    print(f"Selected Start Building: {start_building['id']}")
    print(f"Selected End Building: {end_building['id']}")

    try:
        # Calculate the route between the buildings
        route = calculate_route(start_coords, end_coords)
    except ValueError as e:
        print(f"Error calculating route: {e}")
        traci.close()
        return

    print(f"Generated Route: {route.edges}")
    traci.route.add("randomRoute", route.edges)
    traci.vehicle.add("veh1", "randomRoute")
    traci.vehicle.setColor("veh1", (255, 0, 0))  # Red vehicle

    simulation_data = []
    print("Running the simulation...")
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        veh_pos = traci.vehicle.getPosition("veh1")
        veh_speed = traci.vehicle.getSpeed("veh1")
        simulation_data.append([traci.simulation.getTime(), veh_pos, veh_speed])
        print(f"Time: {traci.simulation.getTime()} | Pos: {veh_pos} | Speed: {veh_speed}")

    traci.close()

    # Export simulation data to an Excel file
    print("Exporting simulation data to simulation_output.xlsx...")
    df = pd.DataFrame(simulation_data, columns=["Time", "Position", "Speed"])
    df.to_excel("simulation_output.xlsx", index=False)
    print("Simulation completed and data exported to simulation_output.xlsx.")


# Run the simulation
if __name__ == "__main__":
    run_simulation()