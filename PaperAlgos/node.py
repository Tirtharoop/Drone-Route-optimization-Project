
class Node:
    def __init__(self, id, demand, distances, truck_served = None, drone_served = None,
                  is_launch_node = False, is_land_node = False) -> None:
        self.id = id
        self.demand = demand
        self.distances = distances
        self.truck_served = truck_served
        self.drone_served = drone_served
        self.is_launch_node = is_launch_node
        self.is_land_node = is_land_node
    
    def get_distance_from_node(self, node_id):
        return self.distances[node_id]
    
    def update_truck_served(self, truck):
        self.truck_served = truck

    def update_drone_served(self, drone):
        self.drone_served = drone
