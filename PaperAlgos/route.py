from node import Node
from drone import Drone
from truck import Truck

class Route:
    def __init__(self, nodes) -> None:
        self.nodes : list[Node] = nodes
    
    def __repr__(self) -> str:
        return " ".join([str(node.id) for node in self.nodes])

    def get_distance_between_nodes(self, src_idx, dst_idx):
        distance = 0
        for i in range(src_idx, dst_idx):
            distance += self.nodes[i].get_distance_from_node(self.nodes[i+1])
        return distance
    
    def get_length(self):
        return self.get_distance_between_nodes(0, len(self.nodes)-1)
    
    def get_all_demands_weight(self):
        weight = 0
        for node in self.nodes[1:-1]:       # exclude the terminal nodes
            weight += node.demand
        return weight
    
    def remove_node(self, node: Node):
        self.nodes.remove(node)

    def insert_node(self, node, index):
        self.nodes.insert(index, node)

class DroneRoute(Route):
    def __init__(self, nodes, assigned_drone = None) -> None:
        super().__init__(nodes)
        self.assigned_drone : Drone = assigned_drone

    def update_assigned_drone(self, drone):
        self.assigned_drone = drone

    def get_travel_time(self):
        return self.get_length() / self.assigned_drone.speed
    
    def get_total_flight_time(self):
        total_service_time = len(self.nodes[1:-1]) * self.assigned_drone.service_time
        return self.assigned_drone.launch_time + self.get_travel_time() + total_service_time + self.assigned_drone.land_time
    
    
class TruckRoute(Route):
    def __init__(self, nodes, assigned_truck = None) -> None:
        super().__init__(nodes)
        self.assigned_truck : Truck = assigned_truck
    
    def update_assigned_truck(self, truck):
        self.assigned_truck = truck

    def get_travel_time(self):
        return self.get_length() / self.assigned_truck.speed
    
    def get_travel_time_between_nodes(self, src: Node, dst: Node):
        src_idx = self.nodes.index(src)
        dst_idx = self.nodes.index(dst)
        return super().get_distance_between_nodes(src_idx, dst_idx) / self.assigned_truck.speed
        