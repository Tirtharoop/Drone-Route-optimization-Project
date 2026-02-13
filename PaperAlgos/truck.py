from drone import Drone
from PaperAlgos.route import *

class Truck:
    def __init__(self, id, drones, route, capacity, speed = 1) -> None:
        self.id = id
        self.drones : list[Drone] = drones
        self.route = TruckRoute(route, self)
        self.capacity = capacity
        self.speed = speed

    def __repr__(self) -> str:
        truck_str = f"\nTruck #{self.id} : {str(self.route)}"
        for drone in self.drones:
            truck_str += str(drone)
        return truck_str + '\n'
        