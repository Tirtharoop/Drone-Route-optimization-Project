from PaperAlgos.route import DroneRoute

class Drone:
    def __init__(self, id, capacity, routes, host_truck = None,
                 speed=1.5, service_time = 1, launch_time = 1, land_time = 1):
        self.id = id
        self.host_truck  = host_truck
        self.routes = [DroneRoute(route, self) for route in routes]
        self.capacity = capacity
        self.speed = speed
        self.service_time = service_time
        self.launch_time = launch_time
        self.land_time = land_time
    
    def __str__(self) -> str:
        drone_str = f"\nDrone #{self.id} : {', '.join([str(route) for route in self.routes])}"
        return drone_str
