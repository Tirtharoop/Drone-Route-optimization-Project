import csv
import random
import copy

def read_distance_matrix(file_path):
    """Reads a distance matrix from a CSV file."""
    distance_matrix = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            distance_matrix.append([float(cell) if cell else 0.0 for cell in row])
    return distance_matrix

# Objective function: Calculates the total cost of the solution
def calculate_cost(solution, distance_matrix):
    cost = 0
    for route in solution:
        for i in range(len(route) - 1):
            cost += distance_matrix[route[i]][route[i + 1]]
    return cost

# Initial solution generator (e.g., a greedy approach)
def generate_initial_solution(customers, vehicle_count):
    solution = [[] for _ in range(vehicle_count)]
    for customer in customers:
        solution[random.randint(0, vehicle_count - 1)].append(customer)
    return solution

# Destroy part of the solution
def destroy(solution, percentage):
    new_solution = copy.deepcopy(solution)
    all_customers = [customer for route in new_solution for customer in route]
    customers_to_remove = random.sample(all_customers, int(len(all_customers) * percentage))

    for customer in customers_to_remove:
        for route in new_solution:
            if customer in route:
                route.remove(customer)
                break
    return new_solution, customers_to_remove

# Repair the solution
def repair(solution, removed_customers, distance_matrix):
    for customer in removed_customers:
        best_cost = float('inf')
        best_position = None
        best_route_index = None

        for route_index, route in enumerate(solution):
            for position in range(len(route) + 1):
                new_route = route[:position] + [customer] + route[position:]
                new_cost = calculate_cost([new_route], distance_matrix)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_position = position
                    best_route_index = route_index

        if best_position is not None:
            solution[best_route_index].insert(best_position, customer)

    return solution

# Large Neighborhood Search (LNS) Algorithm
def large_neighborhood_search(customers, vehicle_count, distance_matrix, iterations=100, destroy_percentage=0.2):
    current_solution = generate_initial_solution(customers, vehicle_count)
    best_solution = current_solution
    best_cost = calculate_cost(best_solution, distance_matrix)

    for _ in range(iterations):
        # Step 1: Destroy part of the solution
        destroyed_solution, removed_customers = destroy(current_solution, destroy_percentage)

        # Step 2: Repair the solution
        repaired_solution = repair(destroyed_solution, removed_customers, distance_matrix)

        # Step 3: Evaluate the new solution
        current_cost = calculate_cost(repaired_solution, distance_matrix)
        if current_cost < best_cost:
            best_solution = repaired_solution
            best_cost = current_cost

        # Accept or reject the new solution (simple acceptance criterion here)
        current_solution = repaired_solution

    return best_solution, best_cost

# Example usage
def main():
    # Read distance matrix from CSV file
    file_path = 'pairwise2.csv'  # Path to the uploaded CSV file
    distance_matrix = read_distance_matrix(file_path)

    # Given routes (from the input code)
    routes = [
        [0, 30, 6, 13, 1, 20, 11, 25, 24, 0],
        [0, 16, 31, 8, 26, 17, 4, 0],
        [0, 19, 15, 9, 2, 22, 32, 18, 27, 21, 14, 23, 0],
        [0, 29, 10, 28, 7, 3, 12, 0],
        [0, 5, 0]
    ]

    # Parameters
    num_vehicles = 2  # Set the number of drones to 2
    destroy_percentage = 0.2
    iterations = 100

    optimized_solutions = []

    for route in routes:
        customers = route[1:-1]  # Exclude depot (start and end point)
        optimized_solution, best_cost = large_neighborhood_search(
            customers,
            num_vehicles,
            distance_matrix,
            iterations,
            destroy_percentage
        )

        truck_route = [route[0]] + [node for subroute in optimized_solution for node in subroute] + [route[-1]]

        # Divide all customers into two drone routes after the truck route
        drone_routes = [[], []]
        for i, subroute in enumerate(optimized_solution):
            for node in subroute:
                drone_routes[i % 2].append(node)

        optimized_solutions.append((truck_route, drone_routes, best_cost))

    for i, (truck_route, drone_routes, cost) in enumerate(optimized_solutions, 1):
        print(f"\nOptimized Route {i}:")
        print(f"Truck Route: {truck_route}")
        print("Drone Routes:")
        for j, drone_route in enumerate(drone_routes, 1):
            print(f"  Drone {j}: {drone_route}")
        print(f"Cost: {cost}")

if __name__ == "__main__":
    main()
