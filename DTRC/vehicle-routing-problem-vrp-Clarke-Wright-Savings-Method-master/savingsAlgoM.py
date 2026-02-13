import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Function to read the file and extract relevant data
def read_cvrp_file(file_path):
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

    return num_trucks, capacity, x_coords, y_coords, demands

# Function to calculate distance matrix
def distance_matrix_from_xy(x_coordinates, y_coordinates):
    n = len(x_coordinates)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            dist = math.sqrt((x_coordinates[i] - x_coordinates[j])**2 + (y_coordinates[i] - y_coordinates[j])**2)
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist

    return pd.DataFrame(dist_matrix)

# Clarke-Wright Savings Algorithm
def clarke_wright_savings(num_trucks, capacity, x_coords, y_coords, demands):
    n_patients = len(demands) - 1  
    distances_df = distance_matrix_from_xy(x_coords, y_coords)

    nodes = pd.DataFrame({'d0': distances_df.iloc[:, 0], 'demand': [demands[i + 1] for i in range(len(demands))]})

    savings = {}
    for r in distances_df.index[1:]:
        for c in distances_df.columns[1:]:
            if r != c:
                key = f'({r},{c})'
                savings[key] = distances_df.iloc[r, 0] + distances_df.iloc[c, 0] - distances_df.iloc[r, c]

    sv = pd.DataFrame.from_dict(savings, orient='index', columns=['saving']).sort_values(by='saving', ascending=False)

    routes = []
    node_list = list(nodes.index)
    node_list.remove(0)

    def sum_cap(route):
        return sum(nodes.demand[node] for node in route)

    def merge(route0, route1):
        return route0 + route1

    for link in sv.index:
        n1, n2 = [int(n) for n in link.strip("()").split(",")]

        route_indices = [-1, -1]
        for i, route in enumerate(routes):
            if n1 in route:
                route_indices[0] = i
            if n2 in route:
                route_indices[1] = i

        if route_indices[0] == route_indices[1] == -1:
            if sum_cap([n1, n2]) <= capacity:
                routes.append([n1, n2])
                node_list.remove(n1)
                node_list.remove(n2)

        elif route_indices[0] != -1 and route_indices[1] == -1:
            route_idx = route_indices[0]
            if sum_cap(routes[route_idx] + [n2]) <= capacity:
                routes[route_idx].append(n2)
                node_list.remove(n2)

        elif route_indices[0] == -1 and route_indices[1] != -1:
            route_idx = route_indices[1]
            if sum_cap(routes[route_idx] + [n1]) <= capacity:
                routes[route_idx].append(n1)
                node_list.remove(n1)

        elif route_indices[0] != -1 and route_indices[1] != -1 and route_indices[0] != route_indices[1]:
            if sum_cap(routes[route_indices[0]] + routes[route_indices[1]]) <= capacity:
                merged_route = merge(routes[route_indices[0]], routes[route_indices[1]])
                for index in sorted(route_indices, reverse=True):
                    routes.pop(index)
                routes.append(merged_route)

    for node in node_list:
        routes.append([node])

    for route in routes:
        route.insert(0, 0)
        route.append(0)

    total_distance = sum(sum(distances_df.iloc[route[i], route[i+1]] for i in range(len(route)-1)) for route in routes)

    return routes, total_distance

# Local Search Heuristics
def route_distance(route, dist_matrix):
    return sum(dist_matrix.iloc[route[i], route[i+1]] for i in range(len(route)-1))

def two_opt(route, dist_matrix):
    best_route = route[:]
    best_distance = route_distance(best_route, dist_matrix)

    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                new_distance = route_distance(new_route, dist_matrix)
                if new_distance < best_distance:
                    best_route, best_distance = new_route, new_distance
                    improved = True
    return best_route

def relocate(route, dist_matrix):
    best_route = route[:]
    best_distance = route_distance(best_route, dist_matrix)

    for i in range(1, len(route) - 1):
        node = best_route[i]
        temp_route = best_route[:i] + best_route[i+1:]

        for j in range(1, len(temp_route)):
            new_route = temp_route[:j] + [node] + temp_route[j:]
            new_distance = route_distance(new_route, dist_matrix)

            if new_distance < best_distance:
                best_route, best_distance = new_route, new_distance

    return best_route

def swap_move(routes, dist_matrix):
    best_routes = [r[:] for r in routes]
    best_distance = sum(route_distance(r, dist_matrix) for r in best_routes)

    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            route1, route2 = routes[i], routes[j]

            for idx1 in range(1, len(route1) - 1):
                for idx2 in range(1, len(route2) - 1):
                    new_route1 = route1[:]
                    new_route2 = route2[:]
                    new_route1[idx1], new_route2[idx2] = new_route2[idx2], new_route1[idx1]

                    new_total_distance = (route_distance(new_route1, dist_matrix) +
                                          route_distance(new_route2, dist_matrix))

                    if new_total_distance < best_distance:
                        best_routes[i], best_routes[j] = new_route1, new_route2
                        best_distance = new_total_distance

    return best_routes

def local_search(routes, dist_matrix):
    improvement = True

    while improvement:
        new_routes = [min([two_opt(r, dist_matrix), relocate(r, dist_matrix)], key=lambda r: route_distance(r, dist_matrix)) for r in routes]
        new_routes = swap_move(new_routes, dist_matrix)

        if sum(route_distance(r, dist_matrix) for r in new_routes) < sum(route_distance(r, dist_matrix) for r in routes):
            routes = new_routes
        else:
            improvement = False

    return routes


# Main Execution
if __name__ == "__main__":
    file_path = "A-n32-k5.vrp"  # Change to your actual file path

    num_trucks, capacity, x_coords, y_coords, demands = read_cvrp_file(file_path)

    routes, total_distance = clarke_wright_savings(num_trucks, capacity, x_coords, y_coords, demands)

    distances_df = distance_matrix_from_xy(x_coords, y_coords)
    optimized_routes = local_search(routes, distances_df)

    new_total_distance = sum(route_distance(route, distances_df) for route in optimized_routes)

    print("\n------")
    print("Optimized Routes:")
    for route in optimized_routes:
        print(route)
    print(f"Original Total Distance: {total_distance:.2f}")
    print(f"Optimized Total Distance: {new_total_distance:.2f}")

    plt.figure(figsize=(10, 6))
    for route in optimized_routes:
        for k in range(len(route)-1):
            i, j = route[k], route[k+1]
            plt.plot([x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]], 'g-')

    plt.scatter(x_coords[1:], y_coords[1:], c='b', label="Customers")
    plt.scatter(x_coords[0], y_coords[0], c='r', marker='^', label="Depot")
    plt.title('Optimized CVRP Solution')
    plt.legend()
    plt.show()
