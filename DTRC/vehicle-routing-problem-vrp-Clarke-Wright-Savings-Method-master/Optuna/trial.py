import optuna
import math

road_points = [(0, 0), (10, 5), (20, 10), (30, 15), (40, 20), (50, 25), (60, 30), (70, 35), (80, 40), (90, 45)]
static_point = (25, 30)
drone_speed = 10.0
truck_speed = drone_speed / 1.5

def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def get_road_distance(a_idx, b_idx):
    return euclidean(road_points[a_idx], road_points[b_idx])

def evaluate(takeoff_idx, landing_idx):
    takeoff = road_points[takeoff_idx]
    landing = road_points[landing_idx]

    d1 = euclidean(takeoff, static_point)
    d2 = euclidean(static_point, landing)
    drone_distance = d1 + d2
    drone_time = drone_distance / drone_speed

    truck_distance = get_road_distance(takeoff_idx, landing_idx)
    truck_time = truck_distance / truck_speed

    if abs(truck_time - drone_time) > 1.0:
        return float('inf')

    return drone_distance


def objective(trial):
    takeoff_idx = trial.suggest_int("takeoff_idx", 0, len(road_points) - 1)
    landing_idx = trial.suggest_int("landing_idx", 0, len(road_points) - 1)

    if takeoff_idx == landing_idx:
        return float('inf')

    return evaluate(takeoff_idx, landing_idx)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=1000)

best = study.best_params
print("Best Takeoff Point:", road_points[best['takeoff_idx']])
print("Best Landing Point:", road_points[best['landing_idx']])
print("Best Drone Distance:", study.best_value)
