import pathlib
from twain_wifco.config import (
    control_evaluation_system_from_json,
    control_optimization_from_json,
    statistics_from_json)

# Config folder path
config_folder = pathlib.Path(__file__).parent / "tests/data/"
    
# Control evaluation System
json_path = config_folder / "control_evaluation_system.json"
control_evaluation_system = control_evaluation_system_from_json(json_path=json_path)

# Ambient conditions
json_path = config_folder / "statistics_discrete_ambient.json"
ambient_statistics = statistics_from_json(json_path=json_path)

# Evaluation period
duration = 20

for file_name in ["optimization_grid_search.json",
                  "optimization_simultaneous.json",
                  "optimization_lagrangian_relaxation.json"]:
    json_path = config_folder / file_name
    optimization = control_optimization_from_json(json_path=json_path)
    
    # Optimization
    optimal_policy = optimization.optimize_policy(
        control_eval_system=control_evaluation_system,
        ambient_condition_statistics=ambient_statistics,
        duration=duration)

    # output

    print(f"\nOptimal control set points for file '{file_name}':")
    print(optimal_policy.control_setpoints.T)

