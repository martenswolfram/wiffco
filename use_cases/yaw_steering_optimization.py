import numpy as np
import logging
import pathlib
from wiffco.config import parse_csv_file, parse_json_file
from wiffco.statistics import statistics_from_table
from wiffco.interface import Ambient, DataTable
from wiffco.optimization import (
    ControlEvaluationSystem,
    control_optimization_from_dict
)
from wiffco.plant_model import plant_model_from_dict
from wiffco.aggregation import aggregation_from_dict
from wiffco.metrics_accumulation import metrics_accumulation_from_dict
from wiffco.constraint import constraint_from_dict
from wiffco.multi_metrics_reduction import multi_metrics_reduction_from_dict


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Config data
data_set = "from_bart"

current_file = pathlib.Path(__file__).resolve()
project_root = current_file.parent.parent

data_dir = project_root / "tests/data" / data_set

# Read wind rose statistics
csv_path = data_dir / "wind_rose.csv"
statistics_data = parse_csv_file(csv_path=csv_path,
                                 delimiter=";")
support_names={"wind_dir": Ambient.WIND_DIRECTION_DEG,
               "wind_speed": Ambient.WIND_SPEED_MPS,
               "turbulence_intensity": Ambient.TURBULENCE_INTENSITY}
prevalence_name="prevalence"             
wind_rose_statistics = statistics_from_table(
    data_dict=statistics_data,
    support_names=support_names,
    prevalence_name=prevalence_name,
    statistics_name=f"{data_set}_statistics"
)

# # Specify custom grid
# wind_speeds = np.arange(0, 30, 2)
# wind_directions = np.arange(0, 360, 30)
# ws, wd = np.meshgrid(wind_speeds, wind_directions, indexing="ij")
# data_matrix = np.stack((ws.flatten(), wd.flatten()))
# custom_support = DataTable(
#     data={Ambient.WIND_SPEED_MPS: ws.flatten(),
#           Ambient.WIND_DIRECTION_DEG: wd.flatten(),
#           Ambient.TURBULENCE_INTENSITY: np.zeros(shape=(ws.size,))})
# custom_statistics = wind_rose_statistics.map_to_discrete(support=custom_support)

# Specify optimization problem
# Plant model
json_path = data_dir / "model_wf_surrogate.jsonc"
param_dict = parse_json_file(path=json_path)
floris_power_del_model = plant_model_from_dict(param_dict=param_dict)

# Aggregation
json_path = project_root / "use_cases" / "aggregation_symbolic.jsonc"
param_dict = parse_json_file(path=json_path)
aggregation = aggregation_from_dict(param_dict=param_dict)

# Accumulation
json_path = project_root / "use_cases" / "discounted_integration.jsonc"
param_dict = parse_json_file(path=json_path)
metrics_accumulation = metrics_accumulation_from_dict(param_dict=param_dict)

# Constraints
json_path = project_root / "use_cases" / "constraint_control.jsonc"
param_dict = parse_json_file(path=json_path)
constraint_control = constraint_from_dict(param_dict=param_dict)

# Metrics reduction
json_path = project_root / "use_cases" / "scalar_weighting.jsonc"
param_dict = parse_json_file(path=json_path)
metrics_reduction = multi_metrics_reduction_from_dict(param_dict=param_dict)

# Optimization
# json_path = project_root / "use_cases" / "optimization_lagrangian_relaxation.jsonc"
json_path = project_root / "use_cases" / "optimization_simultaneous.jsonc"
param_dict = parse_json_file(path=json_path)
control_optimization = control_optimization_from_dict(param_dict=param_dict)

ctrl_eval = ControlEvaluationSystem(
    name="yaw_steering_optimization",
    plant_model=floris_power_del_model,
    aggregation=aggregation,
    metrics_accumulation=metrics_accumulation,
    constraint_control=constraint_control,
    constraint_aggregate=None,
    constraint_accumulated=None,
    multi_metrics_reduction=metrics_reduction
)

# Optimization
optimal_policy = control_optimization.optimize_policy(
    control_eval_system=ctrl_eval,
    # ambient_statistics=custom_statistics,
    ambient_statistics=wind_rose_statistics
)
pass