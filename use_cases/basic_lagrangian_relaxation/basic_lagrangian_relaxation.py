import pathlib
import logging
import os
from wiffco.config import (
    parse_json_file,
    control_evaluation_system_from_json)
from wiffco.statistics import statistics_from_dict
from wiffco.optimization import (
    control_optimization_from_dict,
    LagrangianRelaxation)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

project_root = pathlib.Path(os.getcwd())

data_dir = project_root / "use_cases" / "basic_lagrangian_relaxation"

# System
json_path = data_dir / "control_evaluation_system.jsonc"
control_evaluation_system = control_evaluation_system_from_json(json_path=json_path)

# Ambient conditions
json_path = data_dir / "statistics_discrete_ambient.jsonc"
statistics_dict = parse_json_file(path=json_path)
ambient_statistics = statistics_from_dict(param_dict=statistics_dict)

json_path = data_dir / "optimization_lagrangian_relaxation.jsonc"
param_dict = parse_json_file(path=json_path)
lagrangian_relaxation: LagrangianRelaxation = \
    control_optimization_from_dict(param_dict=param_dict)

# Optimization
optimal_policy = lagrangian_relaxation.optimize_policy(
    control_eval_system=control_evaluation_system,
    ambient_statistics=ambient_statistics)