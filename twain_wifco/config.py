import json5 as json
import pathlib
import csv
import ast
import numpy as np
from typing import Dict
from twain_wifco.interface import (
    DataTable,
    DataType)
from twain_wifco.plant_model import (
    ModelType,
    factorized_scattered_interp_from_dict,
    symbolic_model_from_dict)
from twain_wifco.floris_model.floris_model import (
    floris_model_from_dict
)
from twain_wifco.statistics import (
    StatisticsType,
    discrete_statistics_from_dict,
    DiscreteStatistics)
from twain_wifco.control_policy import (
    discrete_control_policy_from_dict)
from twain_wifco.aggregation import (
    AggregationType,
    simple_product_from_dict)
from twain_wifco.metrics_accumulation import (
    MetricsAccumulationType,
    discounted_integrator_from_dict)
from twain_wifco.constraint import (
    ConstraintType,
    separate_constraints_from_dict)
from twain_wifco.multi_metrics_reduction import (
    MultiMetricsReductionType,
    scalar_weighting_from_dict)
from twain_wifco.optimization import (
    ControlEvaluationSystem,
    OptimizationMethod,
    grid_search_from_dict,
    simultaneous_optimization_from_dict,
    lagrangian_relaxation_from_dict)


def parse_json_file(path):
    with open(path, "r") as json_file:
        data_dict = json.load(json_file)
        return data_dict
    
def discrete_statistics_from_csv(csv_path: pathlib.Path,
                                 support_names: Dict[str, DataType],
                                 prevalence_name: str,
                                 delimiter: str = ";",
                                 statistics_name: str = "statistics_from_csv"):
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        # Header
        header = next(reader)
        columns = {h_str: [] for h_str in header}

        # Read rows
        for row in reader:
            for h_str, value in zip(header, row):
                columns[h_str].append(np.array(ast.literal_eval(value)))

    data = {}
    prevalence = None
    for h_str, col in columns.items():
        if h_str in support_names:
            key = support_names[h_str]
            data[key] = np.stack(col, axis=0)
        elif h_str == prevalence_name:
            prevalence = np.array(col)
    order = list(data.keys())
    
    if prevalence is not None and len(data):
        probabilities = prevalence / np.sum(prevalence)
        support_data = DataTable(data, order)
        return DiscreteStatistics(
            name=statistics_name,
            support_data=support_data,
            probabilities=probabilities)
    else:
        raise ValueError("Could not parse CSV data to statistics")
        
def statistics_from_json(json_path: pathlib.Path):
    param_dict = parse_json_file(path=json_path)
    statistics_type = StatisticsType(param_dict["statistics_type"])
    if statistics_type == StatisticsType.DISCRETE_STATISTICS:
        return discrete_statistics_from_dict(param_dict=param_dict)
    else:
        raise NotImplementedError("Only discrete_statistics implemented.")

def plant_model_from_json(json_path: pathlib.Path):
    param_dict = parse_json_file(path=json_path)
    model_type = ModelType(param_dict["model_type"])
    if model_type == ModelType.FACTORIZED_SCATTERED_INTERPOLATOR:
        return factorized_scattered_interp_from_dict(
            param_dict=param_dict)
    elif model_type == ModelType.SYMBOLIC:
        return symbolic_model_from_dict(
            param_dict=param_dict)
    elif model_type == ModelType.FLORIS:
        return floris_model_from_dict(
            param_dict=param_dict)
    else:
        raise NotImplementedError(f"Only factorized_scattered_interpolator and symbolic"
                                  f" models implemented.")

def control_policy_from_json(json_path: pathlib.Path):
    param_dict = parse_json_file(path=json_path)
    return discrete_control_policy_from_dict(param_dict=param_dict)
    
def aggregation_from_json(json_path: pathlib.Path):
    param_dict = parse_json_file(path=json_path)
    aggregation_type = AggregationType(param_dict["aggregation_type"])
    if aggregation_type == AggregationType.SIMPLE_PRODUCT:
        return simple_product_from_dict(param_dict=param_dict)
    else:
        raise NotImplementedError("Only simple_product aggregation implemented.")

def metrics_accumulation_from_json(json_path: pathlib.Path):
    param_dict = parse_json_file(path=json_path)
    accumulation_type = MetricsAccumulationType(param_dict["accumulation_type"])
    if accumulation_type == MetricsAccumulationType.DISCOUNTED_INTEGRATION:
        return discounted_integrator_from_dict(param_dict=param_dict)
    else:
        raise NotImplementedError("Only discounted-integration metrics accumulation implemented.")

def constraint_from_json(json_path: pathlib.Path):
    param_dict = parse_json_file(path=json_path)
    constraint_type = ConstraintType(param_dict["constraint_type"])
    if constraint_type == ConstraintType.SEPARATE_CONSTRAINTS:
        return separate_constraints_from_dict(param_dict=param_dict)
    else:
        raise NotImplementedError("Only separate-linear constraints implemented.")

def multi_metrics_reduction_from_json(json_path: pathlib.Path):
    param_dict = parse_json_file(path=json_path)
    multi_metrics_reduction_type = MultiMetricsReductionType(param_dict["multi_metrics_reduction_type"])
    if multi_metrics_reduction_type == MultiMetricsReductionType.SCALAR_WEIGHTING:
        return scalar_weighting_from_dict(param_dict=param_dict)
    else:
        raise NotImplementedError("Only scalar weighting implemented.")

def control_evaluation_system_from_json(json_path: pathlib.Path):
    
    param_dict = parse_json_file(path=json_path)
    eval_system_name = param_dict["name"]
    config_folder = json_path.parent / pathlib.Path(param_dict["relative_directory"])
    
    # Plant model
    plant_model = plant_model_from_json(
        json_path=(config_folder / param_dict["plant_model_file"]))
    
    # Aggregation
    aggregation = aggregation_from_json(
        json_path=(config_folder / param_dict["aggregation_file"]))

    # Control constraint
    control_constraint_file = param_dict.get("constraint_control_file", None)
    if control_constraint_file is not None:
        control_constraint = constraint_from_json(
        json_path=(config_folder / control_constraint_file))
    else:
        control_constraint = None
    
    # Aggregated constraint
    aggregated_constraint_file = param_dict.get("constraint_aggregated_file", None)
    if aggregated_constraint_file is not None:
        aggregated_constraint = constraint_from_json(
        json_path=(config_folder / aggregated_constraint_file))
    else:
        aggregated_constraint = None

    # Accumulated constraint
    accumulated_constraint_file = param_dict.get("constraint_accumulated_file", None)
    if accumulated_constraint_file is not None:
        accumulated_constraint = constraint_from_json(
        json_path=(config_folder / accumulated_constraint_file))
    else:
        accumulated_constraint = None

    # Metrics accumulation
    metrics_accumulation = metrics_accumulation_from_json(
        json_path=(config_folder / param_dict["metrics_accumulation_file"]))

    # Multi-metrics reduction
    multi_metrics_reduction = multi_metrics_reduction_from_json(
        json_path=(config_folder / param_dict["metrics_reduction_file"]))
    
    return ControlEvaluationSystem(
        name=eval_system_name,
        plant_model=plant_model,
        aggregation=aggregation,
        constraint_control=control_constraint,
        constraint_aggregated=aggregated_constraint,
        constraint_accumulated=accumulated_constraint,
        metrics_accumulation=metrics_accumulation,
        multi_metrics_reduction=multi_metrics_reduction)

def control_optimization_from_json(json_path: pathlib.Path):
    param_dict = parse_json_file(path=json_path)
    optimization_method = OptimizationMethod(param_dict["optimization_method"])
    if optimization_method == OptimizationMethod.GRID_SEARCH:
        return grid_search_from_dict(param_dict=param_dict)
    elif optimization_method == OptimizationMethod.SIMULTANEOUS_OPTIMIZATION:
        return simultaneous_optimization_from_dict(param_dict=param_dict)
    elif optimization_method == OptimizationMethod.LAGRANGIAN_RELAXATION:
        return lagrangian_relaxation_from_dict(param_dict=param_dict)
    else:
        raise NotImplementedError("Only grid-search, simultaneous opt. and lagrangian relaxation implemented.")

