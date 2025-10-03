import json
import pathlib
from typing import Dict, Any
from twain_wifco.plant_model import (
    ModelType,
    independent_cubic_interp_params_from_dict,
    IndependentCubicInterpolation)
from twain_wifco.statistics import (
    StatisticsType,
    discrete_statistics_params_from_dict,
    DiscreteStatistics)
from twain_wifco.control_input import (
    ControlPolicyType,
    discrete_policy_params_from_dict,
    DiscreteControlPolicy)
from twain_wifco.aggregation import (
    AggregationType,
    simple_product_params_from_dict,
    SimpleProduct)
from twain_wifco.metrics_accumulation import (
    MetricsAccumulationType,
    discounted_integrator_params_from_dict,
    DiscountedIntegration)
from twain_wifco.accumulated_constraint import (
    AccumulatedConstraintType,
    separate_linear_constraints_params_from_dict,
    SeparateLinearConstraints)
from twain_wifco.multi_metrics_handling import (
    MultiMetricsHandlingType,
    scalar_weighting_params_from_dict,
    ScalarWeighting)
from twain_wifco.optimization import (
    ControlEvaluationSystem,
    OptimizationMethod,
    grid_search_params_from_dict,
    GridSearch)


def parse_json_file(path):
    with open(path, "r") as json_file:
        data_dict = json.load(json_file)
        return data_dict

def ambient_statistics_from_json(json_path: pathlib.Path):
    param_dict = parse_json_file(path=json_path)
    statistics_name = param_dict["name"]
    statistics_type = StatisticsType(param_dict["statistics_type"])
    if statistics_type == StatisticsType.DISCRETE_STATISTICS:
        params = discrete_statistics_params_from_dict(param_dict=param_dict["statistics_params"])
        return DiscreteStatistics(statistics_name=statistics_name,
                                  statistics_params=params)
    else:
        raise NotImplementedError("Only discrete_ambient_statistics implemented.")

def plant_model_from_json(json_path: pathlib.Path):
    param_dict = parse_json_file(path=json_path)
    
    plant_name = param_dict["name"]
    model_type = ModelType(param_dict["model_type"])
    if model_type == ModelType.INDEPENDENT_CUBIC_INTERPOLATION:
        params = independent_cubic_interp_params_from_dict(param_dict=param_dict["model_params"])
        return IndependentCubicInterpolation(plant_name=plant_name,
                                            plant_params=params)
    else:
        raise NotImplementedError("Only independent_cubic_interpolation model implemented.")

def control_policy_from_json(json_path: pathlib.Path):
    param_dict = parse_json_file(path=json_path)
    policy_name = param_dict["name"]
    policy_type = ControlPolicyType(param_dict["policy_type"])
    if policy_type == ControlPolicyType.DISCRETE_POLICY:
        params = discrete_policy_params_from_dict(param_dict=param_dict["policy_params"])
        return DiscreteControlPolicy(policy_name=policy_name,
                                     policy_params=params)
    else:
        raise NotImplementedError("Only discrete_policy implemented.")

def aggregation_from_json(json_path: pathlib.Path):
    param_dict = parse_json_file(path=json_path)
    aggregation_name = param_dict["name"]
    aggregation_type = AggregationType(param_dict["aggregation_type"])
    if aggregation_type == AggregationType.SIMPLE_PRODUCTS:
        params = simple_product_params_from_dict(param_dict=param_dict["aggregation_params"])
        return SimpleProduct(aggregation_name=aggregation_name,
                             aggregation_params=params)
    else:
        raise NotImplementedError("Only simple_product aggregation implemented.")

def metrics_accumulation_from_json(json_path: pathlib.Path):
    param_dict = parse_json_file(path=json_path)
    accumulation_name = param_dict["name"]
    accumulation_type = MetricsAccumulationType(param_dict["accumulation_type"])
    if accumulation_type == MetricsAccumulationType.DISCOUNTED_INTEGRATION:
        params = discounted_integrator_params_from_dict(param_dict=param_dict["accumulation_params"])
        return DiscountedIntegration(accumulation_name=accumulation_name,
                                     accumulation_params=params)
    else:
        raise NotImplementedError("Only discounted-integration metrics accumulation implemented.")

def accumulated_constraint_from_json(json_path: pathlib.Path):
    param_dict = parse_json_file(path=json_path)
    acc_constraint_name = param_dict["name"]
    acc_constraint_type = AccumulatedConstraintType(param_dict["constraint_type"])
    if acc_constraint_type == AccumulatedConstraintType.SEPARATE_LINEAR_CONSTRAINTS:
        params = separate_linear_constraints_params_from_dict(param_dict=param_dict["constraint_params"])
        return SeparateLinearConstraints(acc_constraint_name=acc_constraint_name,
                                     acc_constraint_params=params)
    else:
        raise NotImplementedError("Only separate-linear constraints implemented.")

def multi_metrics_handling_from_json(json_path: pathlib.Path):
    param_dict = parse_json_file(path=json_path)
    multi_metrics_handling_name = param_dict["name"]
    multi_metrics_handling_type = MultiMetricsHandlingType(param_dict["multi_metrics_handling_type"])
    if multi_metrics_handling_type == MultiMetricsHandlingType.SCALAR_WEIGHTING:
        params = scalar_weighting_params_from_dict(param_dict=param_dict["multi_metrics_handling_params"])
        return ScalarWeighting(multi_metrics_handling_name=multi_metrics_handling_name,
                               multi_metrics_handling_params=params)
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
       
    # Metrics accumulation
    metrics_accumulation = metrics_accumulation_from_json(
        json_path=(config_folder / param_dict["metrics_accumulation_file"]))

    # Accumulated constraint
    accumulated_constraint = accumulated_constraint_from_json(
        json_path=(config_folder / param_dict["accumulated_constraint_file"]))

    # Multi-metrics handling
    multi_metrics_handling = multi_metrics_handling_from_json(
        json_path=(config_folder / param_dict["multi_metrics_handling_file"]))

    # Initial control policy
    initial_control_policy = control_policy_from_json(
        json_path=(config_folder / param_dict["initial_control_policy_file"]))
    
    return ControlEvaluationSystem(
        name=eval_system_name,
        plant_model=plant_model,
        aggregation=aggregation,
        metrics_accumulation=metrics_accumulation,
        accumulated_constraint=accumulated_constraint,
        multi_metrics_handling=multi_metrics_handling,
        initial_control_policy=initial_control_policy)

def control_optimization_from_json(json_path: pathlib.Path):
    param_dict = parse_json_file(path=json_path)
    optimization_name = param_dict["name"]
    optimization_method = OptimizationMethod(param_dict["optimization_method"])
    if optimization_method == OptimizationMethod.GRID_SEARCH:
        params = grid_search_params_from_dict(param_dict=param_dict["optimization_params"])
        return GridSearch(optimization_name=optimization_name,
                          optimization_params=params)
    else:
        raise NotImplementedError("Only grid-search optimization implemented.")

