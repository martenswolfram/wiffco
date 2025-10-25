import json
import pathlib
from twain_wifco.plant_model import (
    ModelType,
    factorized_scattered_interp_params_from_dict,
    FactorizedScatteredInterp,
    symbolic_model_params_from_dict,
    SymbolicModel)
from twain_wifco.statistics import (
    StatisticsType,
    discrete_statistics_params_from_dict,
    DiscreteStatistics)
from twain_wifco.control_policy import (
    discrete_control_policy_params_from_dict,
    DiscreteControlPolicy)
from twain_wifco.aggregation import (
    AggregationType,
    simple_product_params_from_dict,
    SimpleProduct)
from twain_wifco.metrics_accumulation import (
    MetricsAccumulationType,
    discounted_integrator_params_from_dict,
    DiscountedIntegration)
from twain_wifco.constraint import (
    ConstraintType,
    separate_constraints_params_from_dict,
    SeparateConstraints)
from twain_wifco.multi_metrics_reduction import (
    MultiMetricsReductionType,
    scalar_weighting_params_from_dict,
    ScalarWeighting)
from twain_wifco.optimization import (
    ControlEvaluationSystem,
    OptimizationMethod,
    grid_search_params_from_dict,
    GridSearch,
    simultaneous_optimization_params_from_dict,
    SimultaneousOptimization,
    lagrangian_relaxation_params_from_dict,
    LagrangianRelaxation)


def parse_json_file(path):
    with open(path, "r") as json_file:
        data_dict = json.load(json_file)
        return data_dict

def statistics_from_json(json_path: pathlib.Path):
    param_dict = parse_json_file(path=json_path)
    statistics_name = param_dict["name"]
    statistics_type = StatisticsType(param_dict["statistics_type"])
    if statistics_type == StatisticsType.DISCRETE_STATISTICS:
        params = discrete_statistics_params_from_dict(param_dict=param_dict["statistics_params"])
        return DiscreteStatistics(statistics_name=statistics_name,
                                  statistics_params=params)
    else:
        raise NotImplementedError("Only discrete_statistics implemented.")

def plant_model_from_json(json_path: pathlib.Path):
    param_dict = parse_json_file(path=json_path)
    
    plant_name = param_dict["name"]
    model_type = ModelType(param_dict["model_type"])
    if model_type == ModelType.FACTORIZED_SCATTERED_INTERPOLATOR:
        params = factorized_scattered_interp_params_from_dict(
            param_dict=param_dict["model_params"])
        return FactorizedScatteredInterp(name=plant_name,
                                         params=params)
    elif model_type == ModelType.SYMBOLIC:
        params = symbolic_model_params_from_dict(
            param_dict=param_dict["model_params"])
        return SymbolicModel(name=plant_name,
                             params=params)
    else:
        raise NotImplementedError("Only factorized_rbf_interpolation model implemented.")

def control_policy_from_json(json_path: pathlib.Path):
    param_dict = parse_json_file(path=json_path)
    policy_name = param_dict["name"]
    params = discrete_control_policy_params_from_dict(param_dict=param_dict["policy_params"])
    return DiscreteControlPolicy(name=policy_name,
                                 params=params)
    
def aggregation_from_json(json_path: pathlib.Path):
    param_dict = parse_json_file(path=json_path)
    name = param_dict["name"]
    aggregation_type = AggregationType(param_dict["aggregation_type"])
    if aggregation_type == AggregationType.SIMPLE_PRODUCT:
        params = simple_product_params_from_dict(param_dict=param_dict["params"])
        return SimpleProduct(name=name,
                             params=params)
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

def constraint_from_json(json_path: pathlib.Path):
    param_dict = parse_json_file(path=json_path)
    constraint_name = param_dict["name"]
    constraint_type = ConstraintType(param_dict["constraint_type"])
    if constraint_type == ConstraintType.SEPARATE_CONSTRAINTS:
        params = separate_constraints_params_from_dict(param_dict=param_dict["constraint_params"])
        return SeparateConstraints(constraint_name=constraint_name,
                                         constraint_params=params)
    else:
        raise NotImplementedError("Only separate-linear constraints implemented.")

def multi_metrics_reduction_from_json(json_path: pathlib.Path):
    param_dict = parse_json_file(path=json_path)
    multi_metrics_reduction_name = param_dict["name"]
    maximize = param_dict["maximize"]
    multi_metrics_reduction_type = MultiMetricsReductionType(param_dict["multi_metrics_reduction_type"])
    if multi_metrics_reduction_type == MultiMetricsReductionType.SCALAR_WEIGHTING:
        params = scalar_weighting_params_from_dict(param_dict=param_dict["multi_metrics_reduction_params"])
        return ScalarWeighting(multi_metrics_reduction_name=multi_metrics_reduction_name,
                               maximize=maximize,
                               multi_metrics_reduction_params=params)
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
    control_constraint = constraint_from_json(
        json_path=(config_folder / param_dict["constraint_control_file"]))
    
    # Aggregated constraint
    aggregated_constraint = constraint_from_json(
        json_path=(config_folder / param_dict["constraint_aggregated_file"]))

    # Accumulated constraint
    accumulated_constraint = constraint_from_json(
        json_path=(config_folder / param_dict["constraint_accumulated_file"]))

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
    optimization_name = param_dict["name"]
    optimization_method = OptimizationMethod(param_dict["optimization_method"])
    if optimization_method == OptimizationMethod.GRID_SEARCH:
        params = grid_search_params_from_dict(param_dict=param_dict["optimization_params"])
        return GridSearch(optimization_name=optimization_name,
                          optimization_params=params)
    elif optimization_method == OptimizationMethod.SIMULTANEOUS_OPTIMIZATION:
        params = simultaneous_optimization_params_from_dict(param_dict=param_dict["optimization_params"])
        return SimultaneousOptimization(optimization_name=optimization_name,
                                        optimization_params=params)
    elif optimization_method == OptimizationMethod.LAGRANGIAN_RELAXATION:
        params = lagrangian_relaxation_params_from_dict(param_dict=param_dict["optimization_params"])
        return LagrangianRelaxation(optimization_name=optimization_name,
                                    optimization_params=params)
    else:
        raise NotImplementedError("Only grid-search, simultaneous opt. and lagrangian relaxation implemented.")

