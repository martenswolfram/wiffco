import json
from typing import Dict, Any
from twain_wifco.wind_farm_model import (
    ModelType,
    independent_cubic_interp_params_from_dict,
    IndependentCubicInterpolator)
from twain_wifco.statistics import (
    StatisticsType,
    discrete_statistics_params_from_dict,
    DiscreteStatistics)
from twain_wifco.output_aggregation import (
    AggregationType,
    simple_product_params_from_dict,
    SimpleProduct)
from twain_wifco.control_input import (
    ControlPolicyType,
    discrete_policy_params_from_dict,
    DiscreteControlPolicy)

def parse_json_file(path):
    with open(path, "r") as json_file:
        data_dict = json.load(json_file)
        return data_dict

def ambient_statistics_from_dict(param_dict: Dict[str, Any]):
    name = param_dict["name"]
    statistics_type = StatisticsType(param_dict["statistics_type"])
    if statistics_type == StatisticsType.DISCRETE_STATISTICS:
        params = discrete_statistics_params_from_dict(param_dict=param_dict["statistics_params"])
        return DiscreteStatistics(name=name,
                                         params=params)
    else:
        raise NotImplementedError("Only discrete_ambient_statistics implemented.")

def wind_farm_model_from_dict(param_dict: Dict[str, Any]):
    
    name = param_dict["name"]
    model_type = ModelType(param_dict["model_type"])
    if model_type == ModelType.INDEPENDENT_CUBIC_INTERPOLATOR:
        params = independent_cubic_interp_params_from_dict(name=name,
                                                           param_dict=param_dict["model_params"])
        return IndependentCubicInterpolator(params=params)
    else:
        raise NotImplementedError("Only independent_cubic_interpolator model implemented.")

def output_aggregation_from_dict(param_dict: Dict[str, Any]):
    name = param_dict["name"]
    aggregation_type = AggregationType(param_dict["aggregation_type"])
    if aggregation_type == AggregationType.SIMPLE_PRODUCT:
        params = simple_product_params_from_dict(name=name,
                                                 param_dict=param_dict["aggregation_params"])
        return SimpleProduct(params=params)
    else:
        raise NotImplementedError("Only simple_product aggregation implemented.")

def control_policy_from_dict(param_dict: Dict[str, Any]):
    name = param_dict["name"]
    policy_type = ControlPolicyType(param_dict["policy_type"])
    if policy_type == ControlPolicyType.DISCRETE_CONTROL_POLICY:
        params = discrete_policy_params_from_dict(name=name,
                                                  param_dict=param_dict["policy_params"])
        return DiscreteControlPolicy(params=params)
    else:
        raise NotImplementedError("Only discrete_control_policy implemented.")
