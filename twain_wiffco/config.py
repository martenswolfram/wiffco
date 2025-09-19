import json
from typing import Dict, Any
from twain_wiffco.wind_farm_model import ModelType, IndependentCubicInterpolatorParams, IndependentCubicInterpolator
from twain_wiffco.ambient_conditions import AmbientVariable, AmbientStatisticsType, DiscreteAmbientStatisticsParams, DiscreteAmbientStatistics
from twain_wiffco.control_input import ControlPolicyType, DiscreteControlPolicyParams, DiscreteControlPolicy
from twain_wiffco.model_output import OutputVariable
from twain_wiffco.output_aggregation import AggregationType, SimpleProductParams, SimpleProduct

def parse_json_file(path):
    with open(path, "r") as json_file:
        data_dict = json.load(json_file)
        return data_dict

def wind_farm_model_from_dict(param_dict: Dict[str, Any]):
    
    name = param_dict["name"]
    model_type = ModelType(param_dict["model_type"])
    if model_type == ModelType.INDEPENDENT_CUBIC_INTERPOLATOR:
        params = IndependentCubicInterpolatorParams(param_dict=param_dict["model_params"])
        return IndependentCubicInterpolator(name=name,
                                            params=params)
    else:
        raise NotImplementedError("Only independent_cubic_interpolator model implemented.")

def ambient_statistics_from_dict(param_dict: Dict[str, Any]):
    name = param_dict["name"]
    statistics_type = AmbientStatisticsType(param_dict["statistics_type"])
    if statistics_type == AmbientStatisticsType.DISCRETE_ABIENT_STATISTICS:
        params = DiscreteAmbientStatisticsParams(param_dict=param_dict["statistics_params"])
        return DiscreteAmbientStatistics(name=name,
                                         params=params)
    else:
        raise NotImplementedError("Only discrete_ambient_statistics implemented.")

def control_policy_from_dict(param_dict: Dict[str, Any]):
    name = param_dict["name"]
    policy_type = ControlPolicyType(param_dict["policy_type"])
    if policy_type == ControlPolicyType.DISCRETE_CONTROL_POLICY:
        params = DiscreteControlPolicyParams(param_dict=param_dict["policy_params"])
        return DiscreteControlPolicy(name=name,
                                     params=params)
    else:
        raise NotImplementedError("Only discrete_control_policy implemented.")

def output_aggregation_from_dict(param_dict: Dict[str, Any]):
    name = param_dict["name"]
    aggregation_type = AggregationType(param_dict["aggregation_type"])
    if aggregation_type == AggregationType.SIMPLE_PRODUCT:
        params = SimpleProductParams(param_dict=param_dict["aggregation_params"])
        return SimpleProduct(name=name,
                             params=params)
    else:
        raise NotImplementedError("Only simple_product aggregation implemented.")
