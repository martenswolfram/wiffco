from typing import Dict, Any
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from twain_wifco.statistics import Statistics, SystematicSample
from twain_wifco.control_input import ControlPolicy
from twain_wifco.wind_farm_model import WindFarmModel
from twain_wifco.output_aggregation import OutputAggregation
from twain_wifco.interface import (
    OutputVariable,
    ComponentParams)

def ambient_to_output_statistics(ambient_condition_statistics: Statistics,
                                 control_policy: ControlPolicy,
                                 wind_farm_model: WindFarmModel,
                                 output_aggregation: OutputAggregation):
    ambient_condition_sample: SystematicSample = ambient_condition_statistics.systematic_sample()

    ambient_variables = ambient_condition_sample.support_variables
    aggregated_output_variables = output_aggregation.interface.outputs.output_variables
    support_values = np.empty(shape=(len(aggregated_output_variables), ambient_condition_sample.N))
    for i, ambient_condition_values in enumerate(ambient_condition_sample.support_values.T):
        ambient_condition = {
            ambient_var: val for ambient_var, val in zip(ambient_variables, ambient_condition_values)}
        control_input = control_policy.get_control_setpoints(ambient_condition=ambient_condition)
        model_output = wind_farm_model.evaluate(meteorological_condition=ambient_condition,
                                                control_input=control_input)
        aggregated_output = output_aggregation.compute_aggregate(output_variables=model_output,
                                                                 ambient_condition=ambient_condition)
        support_values[:, i] = [aggregated_output[out_var] for out_var in aggregated_output_variables]

    return SystematicSample(support_variables=aggregated_output_variables,
                            support_values=support_values,
                            normalized_weights=ambient_condition_sample.normalized_weights,
                            probability_covered=ambient_condition_sample.probability_covered)
