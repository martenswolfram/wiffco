from typing import Dict, Any
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from twain_wifco.statistics import (
    Statistics,
    DiscreteStatisticsParams,
    DiscreteStatistics,
    SystematicSample)
from twain_wifco.control_input import ControlPolicy
from twain_wifco.wind_farm_model import WindFarmModel
from twain_wifco.output_aggregation import OutputAggregation
from twain_wifco.interface import (
    OutputVariable,
    AccumulatedMetric,
    InterfaceVariables,
    ComponentParams,
    ComponentType)

def ambient_to_output_statistics(ambient_condition_statistics: Statistics,
                                 control_policy: ControlPolicy,
                                 wind_farm_model: WindFarmModel,
                                 output_aggregation: OutputAggregation):
    
    if not isinstance(ambient_condition_statistics, DiscreteStatistics):
        raise NotImplementedError("Statistics transformation only implemented for DiscreteStatistics.")
    
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

    new_name = ambient_condition_statistics.interface.name + "_transformed"
    discrete_statistics_params = DiscreteStatisticsParams(
        component_name=new_name,
        support_variables=aggregated_output_variables,
        prevalence = ambient_condition_sample.normalized_weights,
        support_points=support_values)
    
    return DiscreteStatistics(params=discrete_statistics_params)

class AccumulationType(Enum):
    DISCOUNTED_INTEGRATION = "discounted_integration"

class OutputAccumulation(ABC):
    def __init__(self,
                 params: ComponentParams):
        self.interface = params.interface()

    def expected_value(self,
                       output_statistics: Statistics,
                       duration: int) -> Dict[AccumulatedMetric, float]:

        self.interface.validate_inputs(output_variables=output_statistics.interface.outputs.output_variables)

        return self._expected_value(output_statistics=output_statistics,
                                    duration=duration)
    
    @abstractmethod
    def _expected_value(self,
                        output_statistics: Statistics,
                        duration: int):
        pass


class DiscountedIntegratorParams(ComponentParams):
    def __init__(self,
                 component_name: str,
                 in_out_mappings: Dict[OutputVariable, AccumulatedMetric],
                 discount_rates: Dict[OutputVariable, float]):
        super().__init__(component_type=ComponentType.OUTPUT_ACCUMULATOR,
                         component_name=component_name)
        self.in_out_mappings = in_out_mappings
        self.discount_rates = discount_rates

    def _interface(self):
        inputs = InterfaceVariables(output_variables=self.in_out_mappings.keys())
        outputs = InterfaceVariables(accumulated_metrics=self.in_out_mappings.values())
        return inputs, outputs

def discounted_integrator_params_from_dict(name: str,
                                           param_dict: Dict[str, Dict | Any]):
    in_out_mappings = {OutputVariable(instant_var): AccumulatedMetric(acc_metric) for \
                       instant_var, acc_metric in param_dict["in_out_mappings"].items()}
    discount_rates = {OutputVariable(out_var): val for out_var, val in param_dict["discount_rates"].items()}
    if set(in_out_mappings) != set(discount_rates):
        raise ValueError("DiscountedIntegratorParams: OutputVariable keys mismatch.")
    return DiscountedIntegratorParams(component_name=name,
                                      in_out_mappings=in_out_mappings,
                                      discount_rates=discount_rates) 

        
class DiscountedIntegrator(OutputAccumulation):
    def __init__(self,
                 params: DiscountedIntegratorParams):
        super().__init__(params=params)
        self.params = params

    def _expected_value(self,
                        output_statistics: Statistics,
                        duration: int):
        accumulated_metrics = {}
        output_expectation = output_statistics.expected_value()
        for out_var, acc_metric in self.params.in_out_mappings.items():
            discount_rate = self.params.discount_rates[out_var]
            if discount_rate == 0:
                accumulated_metrics[acc_metric] = duration * output_expectation[out_var]
            else:
                discount_factor = 1 / (1 + discount_rate)
                duration_discount_factor = (1 - discount_factor**duration) / (1 - discount_factor)
                accumulated_metrics[acc_metric] = output_expectation[out_var] * duration_discount_factor
        
        return accumulated_metrics

        
 