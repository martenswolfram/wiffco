from typing import Dict, Any
from abc import abstractmethod
from enum import Enum
import numpy as np
from twain_wifco.statistics import (
    Statistics,
    DiscreteStatistics,
    DiscreteStatisticsParams,
    SystematicSample)
from twain_wifco.control_input import ControlPolicy
from twain_wifco.plant_model import PlantModel
from twain_wifco.aggregation import Aggregation
from twain_wifco.interface import (
    Component,
    ComponentParams,
    Ambient,
    Control,
    Aggregated,
    AccumulatedMetric)

def ambient_to_discrete_aggregate_statistics(
        ambient_condition_statistics: Statistics,
        control_policy: ControlPolicy,
        plant_model: PlantModel,
        aggregation: Aggregation,
        N = None):
        
    ambient_condition_sample: SystematicSample = ambient_condition_statistics.systematic_sample(N=N)

    ambient_variables = ambient_condition_sample.support_variables
    aggregated_variables = aggregation.output_variables
    aggregated_support_values = np.empty(shape=(len(aggregated_variables), ambient_condition_sample.N))
    for i, ambient_condition_values in enumerate(ambient_condition_sample.support_values.T):
        ambient_condition = {
            ambient_var: val for ambient_var, val in zip(ambient_variables, ambient_condition_values)}
        control_setpoints = control_policy.get_control_setpoints(ambient_condition=ambient_condition)
        model_output = plant_model.evaluate(meteorological_condition=ambient_condition,
                                            control_input=control_setpoints)
        aggregated = aggregation.compute_aggregate(model_output=model_output,
                                                   ambient_condition=ambient_condition,
                                                   control_setpoints=control_setpoints)
        aggregated_support_values[:, i] = [aggregated[aggr_var] for aggr_var in aggregated_variables]

    discrete_statistics_params = DiscreteStatisticsParams(
        support_variables=aggregated_variables,
        prevalence = ambient_condition_sample.normalized_weights,
        support_points=aggregated_support_values)
    
    new_name = ambient_condition_statistics.component_name + "_transformed_to_aggregated"
    return DiscreteStatistics(statistics_name=new_name,
                              statistics_params=discrete_statistics_params)

class MetricsAccumulation(Component):
    def __init__(self,
                 accumulation_name: str,
                 accumulation_params: ComponentParams):
        super().__init__(component_name=accumulation_name,
                         component_params=accumulation_params)
    

    def expected_value(self,
                       aggregate_statistics: Statistics,
                       duration: int) -> Dict[AccumulatedMetric, float]:

        self._validate_inputs(inputs=aggregate_statistics.output_variables)

        return self._expected_value(aggregate_statistics=aggregate_statistics,
                                    duration=duration)
    
    @abstractmethod
    def _expected_value(self,
                        aggregate_statistics: Statistics,
                        duration: int):
        pass

class MetricsAccumulationType(Enum):
    DISCOUNTED_INTEGRATION = "discounted_integration"

class IntegrationMapping:
    def __init__(self,
                 aggregate: Aggregated,
                 discount_rate: float):
        self.aggregate = aggregate
        self.discount_rate = discount_rate

class DiscountedIntegrationParams(ComponentParams):
    def __init__(self,
                 integration_mappings: Dict[AccumulatedMetric, IntegrationMapping]):
        self.integration_mappings = integration_mappings

    def input_variables(self):
        required_aggregates = set([mapping.aggregate for mapping in self.integration_mappings.values()])
        return required_aggregates
    
    def output_variables(self):
        return set(self.integration_mappings.keys())

def discounted_integrator_params_from_dict(param_dict: Dict[str, Dict | Any]):
    integration_mappings = {}
    for acc_metric, integration_mapping in param_dict["integration_mappings"].items():
         integration_mappings[AccumulatedMetric(acc_metric)] = \
            IntegrationMapping(aggregate=Aggregated(integration_mapping["aggregate"]),
                               discount_rate=float(integration_mapping["discount_rate"]))
    
    return DiscountedIntegrationParams(integration_mappings=integration_mappings)
        
class DiscountedIntegration(MetricsAccumulation):
    def __init__(self,
                 accumulation_name: str,
                 accumulation_params: DiscountedIntegrationParams):
        super().__init__(accumulation_name=accumulation_name,
                         accumulation_params=accumulation_params)
        self.integration_mappings = accumulation_params.integration_mappings

    def _expected_value(self,
                        aggregate_statistics: Statistics,
                        duration: int):
        accumulated_metrics = {}
        aggregate_expectation = aggregate_statistics.expected_value()
        for acc_metric, integration_mapping in self.integration_mappings.items():
            discount_rate = integration_mapping.discount_rate
            if discount_rate == 0:
                accumulated_metrics[acc_metric] = duration * aggregate_expectation[integration_mapping.aggregate]
            else:
                discount_factor = 1 / (1 + discount_rate)
                duration_discount_factor = (1 - discount_factor**duration) / (1 - discount_factor)
                accumulated_metrics[acc_metric] = aggregate_expectation[integration_mapping.aggregate] * duration_discount_factor
        
        return accumulated_metrics

        
 