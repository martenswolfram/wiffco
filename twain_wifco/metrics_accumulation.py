from typing import Dict, Any
import numpy as np
from abc import abstractmethod
from enum import Enum
from twain_wifco.statistics import (
    Statistics)
from twain_wifco.interface import (
    Component,
    ComponentParams,
    Aggregated,
    AccumulatedMetric,
    DataPoint,
    Interface)

class MetricsAccumulation(Component):
    def __init__(self,
                 accumulation_name: str,
                 accumulation_params: ComponentParams):
        super().__init__(component_name=accumulation_name,
                         component_params=accumulation_params)
    
    def acc_metrics(self,
                    aggregate: DataPoint[Aggregated],
                    duration: int):
        
        self.validate_inputs(aggregated=aggregate)

        return self._acc_metrics(aggregate=aggregate,
                                 duration=duration)

    @abstractmethod
    def _acc_metrics(self,
                     aggregate: DataPoint[Aggregated],
                     duration: int):
        pass
    
    def expected_acc_metrics(self,
                       aggregate_statistics: Statistics,
                       duration: int):
        aggregate_expectation = aggregate_statistics.expected_value()
        
        return self._acc_metrics(aggregate=aggregate_expectation,
                                 duration=duration)
        

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

    def input_interface(self) -> Interface:        
        return Interface(
            aggregated_shapes={aggr_mapping.aggregate: None for \
                               aggr_mapping in self.integration_mappings.values()})

    def output_interface(self) -> Interface:
        accumulated_metric_shapes = {acc_metric: None for \
                                     acc_metric in self.integration_mappings.keys()}
        return Interface(
            accumulated_metric_shapes=accumulated_metric_shapes)

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

    def _acc_metrics(self,
                    aggregate: DataPoint[Aggregated],
                    duration: int):
        accumulated_metrics = {}
        for acc_metric, integration_mapping in self.integration_mappings.items():
            discount_rate = integration_mapping.discount_rate
            if discount_rate == 0:
                accumulated_metrics[acc_metric] = duration * aggregate[integration_mapping.aggregate]
            else:
                discount_factor = 1 / (1 + discount_rate)
                duration_discount_factor = (1 - discount_factor**duration) / (1 - discount_factor)
                accumulated_metrics[acc_metric] = aggregate[integration_mapping.aggregate] * duration_discount_factor

        return DataPoint(accumulated_metrics)
     