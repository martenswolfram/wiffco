from typing import Dict, Any, Set
from abc import abstractmethod
import numpy as np
from enum import Enum
from twain_wifco.interface import (
    Component,
    ComponentParams,
    AccumulatedMetric)

class MultiMetricsReduction(Component):
    def __init__(self,
                 multi_metrics_name: str,
                 maximize: bool,
                 multi_metrics_params: ComponentParams):
        super().__init__(component_name=multi_metrics_name,
                         component_params=multi_metrics_params)
        self.maximize = maximize

    def evaluate(self,
                 acc_metrics: Dict[AccumulatedMetric, float]):
        
        self._validate_inputs(inputs=acc_metrics.keys())

        return self._evaluate(acc_metrics=acc_metrics)
            
    @abstractmethod
    def _evaluate(self,
                  acc_metrics: Dict[AccumulatedMetric, float]):
        pass

class MultiMetricsReductionType(Enum):
    SCALAR_WEIGHTING = "scalar_weighting"

class ScalarWeightingParams(ComponentParams):
    def __init__(self,
                 metric_weights: Dict[AccumulatedMetric, float]):
        self.metric_weights = metric_weights
        
    def input_variables(self):
        return self.metric_weights.keys()
    
    def output_variables(self):
        return set()

def scalar_weighting_params_from_dict(param_dict: Dict[str, Dict | Any]):
    metric_weights = {}
    for acc_metric, weight in param_dict["metric_weights"].items():
        metric_weights[AccumulatedMetric(acc_metric)] = float(weight)
    return ScalarWeightingParams(metric_weights=metric_weights)

class ScalarWeighting(MultiMetricsReduction):
    def __init__(self,
                 multi_metrics_reduction_name: str,
                 maximize: bool,
                 multi_metrics_reduction_params: ScalarWeightingParams):
        super().__init__(multi_metrics_name=multi_metrics_reduction_name,
                         maximize=maximize,
                         multi_metrics_params=multi_metrics_reduction_params)
        self.metric_weights = multi_metrics_reduction_params.metric_weights
        
    def _evaluate(self,
                  acc_metrics: Dict[AccumulatedMetric, float]):
        return sum(weight * acc_metrics[metric] for metric, weight in self.metric_weights.items())
