from typing import Dict, Any
import numpy as np
from abc import abstractmethod
from enum import Enum
from twain_wifco.interface import (
    Component,
    ComponentParams,
    AccumulatedMetric,
    DataPoint,
    Interface)

class MultiMetricsReduction(Component):
    def __init__(self,
                 multi_metrics_name: str,
                 maximize: bool,
                 multi_metrics_params: ComponentParams):
        super().__init__(component_name=multi_metrics_name,
                         component_params=multi_metrics_params)
        self.maximize = maximize

    def evaluate(self,
                 acc_metrics: DataPoint[AccumulatedMetric]):
        
        self.validate_inputs(input_data={AccumulatedMetric: acc_metrics})

        return self._evaluate(acc_metrics=acc_metrics)
    
    def cost_function(self, eval_acc_metrics_from_x):
        
        def eval_cost(x):
            acc_metrics_eval = eval_acc_metrics_from_x(x)            
            result = self.evaluate(acc_metrics=acc_metrics_eval)
            if self.maximize:
                return - result
            else:
                return result
        return eval_cost
            
    @abstractmethod
    def _evaluate(self,
                  acc_metrics: DataPoint[AccumulatedMetric]):
        pass

class MultiMetricsReductionType(Enum):
    SCALAR_WEIGHTING = "scalar_weighting"

class ScalarWeightingParams(ComponentParams):
    def __init__(self,
                 metric_weights: DataPoint[AccumulatedMetric]):
        self.metric_weights = metric_weights
        
    def input_interface(self) -> Interface:
        accumulated_matric_shapes = {acc_metric: None for \
                                     acc_metric in self.metric_weights.keys()}
        return Interface(all_shapes={
            AccumulatedMetric: accumulated_matric_shapes})
    
    def output_interface(self) -> Interface:
        return Interface(all_shapes={})

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
                  acc_metrics: Dict[AccumulatedMetric, np.ndarray]):
        return sum(weight * np.sum(acc_metrics[metric]) for metric, weight in self.metric_weights.items())
    