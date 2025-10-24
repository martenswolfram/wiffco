from typing import Dict, Any
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from twain_wifco.interface import (
    Component,
    ComponentParams,
    AccumulatedMetric,
    DataPoint,
    Interface
)


# ----------------------------
# MultiMetricsReduction Base Class
# ----------------------------
class MultiMetricsReduction(Component, ABC):
    """Abstract base class to reduce multiple accumulated metrics to a scalar value.

    Args:
        multi_metrics_name (str): Name of the multi-metrics reduction component.
        maximize (bool): Whether the evaluation is to be maximized.
        multi_metrics_params (ComponentParams): Parameters for the reduction.
    """
    def __init__(self,
                 multi_metrics_name: str,
                 maximize: bool,
                 multi_metrics_params: ComponentParams):
        super().__init__(component_name=multi_metrics_name,
                         component_params=multi_metrics_params)
        self.maximize = maximize

    def evaluate(self, acc_metrics: DataPoint[AccumulatedMetric]) -> float:
        """Evaluate the multi-metrics reduction for given accumulated metrics.

        Args:
            acc_metrics (DataPoint[AccumulatedMetric]): Accumulated metrics to reduce.

        Returns:
            float: Scalar reduction of the metrics.
        """
        self.validate_inputs(input_data={AccumulatedMetric: acc_metrics})
        return self._evaluate(acc_metrics=acc_metrics)

    def cost_function(self, eval_acc_metrics_from_x):
        """Return a cost function suitable for optimization routines.

        Args:
            eval_acc_metrics_from_x (Callable): Function that evaluates accumulated metrics from decision variables.

        Returns:
            Callable: Cost function that returns a scalar.
        """
        def eval_cost(x):
            acc_metrics_eval = eval_acc_metrics_from_x(x)
            result = self.evaluate(acc_metrics=acc_metrics_eval)
            return -result if self.maximize else result
        return eval_cost

    @abstractmethod
    def _evaluate(self, acc_metrics: DataPoint[AccumulatedMetric]) -> float:
        """Implementation-specific metric reduction logic.

        Args:
            acc_metrics (DataPoint[AccumulatedMetric]): Accumulated metrics to reduce.

        Returns:
            float: Scalar evaluation.
        """
        pass


# ----------------------------
# MultiMetricsReduction Types
# ----------------------------
class MultiMetricsReductionType(Enum):
    SCALAR_WEIGHTING = "scalar_weighting"


# ----------------------------
# ScalarWeighting Implementation
# ----------------------------
class ScalarWeightingParams(ComponentParams):
    """Parameters for ScalarWeighting reduction.

    Attributes:
        metric_weights (DataPoint[AccumulatedMetric]): Weights for each accumulated metric, component-wise.
    """
    def __init__(self,
                 metric_weights: DataPoint[AccumulatedMetric]):
        self.metric_weights = metric_weights

    def input_interface(self) -> Interface:
        """Input interface describing accumulated metric shapes."""
        accumulated_metric_shapes = {acc_metric: None for acc_metric in self.metric_weights.keys()}
        return Interface(all_shapes={AccumulatedMetric: accumulated_metric_shapes})

    def output_interface(self) -> Interface:
        """Output interface (scalar output, empty shapes)."""
        return Interface(all_shapes={})


def scalar_weighting_params_from_dict(param_dict: Dict[str, Any]) -> ScalarWeightingParams:
    """Create ScalarWeightingParams from a dictionary.

    Args:
        param_dict (Dict[str, Any]): Dictionary containing 'metric_weights'.

    Returns:
        ScalarWeightingParams: Constructed parameters object.
    """
    metric_weights = {AccumulatedMetric(k): float(v) for k, v in param_dict["metric_weights"].items()}
    return ScalarWeightingParams(metric_weights=metric_weights)


class ScalarWeighting(MultiMetricsReduction):
    """Reduce accumulated metrics using weighted scalar sum."""

    def __init__(self,
                 multi_metrics_reduction_name: str,
                 maximize: bool,
                 multi_metrics_reduction_params: ScalarWeightingParams):
        super().__init__(multi_metrics_name=multi_metrics_reduction_name,
                         maximize=maximize,
                         multi_metrics_params=multi_metrics_reduction_params)
        self.metric_weights = multi_metrics_reduction_params.metric_weights

    def _evaluate(self, acc_metrics: DataPoint[AccumulatedMetric]) -> float:
        """Compute scalar weighted sum of accumulated metrics.

        Args:
            acc_metrics (DataPoint[AccumulatedMetric]): Accumulated metrics to reduce.

        Returns:
            float: Weighted sum of metrics.
        """
        result = sum(
            weight * np.sum(acc_metrics[metric])
            for metric, weight in self.metric_weights.items()
        )
        return result
