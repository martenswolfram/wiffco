from typing import Dict, Any
import numpy as np
from abc import abstractmethod
from enum import Enum
from twain_wifco.interface import (
    Component,
    AccumulatedMetric,
    DataPoint,
    Interface
)


# ----------------------------
# MultiMetricsReduction Base Class
# ----------------------------
class MultiMetricsReduction(Component):
    """Abstract base class to reduce multiple accumulated metrics to a scalar value.

    Args:
        maximize (bool): Whether the evaluation is to be maximized.
    """
    def __init__(self, maximize: bool):
        self._maximize = maximize

    @property
    def maximize(self):
        return self._maximize

    @abstractmethod
    @Component.with_validation
    def evaluate(self, acc_metrics: DataPoint[AccumulatedMetric]) -> float:
        """Evaluate the multi-metrics reduction for given accumulated metrics.

        Args:
            acc_metrics (DataPoint[AccumulatedMetric]): Accumulated metrics to reduce.

        Returns:
            float: Scalar reduction of the metrics.
        """
        ...

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

# ----------------------------
# MultiMetricsReduction Types
# ----------------------------
class MultiMetricsReductionType(Enum):
    SCALAR_WEIGHTING = "scalar_weighting"


# ----------------------------
# ScalarWeighting Implementation
# ----------------------------
class ScalarWeighting(MultiMetricsReduction):
    """Parameters for ScalarWeighting reduction.

    Attributes:
        metric_weights (DataPoint[AccumulatedMetric]): Weights for each accumulated metric, component-wise.
    """
    def __init__(self,
                 name: str,
                 maximize: bool,
                 metric_weights: DataPoint[AccumulatedMetric]):
        super().__init__(maximize=maximize)
        self._component_name = name
        self._metric_weights = metric_weights

        accumulated_metric_shapes = {
            acc_metric: None for acc_metric in self._metric_weights.keys()}
        self._input_interface = Interface(
            all_shapes={AccumulatedMetric: accumulated_metric_shapes})

        self._output_interface = Interface()

    @Component.with_validation
    def evaluate(self, acc_metrics: DataPoint[AccumulatedMetric]) -> float:
        """Compute scalar weighted sum of accumulated metrics.

        Args:
            acc_metrics (DataPoint[AccumulatedMetric]): Accumulated metrics to reduce.

        Returns:
            float: Weighted sum of metrics.
        """
        result = sum(
            weight * np.sum(acc_metrics[metric])
            for metric, weight in self._metric_weights.items()
        )
        return result


def scalar_weighting_from_dict(param_dict: Dict[str, Any]) -> ScalarWeighting:
    """Create ScalarWeighting from a dictionary.

    Args:
        param_dict (Dict[str, Any]): Dictionary containing 'metric_weights'.

    Returns:
        ScalarWeighting: Constructed ScalarWeighting metrics reduction object.
    """
    name = param_dict["name"]
    maximize = param_dict["maximize"]
    metric_weights = {AccumulatedMetric(k): float(v) for \
                      k, v in param_dict["metric_weights"].items()}
    return ScalarWeighting(
        name=name,
        maximize=maximize,
        metric_weights=metric_weights)

    