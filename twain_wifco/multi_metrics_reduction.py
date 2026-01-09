from typing import Dict, Any
import numpy as np
from abc import abstractmethod
from enum import Enum
from twain_wifco.interface import (
    Component,
    AccumulatedMetric,
    DataTable,
    Interface
)


# ----------------------------
# MultiMetricsReduction Base Class
# ----------------------------
class MultiMetricsReduction(Component):
    """Abstract base class to reduce multiple accumulated metrics to a scalar value.

    Args:
        maximize (bool): Whether the evaluation is to be maximized or minimized.
    """
    def __init__(self, maximize: bool):
        self._maximize = maximize

    @property
    def maximize(self):
        return self._maximize

    def evaluate(self, acc_metrics: DataTable[AccumulatedMetric]) -> np.array:
        """Evaluate the multi-metrics reduction for given accumulated metrics.

        Args:
            acc_metrics (DataTable[AccumulatedMetric]): Accumulated metrics to reduce

        Returns:
            np.array: Scalar reduction of the metrics for each point in acc. metrics DataTable
        """
        self.validate_input(input_tables=[acc_metrics])
        return self._evaluate(acc_metrics=acc_metrics)

    @abstractmethod
    def _evaluate(self, acc_metrics: DataTable[AccumulatedMetric]) -> np.array:
        """Evaluate the multi-metrics reduction for given accumulated metrics.

        Args:
            acc_metrics (DataTable[AccumulatedMetric]): Accumulated metrics to reduce.

        Returns:
            np.array: Scalar reduction of the metrics for each point in acc. metrics DataTable
        """
        ...


    def cost_function(self, eval_metrics_from_x):
        """Return a cost function suitable for optimization routines.

        Args:
            eval_acc_metrics_from_x (Callable): Function that evaluates accumulated metrics from decision variables.

        Returns:
            Callable: Cost function that returns a scalar.
        """
        def eval_cost(x):
            acc_metrics_eval = eval_metrics_from_x(x)
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
        metric_weights (DataTable[AccumulatedMetric]): Weights for each accumulated metric, component-wise.
    """
    def __init__(self,
                 name: str,
                 maximize: bool,
                 metric_weights: DataTable[AccumulatedMetric]):
        super().__init__(maximize=maximize)
        self.component_name = name
        self._metric_weights = metric_weights

        accumulated_metric_shapes = {
            acc_metric: None for acc_metric in self._metric_weights.keys()}
        self.input_interface = Interface(
            all_shapes={AccumulatedMetric: accumulated_metric_shapes})

        self.output_interface = Interface()

    def _evaluate(self, acc_metrics: DataTable[AccumulatedMetric]) -> np.array:
        """Compute scalar weighted sum of accumulated metrics.

        Args:
            acc_metrics (DataTable[AccumulatedMetric]): Accumulated metrics to reduce.

        Returns:
            np.array: Scalar reduction of the metrics for each point in acc. metrics DataTable
        """
        result = sum(
            weight * np.sum(acc_metrics[metric],
                            axis=tuple(range(1, acc_metrics[metric].ndim)))
            for metric, weight in self._metric_weights.items()
        )
        return result


def multi_metrics_reduction_from_dict(param_dict: Dict[str, Any]) -> MultiMetricsReduction:
    name = param_dict["name"]
    multi_metrics_reduction_type = MultiMetricsReductionType(param_dict["multi_metrics_reduction_type"])
    maximize = param_dict["maximize"]
    if multi_metrics_reduction_type == MultiMetricsReductionType.SCALAR_WEIGHTING:
        metric_weights = {AccumulatedMetric(k): float(v) for \
                          k, v in param_dict["metric_weights"].items()}
        return ScalarWeighting(
            name=name,
            maximize=maximize,
            metric_weights=metric_weights)
    else:
        raise NotImplementedError("Only scalar-weighting multi-metrics reduction implemented.")
    