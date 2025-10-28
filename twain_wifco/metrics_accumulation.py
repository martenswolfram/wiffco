from typing import Dict, Any
from abc import abstractmethod
from enum import Enum
import numpy as np 
from twain_wifco.statistics import (
    Statistics)
from twain_wifco.interface import (
    Component,
    ComponentParams,
    Aggregated,
    AccumulatedMetric,
    DataPoint,
    Interface)

# ----------------------------
# Metrics Accumulation Base Class
# ----------------------------
class MetricsAccumulation(Component):
    """Abstract base class for accumulation of aggregate variables over time."""

    def __init__(self,
                 accumulation_name: str,
                 accumulation_params: ComponentParams):
        super().__init__(component_name=accumulation_name,
                         component_params=accumulation_params)

    def acc_metrics(self,
                    aggregate: DataPoint[Aggregated],
                    duration: int) -> DataPoint[AccumulatedMetric]:
        """Accumulate metrics for a given aggregate over a duration.

        Args:
            aggregate (DataPoint[Aggregated]): Input aggregated data.
            duration (int): Duration over which to accumulate.

        Returns:
            DataPoint[AccumulatedMetric]: Accumulated metrics.
        """
        self.validate_inputs(input_data={Aggregated: aggregate})
        return self._acc_metrics(aggregate=aggregate, duration=duration)

    def expected_acc_metrics(self,
                             aggregate_statistics: Statistics[Aggregated],
                             duration: int) -> DataPoint[AccumulatedMetric]:
        """Compute expected accumulated metrics given aggregate statistics.

        Args:
            aggregate_statistics (Statistics[Aggregated]): Statistics of aggregated data.
            duration (int): Duration over which to accumulate.

        Returns:
            DataPoint[AccumulatedMetric]: Expected accumulated metrics.
        """
        self.validate_input_shapes(input_shapes={
            Aggregated: aggregate_statistics.output_interface.shapes(data_type=Aggregated)
        })
        return self._expected_acc_metrics(aggregate_statistics=aggregate_statistics,
                                          duration=duration)

    @abstractmethod
    def _acc_metrics(self,
                     aggregate: DataPoint[Aggregated],
                     duration: int) -> DataPoint[AccumulatedMetric]:
        """Implementation-specific accumulation logic."""
        pass

    @abstractmethod
    def _expected_acc_metrics(self,
                              aggregate_statistics: Statistics[Aggregated],
                              duration: int) -> DataPoint[AccumulatedMetric]:
        """Implementation-specific expected accumulation logic."""
        pass


# ----------------------------
# MetricsAccumulation Types
# ----------------------------
class MetricsAccumulationType(Enum):
    DISCOUNTED_INTEGRATION = "discounted_integration"


# ----------------------------
# Discounted Integration Implementation
# ----------------------------
class IntegrationMapping:
    """Mapping for integration with optional discounting.

    Args:
        aggregate (Aggregated): The aggregate variable.
        collapse (bool): Specifies whether arrays should be collapse into a scalar
        discount_rate (float): Discount rate applied per time step.
    """
    def __init__(self,
                 aggregate: Aggregated,
                 collapse: bool,
                 discount_rate: float):
        self.aggregate = aggregate
        self.collapse = collapse
        self.discount_rate = discount_rate


class DiscountedIntegrationParams(ComponentParams):
    """Parameters for DiscountedIntegration.

    Attributes:
        integration_mappings (Dict[AccumulatedMetric, IntegrationMapping]): Mapping of metrics to integration rules.
    """
    def __init__(self,
                 integration_mappings: Dict[AccumulatedMetric, IntegrationMapping]):
        self.integration_mappings = integration_mappings

    def input_interface(self) -> Interface:
        aggregated_shapes = {mapping.aggregate: None for mapping in self.integration_mappings.values()}
        return Interface(all_shapes={Aggregated: aggregated_shapes})

    def output_interface(self) -> Interface:
        accumulated_metric_shapes = {metric: None for metric in self.integration_mappings.keys()}
        return Interface(all_shapes={AccumulatedMetric: accumulated_metric_shapes})


def discounted_integrator_params_from_dict(param_dict: Dict[str, Any]) -> DiscountedIntegrationParams:
    """Construct DiscountedIntegrationParams from a dictionary.

    Args:
        param_dict (Dict[str, Any]): Dictionary containing integration mappings.

    Returns:
        DiscountedIntegrationParams: Constructed parameters.
    """
    integration_mappings = {}
    for acc_metric, mapping_dict in param_dict["integration_mappings"].items():
        integration_mappings[AccumulatedMetric(acc_metric)] = IntegrationMapping(
            aggregate=Aggregated(mapping_dict["aggregate"]),
            collapse=bool(mapping_dict["collapse"]),
            discount_rate=float(mapping_dict["discount_rate"])
        )
    return DiscountedIntegrationParams(integration_mappings=integration_mappings)


class DiscountedIntegration(MetricsAccumulation):
    """Accumulate metrics using discounted integration over time."""

    def __init__(self,
                 accumulation_name: str,
                 accumulation_params: DiscountedIntegrationParams):
        super().__init__(accumulation_name=accumulation_name,
                         accumulation_params=accumulation_params)
        self.integration_mappings = accumulation_params.integration_mappings

    def _acc_metrics(self,
                     aggregate: DataPoint[Aggregated],
                     duration: int) -> DataPoint[AccumulatedMetric]:
        accumulated_metrics = {}
        for acc_metric, mapping in self.integration_mappings.items():
            discount_rate = mapping.discount_rate
            value = aggregate[mapping.aggregate]
            if discount_rate == 0:
                accumulated_metrics[acc_metric] = duration * value
            else:
                discount_factor = 1 / (1 + discount_rate)
                duration_discount_factor = (1 - discount_factor ** duration) / (1 - discount_factor)
                accumulated_metrics[acc_metric] = value * duration_discount_factor
            if mapping.collapse:
                accumulated_metrics[acc_metric] = np.array(np.sum(accumulated_metrics[acc_metric]))
        return DataPoint(accumulated_metrics)

    def _expected_acc_metrics(self,
                              aggregate_statistics: Statistics[Aggregated],
                              duration: int) -> DataPoint[AccumulatedMetric]:
        aggregate_expectation = aggregate_statistics.expected_value()
        return self._acc_metrics(aggregate=aggregate_expectation, duration=duration)