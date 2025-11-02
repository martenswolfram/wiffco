from typing import Dict, Any
from abc import abstractmethod
from enum import Enum
import numpy as np 
from twain_wifco.statistics import (
    Statistics)
from twain_wifco.interface import (
    Component,
    Aggregate,
    AccumulatedMetric,
    DataTable,
    Interface)

# ----------------------------
# Metrics Accumulation Base Class
# ----------------------------
class MetricsAccumulation(Component):
    """Abstract base class for accumulation of aggregate variables over time."""

    @abstractmethod
    @Component.with_validation
    def acc_metrics(self,
                    aggregate: DataTable[Aggregate]
                    ) -> DataTable[AccumulatedMetric]:
        """Accumulate metrics for a given aggregate over a duration.

        Args:
            aggregate (DataTable[Aggregated]): Input aggregated data.
            duration (int): Duration over which to accumulate.

        Returns:
            DataTable[AccumulatedMetric]: Accumulated metrics.
        """
        ...

    @abstractmethod
    def expected_acc_metrics(self,
                             aggregate_statistics: Statistics[Aggregate]
                             ) -> DataTable[AccumulatedMetric]:
        """Compute expected accumulated metrics given aggregate statistics.

        Args:
            aggregate_statistics (Statistics[Aggregated]): Statistics of aggregated data.
            duration (int): Duration over which to accumulate.

        Returns:
            DataTable[AccumulatedMetric]: Expected accumulated metrics.
        """
        ...

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
                 aggregate: Aggregate,
                 collapse: bool,
                 discount_rate: float):
        self.aggregate = aggregate
        self.collapse = collapse
        self.discount_rate = discount_rate


class DiscountedIntegration(Component):
    """Parameters for DiscountedIntegration.

    Attributes:
        integration_mappings (Dict[AccumulatedMetric, IntegrationMapping]): Mapping of metrics to integration rules.
    """
    def __init__(self,
                 name: str,
                 integration_mappings: Dict[AccumulatedMetric, IntegrationMapping],
                 duration: int):
        
        self.component_name = name
        self._integration_mappings = integration_mappings
        self._duration = duration

        aggregated_shapes = {mapping.aggregate: None for \
                             mapping in self._integration_mappings.values()}
        
        self.input_interface = Interface(all_shapes={Aggregate: aggregated_shapes})

        accumulated_metric_shapes = {metric: None for metric in self._integration_mappings.keys()}
        self.output_interface = Interface(all_shapes={AccumulatedMetric: accumulated_metric_shapes})

    @Component.with_validation
    def acc_metrics(self,
                     aggregate: DataTable[Aggregate]) -> DataTable[AccumulatedMetric]:
        accumulated_metrics = {}
        for acc_metric, mapping in self._integration_mappings.items():
            discount_rate = mapping.discount_rate
            value = aggregate[mapping.aggregate]
            if discount_rate == 0:
                accumulated_metrics[acc_metric] = self._duration * value
            else:
                discount_factor = 1 / (1 + discount_rate)
                duration_discount_factor = (1 - discount_factor ** self._duration) / (1 - discount_factor)
                accumulated_metrics[acc_metric] = value * duration_discount_factor
            if mapping.collapse:
                accumulated_metrics[acc_metric] = np.array(np.sum(accumulated_metrics[acc_metric]))
        return DataTable(accumulated_metrics)

    def expected_acc_metrics(self,
                              aggregate_statistics: Statistics[Aggregate]) -> DataTable[AccumulatedMetric]:
        aggregate_expectation = aggregate_statistics.expected_value()
        return self.acc_metrics(aggregate=aggregate_expectation)
    
def discounted_integrator_from_dict(param_dict: Dict[str, Any]) -> DiscountedIntegration:
    
    name = param_dict["name"]
    integration_mappings = {}
    for acc_metric, mapping_dict in param_dict["integration_mappings"].items():
        integration_mappings[AccumulatedMetric(acc_metric)] = IntegrationMapping(
            aggregate=Aggregate(mapping_dict["aggregate"]),
            collapse=bool(mapping_dict["collapse"]),
            discount_rate=float(mapping_dict["discount_rate"])
        )
    duration = param_dict["duration"]
    return DiscountedIntegration(name=name,
                                 integration_mappings=integration_mappings,
                                 duration=duration)

