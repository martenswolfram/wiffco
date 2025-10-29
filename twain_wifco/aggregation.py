from __future__ import annotations
from typing import Dict, Any, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from enum import Enum

from twain_wifco.interface import (
    Component,
    ComponentParams,
    Ambient,
    ModelOutput,
    Control,
    Aggregated,
    DataPoint,
    Interface,
)

# ======================================================================
# Base Aggregation Component
# ======================================================================

class Aggregation(Component, ABC):
    """Abstract base class for output aggregation components.

    An aggregation combines multiple sources of data (model outputs,
    ambient conditions, and control setpoints) into higher-level
    aggregated quantities (e.g., revenue rate or damage rate).
    """

    def __init__(self, name: str, params: ComponentParams):
        """Initialize the aggregation component.

        Args:
            name: Name of this aggregation component.
            params: Parameter object defining configuration and interfaces.
        """
        super().__init__(component_name=name, component_params=params)

    def compute_aggregate(
        self,
        model_output: DataPoint[ModelOutput],
        ambient_condition: DataPoint[Ambient],
        control_setpoints: DataPoint[Control],
    ) -> DataPoint[Aggregated]:
        """Compute aggregated outputs from model, ambient, and control data.

        This method performs input validation before delegating the actual
        computation to the subclass-specific implementation `_compute()`.

        Args:
            model_output: Model output data (e.g. electrical power, damage rate).
            ambient_condition: Ambient condition data (e.g. wind speed, direction).
            control_setpoints: Control input data (e.g. yaw steering, power regulation).

        Returns:
            DataPoint[Aggregated]: Computed aggregated outputs.
        """
        self.validate_input(
            input_data={
                ModelOutput: model_output,
                Ambient: ambient_condition,
                Control: control_setpoints,
            }
        )

        return self._compute(
            model_output=model_output,
            ambient_condition=ambient_condition,
            control_setpoints=control_setpoints,
        )

    @abstractmethod
    def _compute(
        self,
        model_output: DataPoint[ModelOutput],
        ambient_condition: DataPoint[Ambient],
        control_setpoints: DataPoint[Control],
    ) -> DataPoint[Aggregated]:
        """Subclass-specific implementation of the aggregation.

        Args:
            model_output: Model output data.
            ambient_condition: Ambient data.
            control_setpoints: Control data.

        Returns:
            DataPoint[Aggregated]: Computed aggregated results.
        """
        ...


# ======================================================================
# Aggregation Type Enum
# ======================================================================

class AggregationType(Enum):
    """Enumeration of available aggregation types."""
    SIMPLE_PRODUCT = "simple_product"


# ======================================================================
# Aggregate Mapping
# ======================================================================

@dataclass(frozen=True)
class ProductAggregateMapping:
    """Defines which input variables contribute to a given aggregated output.

    Attributes:
        model_inputs: Set of model output variables used in the product.
        ambient_inputs: Set of ambient condition variables used in the product.
        control_inputs: Set of control variables used in the product.
    """
    model_inputs: Set[ModelOutput] = field(default_factory=set)
    ambient_inputs: Set[Ambient] = field(default_factory=set)
    control_inputs: Set[Control] = field(default_factory=set)


# ======================================================================
# Parameter Class
# ======================================================================

class SimpleProductParams(ComponentParams):
    """Defines parameters and interface structure for simple product aggregation."""

    def __init__(self, aggregate_mappings: Dict[Aggregated, ProductAggregateMapping]):
        """Initialize parameters for the simple product aggregation.

        Args:
            aggregate_mappings: Mapping from aggregated outputs to their
                corresponding input variable sets.
        """
        self._aggregate_mappings = aggregate_mappings

    @property
    def aggregate_mappings(self) -> Dict[Aggregated, ProductAggregateMapping]:
        """Mapping from aggregated outputs to their contributing inputs."""
        return self._aggregate_mappings

    def input_interface(self) -> Interface:
        """Define the required input variables for this aggregation.

        Returns:
            Interface: Input variable interface containing required variables
            grouped by type (model, ambient, control).
        """
        model_shapes, ambient_shapes, control_shapes = {}, {}, {}

        for mapping in self._aggregate_mappings.values():
            model_shapes.update({var: None for var in mapping.model_inputs})
            ambient_shapes.update({var: None for var in mapping.ambient_inputs})
            control_shapes.update({var: None for var in mapping.control_inputs})

        return Interface(
            all_shapes={
                ModelOutput: model_shapes,
                Ambient: ambient_shapes,
                Control: control_shapes,
            }
        )

    def output_interface(self) -> Interface:
        """Define the expected output structure of this aggregation.

        Returns:
            Interface: Output variable interface defining shapes for each
            aggregated variable (by default scalar outputs of shape (1,)).
        """
        return Interface(
            all_shapes={
                Aggregated: {aggr: (1,) for aggr in self._aggregate_mappings.keys()}
            }
        )


# ======================================================================
# Helper: Construct Parameters from Dictionary
# ======================================================================

def simple_product_params_from_dict(param_dict: Dict[str, Any]) -> SimpleProductParams:
    """Construct a `SimpleProductParams` instance from a plain dictionary.

    This function enables loading configuration data from JSON or YAML files.

    Args:
        param_dict: Dictionary containing the field ``aggregate_mappings`` with
            variable names under ``from_model``, ``from_ambient``, and ``from_control``.

    Returns:
        SimpleProductParams: Parsed parameter object.
    """
    aggregate_mappings: Dict[Aggregated, ProductAggregateMapping] = {}

    for agg_key, mapping_def in param_dict["aggregate_mappings"].items():
        aggregate_mappings[Aggregated(agg_key)] = ProductAggregateMapping(
            model_inputs={ModelOutput(m) for m in mapping_def["from_model"]},
            ambient_inputs={Ambient(a) for a in mapping_def["from_ambient"]},
            control_inputs={Control(c) for c in mapping_def["from_control"]},
        )

    return SimpleProductParams(aggregate_mappings=aggregate_mappings)


# ======================================================================
# Aggregation Implementation
# ======================================================================

class SimpleProduct(Aggregation):
    """Aggregation that computes the product of selected model, ambient,
    and control variables for each defined aggregate output.
    """

    def __init__(self, name: str, params: SimpleProductParams):
        """Initialize the simple product aggregation.

        Args:
            name: Name of the aggregation component.
            params: Parameter object defining variable mappings.
        """
        super().__init__(name=name, params=params)
        self._mappings = params.aggregate_mappings

    def _prod_values(self, data_point: DataPoint, variables: Set) -> float:
        """Compute the product of all variable values in a given data point.

        Args:
            data_point: DataPoint containing variable arrays.
            variables: Set of variables whose values should be multiplied.

        Returns:
            float: Product of all variable values (1.0 if empty).
        """
        return np.prod([data_point[v] for v in variables]) if variables else 1.0

    def _compute(
        self,
        model_output: DataPoint[ModelOutput],
        ambient_condition: DataPoint[Ambient],
        control_setpoints: DataPoint[Control],
    ) -> DataPoint[Aggregated]:
        """Compute aggregated outputs using the defined variable mappings.

        Args:
            model_output: Model output data.
            ambient_condition: Ambient data.
            control_setpoints: Control data.

        Returns:
            DataPoint[Aggregated]: Aggregated output data.
        """
        aggregated_output: Dict[Aggregated, np.ndarray] = {}

        for out_var, mapping in self._mappings.items():
            res = 1
            for model_in in mapping.model_inputs:
                res *= model_output[model_in]
            for ambient_in in mapping.ambient_inputs:
                res *= ambient_condition[ambient_in]
            for control_in in mapping.control_inputs:
                res *= control_setpoints[control_in]
            aggregated_output[out_var] = res

        return DataPoint(data=aggregated_output)
