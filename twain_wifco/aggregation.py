from __future__ import annotations
from typing import Dict, Any, Set
from abc import abstractmethod
from dataclasses import dataclass, field
import numpy as np
from enum import Enum

from twain_wifco.interface import (
    Component,
    Ambient,
    ModelOutput,
    Control,
    Aggregated,
    DataTable,
    Interface,
)

# ======================================================================
# Base Aggregation Component
# ======================================================================

class Aggregation(Component):
    """Abstract base class for output aggregation components.

    An aggregation combines multiple sources of data (model outputs,
    ambient conditions, and control setpoints) into higher-level
    aggregated quantities (e.g., revenue rate or damage rate).
    """

    @abstractmethod
    @Component.with_validation
    def compute_aggregate(
        self,
        model_output: DataTable[ModelOutput],
        ambient: DataTable[Ambient],
        control: DataTable[Control],
    ) -> DataTable[Aggregated]:
        """Compute aggregated outputs from model, ambient, and control data.

        This method performs input validation before delegating the actual
        computation to the subclass-specific implementation `_compute()`.

        Args:
            model_output: Model output data (e.g. electrical power, damage rate).
            ambient: Ambient condition data (e.g. wind speed, direction).
            control: Control input data (e.g. yaw steering, power regulation).

        Returns:
            DataTable[Aggregated]: Computed aggregated outputs.
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
        model: Set of model output variables used in the product.
        ambient: Set of ambient condition variables used in the product.
        control: Set of control variables used in the product.
    """
    model_output: Set[ModelOutput] = field(default_factory=set)
    ambient: Set[Ambient] = field(default_factory=set)
    control: Set[Control] = field(default_factory=set)

class SimpleProduct(Component):
    """Defines parameters and interface structure for simple product aggregation."""

    def __init__(self,
                 name: str,
                 aggregate_mappings: Dict[Aggregated, ProductAggregateMapping]):
        """Initialize parameters for the simple product aggregation.

        Args:
            aggregate_mappings: Mapping from aggregated outputs to their
                corresponding input variable sets.
        """
        self.component_name = name
        self._aggregate_mappings = aggregate_mappings

        model_shapes, ambient_shapes, control_shapes = {}, {}, {}

        for mapping in self._aggregate_mappings.values():
            model_shapes.update({var: None for var in mapping.model_output})
            ambient_shapes.update({var: None for var in mapping.ambient})
            control_shapes.update({var: None for var in mapping.control})

        self.input_interface = Interface(
            all_shapes={
                ModelOutput: model_shapes,
                Ambient: ambient_shapes,
                Control: control_shapes,
            }
        )


        self.output_interface = Interface(
            all_shapes={
                Aggregated: {aggr: (1,) for aggr in self._aggregate_mappings.keys()}
            }
        )

    def _prod_values(self, data_point: DataTable, variables: Set) -> float:
        """Compute the product of all variable values in a given data point.

        Args:
            data_point: DataTable containing variable arrays.
            variables: Set of variables whose values should be multiplied.

        Returns:
            float: Product of all variable values (1.0 if empty).
        """
        return np.prod([data_point[v] for v in variables]) if variables else 1.0

    @Component.with_validation
    def compute_aggregate(
        self,
        model_output: DataTable[ModelOutput],
        ambient: DataTable[Ambient],
        control: DataTable[Control],
    ) -> DataTable[Aggregated]:
        """Compute aggregated outputs using the defined variable mappings.

        Args:
            model_output: Model output data.
            ambient: Ambient data.
            control: Control data.

        Returns:
            DataTable[Aggregated]: Aggregated output data.
        """
        aggregated_output: Dict[Aggregated, np.ndarray] = {}

        for out_var, mapping in self._aggregate_mappings.items():
            res = 1
            for model_in in mapping.model_output:
                res *= model_output[model_in]
            for ambient_in in mapping.ambient:
                res *= ambient[ambient_in]
            for control_in in mapping.control:
                res *= control[control_in]
            aggregated_output[out_var] = res

        return DataTable(data=aggregated_output)

# ======================================================================
# Helper: Construct from Dictionary
# ======================================================================

def simple_product_from_dict(param_dict: Dict[str, Any]) -> SimpleProduct:
    """Construct a `SimpleProduct` instance from a plain dictionary.

    This function enables loading configuration data from JSON or YAML files.

    Args:
        param_dict: Dictionary containing the field ``aggregate_mappings`` with
            variable names under ``from_model``, ``from_ambient``, and ``from_control``.

    Returns:
        SimpleProduct: Parsed SimpleProduct aggregation object.
    """
    name = param_dict["name"]
    aggregate_mappings: Dict[Aggregated, ProductAggregateMapping] = {}

    for agg_key, mapping_def in param_dict["aggregate_mappings"].items():
        aggregate_mappings[Aggregated(agg_key)] = ProductAggregateMapping(
            model_output={ModelOutput(m) for m in mapping_def["from_model"]},
            ambient={Ambient(a) for a in mapping_def["from_ambient"]},
            control={Control(c) for c in mapping_def["from_control"]},
        )

    return SimpleProduct(name=name, 
                         aggregate_mappings=aggregate_mappings)


