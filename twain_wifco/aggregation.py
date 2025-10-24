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


# ===============================================================
# Base Aggregation Component
# ===============================================================

class Aggregation(Component, ABC):
    """Base class for output aggregations combining model, ambient, and control data."""

    def __init__(self, name: str, params: ComponentParams):
        """
        Parameters
        ----------
        name : str
            Name of this aggregation component.
        params : ComponentParams
            Parameter object defining parameters and interfaces.
        """
        super().__init__(component_name=name, component_params=params)

    def compute_aggregate(
        self,
        model_output: DataPoint[ModelOutput],
        ambient_condition: DataPoint[Ambient],
        control_setpoints: DataPoint[Control],
    ) -> DataPoint[Aggregated]:
        """Compute aggregated outputs from model, ambient, and control data."""

        self.validate_inputs(
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
        """Subclass-specific implementation of the aggregation."""
        ...


# ===============================================================
# Aggregation Type Enum
# ===============================================================

class AggregationType(Enum):
    SIMPLE_PRODUCT = "simple_product"


# ===============================================================
# Aggregate Mapping (dataclass)
# ===============================================================

@dataclass(frozen=True)
class ProductAggregateMapping:
    """Defines which input variables contribute to a given aggregate output."""

    model_inputs: Set[ModelOutput] = field(default_factory=set)
    ambient_inputs: Set[Ambient] = field(default_factory=set)
    control_inputs: Set[Control] = field(default_factory=set)


# ===============================================================
# Parameter Class
# ===============================================================

class SimpleProductParams(ComponentParams):
    """Defines parameters and interface structure for simple product aggregation."""

    def __init__(self, aggregate_mappings: Dict[Aggregated, ProductAggregateMapping]):
        self._aggregate_mappings = aggregate_mappings

    @property
    def aggregate_mappings(self) -> Dict[Aggregated, ProductAggregateMapping]:
        """Mapping from aggregated outputs to their contributing inputs."""
        return self._aggregate_mappings

    def input_interface(self) -> Interface:
        """Define the required inputs for the aggregation."""
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
        """Define the expected output structure."""
        return Interface(
            all_shapes={
                Aggregated: {aggr: (1,) for aggr in self._aggregate_mappings.keys()}
            }
        )


# ===============================================================
# Helper: construct params from dictionary
# ===============================================================

def simple_product_params_from_dict(param_dict: Dict[str, Any]) -> SimpleProductParams:
    """Create SimpleProductParams instance from a plain dictionary (e.g. loaded from JSON)."""

    aggregate_mappings: Dict[Aggregated, ProductAggregateMapping] = {}

    for agg_key, mapping_def in param_dict["aggregate_mappings"].items():
        aggregate_mappings[Aggregated(agg_key)] = ProductAggregateMapping(
            model_inputs={ModelOutput(m) for m in mapping_def["from_model"]},
            ambient_inputs={Ambient(a) for a in mapping_def["from_ambient"]},
            control_inputs={Control(c) for c in mapping_def["from_control"]},
        )

    return SimpleProductParams(aggregate_mappings=aggregate_mappings)


# ===============================================================
# Aggregation Implementation
# ===============================================================

class SimpleProduct(Aggregation):
    """Aggregation that computes products of selected model, ambient, and control variables."""

    def __init__(self, name: str, params: SimpleProductParams):
        super().__init__(name=name, params=params)
        self._mappings = params.aggregate_mappings

    def _prod_values(self, data_point: DataPoint, variables: Set) -> float:
        """Helper function: product of all variable values."""
        return np.prod([data_point[v] for v in variables]) if variables else 1.0

    def _compute(
        self,
        model_output: DataPoint[ModelOutput],
        ambient_condition: DataPoint[Ambient],
        control_setpoints: DataPoint[Control],
    ) -> DataPoint[Aggregated]:
        """Compute aggregate outputs for all defined mappings."""

        aggregated_output: Dict[Aggregated, np.ndarray] = {}

        for out_var, mapping in self._mappings.items():
            res = (
                self._prod_values(model_output, mapping.model_inputs)
                * self._prod_values(ambient_condition, mapping.ambient_inputs)
                * self._prod_values(control_setpoints, mapping.control_inputs)
            )
            aggregated_output[out_var] = np.array([res])

        return DataPoint(data=aggregated_output)
