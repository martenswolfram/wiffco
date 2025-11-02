from __future__ import annotations
from typing import Dict, Any
from abc import abstractmethod
from enum import Enum

from twain_wifco.interface import (
    Component,
    Ambient,
    ModelOutput,
    Control,
    Aggregate,
    DataTable,
    Interface,
)

from twain_wifco.symbolic import (
    symbolic_function_from_dict,
    SymbolicFunction
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
        ambient: DataTable[Ambient] | None = None,
        control: DataTable[Control] | None = None,
        model_output: DataTable[ModelOutput] | None = None,
    ) -> DataTable[Aggregate]:
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
    SYMBOLIC = "symbolic"

class SymbolicAggregation(Component):
    """Symbolic aggregation defined by symbolic expressions."""

    def __init__(self,
                 name: str,
                 symbolic_function: SymbolicFunction):
        self.component_name = name
        self._symbolic_function = symbolic_function
        self.input_interface = self._symbolic_function.input_interface()
        self.output_interface = self._symbolic_function.output_interface()

    @Component.with_validation
    def compute_aggregate(
        self,
        ambient: DataTable[Ambient] | None = None,
        control: DataTable[Control] | None = None,
        model_output: DataTable[ModelOutput] | None = None,
    ) -> DataTable[Aggregate]:
        
        if ambient is None and control is None and model_output is None:
            raise ValueError("At least one input must be not None for symbolic aggregation computation.")
        function_output = self._symbolic_function.evaluate(
            {
                Ambient: ambient,
                Control: control,
                ModelOutput: model_output
            }
        )
        return function_output[Aggregate]

# ======================================================================
# Helper: Construct from Dictionary
# ======================================================================

def symbolic_aggregation_from_dict(param_dict: Dict[str, Any]) -> SymbolicAggregation:
    """Construct `SymbolicAggregation` from a dictionary definition."""
    name = param_dict["name"]
    symbolic_function = symbolic_function_from_dict(param_dict=param_dict)

    return SymbolicAggregation(
        name=name,
        symbolic_function=symbolic_function
    )