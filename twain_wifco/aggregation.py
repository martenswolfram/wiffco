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

    def compute_aggregate(
        self,
        ambient: DataTable[Ambient] | None = None,
        control: DataTable[Control] | None = None,
        model_output: DataTable[ModelOutput] | None = None,
    ) -> DataTable[Aggregate]:
        input_tables = [table for table in [ambient, control, model_output] if table is not None]
        self.validate_input(input_tables=input_tables)
        return self._compute_aggregate(ambient=ambient,
                                       control=control,
                                       model_output=model_output)
        
    @abstractmethod
    def _compute_aggregate(
        self,
        ambient: DataTable[Ambient] | None = None,
        control: DataTable[Control] | None = None,
        model_output: DataTable[ModelOutput] | None = None,
    ) -> DataTable[Aggregate]:
        ...
    

# ======================================================================
# Aggregation Type Enum
# ======================================================================

class AggregationType(Enum):
    """Enumeration of available aggregation types."""
    SYMBOLIC = "symbolic"

class SymbolicAggregation(Aggregation):
    """Symbolic aggregation defined by symbolic expressions."""

    def __init__(self,
                 name: str,
                 symbolic_function: SymbolicFunction):
        self.component_name = name
        self._symbolic_function = symbolic_function
        self.input_interface = self._symbolic_function.input_interface()
        self.output_interface = self._symbolic_function.output_interface()

    def _compute_aggregate(
        self,
        ambient: DataTable[Ambient] | None = None,
        control: DataTable[Control] | None = None,
        model_output: DataTable[ModelOutput] | None = None,
    ) -> DataTable[Aggregate]:
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