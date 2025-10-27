from typing import Dict, Any, Callable
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import sympy as sp

from twain_wifco.interface import (
    DataPoint,
    Component,
    ComponentParams,
    Ambient,
    Control,
    ModelOutput,
    Interface,
)
from twain_wifco.scattered_interpolation import (
    ScatteredInterpolatorParams,
    ScatteredInterpolator,
    scattered_interpolator_params_from_dict,
)

# ======================================================================
# Base Model Class
# ======================================================================

class PlantModel(Component, ABC):
    """Abstract base class for physical or data-driven plant models.

    A `PlantModel` maps ambient and control conditions to model outputs,
    e.g., computing electrical power and damage rate based on wind speed
    and control setpoints.
    """

    def __init__(self, name: str, params: ComponentParams):
        """Initialize a plant model.

        Args:
            name: Name of this model component.
            params: Component parameters defining interfaces and metadata.
        """
        super().__init__(component_name=name, component_params=params)

    def evaluate(
        self,
        meteorological_condition: DataPoint[Ambient],
        control_input: DataPoint[Control],
    ) -> DataPoint[ModelOutput]:
        """Evaluate the model for given ambient and control inputs.

        Args:
            meteorological_condition: Ambient conditions such as wind speed or direction.
            control_input: Control variables such as yaw or power regulation.

        Returns:
            DataPoint[ModelOutput]: Model output quantities.
        """
        self.validate_inputs(
            input_data={Ambient: meteorological_condition, Control: control_input}
        )
        return self._evaluate(
            meteorological_condition=meteorological_condition,
            control_input=control_input,
        )

    @abstractmethod
    def _evaluate(
        self,
        meteorological_condition: DataPoint[Ambient],
        control_input: DataPoint[Control],
    ) -> DataPoint[ModelOutput]:
        """Subclass-specific model evaluation logic."""
        ...


# ======================================================================
# Model Type Enum
# ======================================================================

class ModelType(Enum):
    """Enumeration of available plant model types."""
    FACTORIZED_SCATTERED_INTERPOLATOR = "factorized_scattered_interpolator"
    SYMBOLIC_PRODUCT = "symbolic_product"


# ======================================================================
# Factorized Scattered Interpolation
# ======================================================================

class FactorizedScatteredInterpParams(ComponentParams):
    """Parameters for a factorized scattered interpolator model.

    The model assumes separability between ambient and control factors:
        f(ambient, control) = f_a(ambient) * f_c(control)
    """

    def __init__(
        self,
        control_interp_params: ScatteredInterpolatorParams,
        ambient_interp_params: ScatteredInterpolatorParams,
    ):
        """Initialize parameters for a factorized interpolator.

        Args:
            control_interp_params: Interpolator parameters for the control-dependent factor.
            ambient_interp_params: Interpolator parameters for the ambient-dependent factor.

        Raises:
            ValueError: If output variable shapes of both factors do not match.
        """
        if control_interp_params.out_data.shapes() != ambient_interp_params.out_data.shapes():
            raise ValueError(
                "Output variable shapes of interpolation factors must be identical."
            )

        self.control_interp_params = control_interp_params
        self.ambient_interp_params = ambient_interp_params

    def input_interface(self) -> Interface:
        """Define required inputs for ambient and control factors."""
        return Interface(
            all_shapes={
                Ambient: self.ambient_interp_params.support_data.shapes(),
                Control: self.control_interp_params.support_data.shapes(),
            }
        )

    def output_interface(self) -> Interface:
        """Define the output interface (matching factor outputs)."""
        return Interface(
            all_shapes={ModelOutput: self.ambient_interp_params.out_data.shapes()}
        )


def factorized_scattered_interp_params_from_dict(
    param_dict: Dict[str, Any]
) -> FactorizedScatteredInterpParams:
    """Create `FactorizedScatteredInterpParams` from a configuration dictionary."""
    control_interp_params = scattered_interpolator_params_from_dict(
        param_dict=param_dict["control"],
        support_data_type=Control,
        out_data_type=ModelOutput,
    )
    ambient_interp_params = scattered_interpolator_params_from_dict(
        param_dict=param_dict["ambient"],
        support_data_type=Ambient,
        out_data_type=ModelOutput,
    )
    return FactorizedScatteredInterpParams(
        control_interp_params=control_interp_params,
        ambient_interp_params=ambient_interp_params,
    )


class FactorizedScatteredInterp(PlantModel):
    """Plant model where outputs are the product of two scattered interpolators:
    one for ambient conditions and one for control inputs.
    """

    def __init__(self, name: str, params: FactorizedScatteredInterpParams):
        """Initialize the factorized interpolator model."""
        super().__init__(name=name, params=params)
        self._ambient_interp = ScatteredInterpolator(params.ambient_interp_params)
        self._control_interp = ScatteredInterpolator(params.control_interp_params)

    def _evaluate(
        self,
        meteorological_condition: DataPoint[Ambient],
        control_input: DataPoint[Control],
    ) -> DataPoint[ModelOutput]:
        """Evaluate the factorized model."""
        ambient_eval = self._ambient_interp.evaluate(query=meteorological_condition)
        control_eval = self._control_interp.evaluate(query=control_input)

        result = {
            out_var: ambient_eval[out_var] * control_eval[out_var]
            for out_var in self._ambient_interp.out_data_point.keys()
        }
        return DataPoint(data=result)


# ======================================================================
# Symbolic Model
# ======================================================================

class SymbolicModelParams(ComponentParams):
    """Parameters for a symbolic model defined by symbolic expressions."""

    def __init__(
        self,
        symbols: Dict[str, sp.Symbol],
        control_mappings: Dict[Control, str],
        ambient_mappings: Dict[Ambient, str],
        output_functions: Dict[ModelOutput, Callable[..., np.ndarray]],
    ):
        """Initialize symbolic model parameters.

        Args:
            symbols: Mapping from symbol names to SymPy symbols.
            control_mappings: Mapping from control variables to symbol names.
            ambient_mappings: Mapping from ambient variables to symbol names.
            output_functions: Mapping from output variables to numerical functions.
        """
        self.symbols = symbols
        self.control_mappings = control_mappings
        self.ambient_mappings = ambient_mappings
        self.output_functions = output_functions

    def input_interface(self) -> Interface:
        """Define required input interface for symbolic model."""
        return Interface(
            all_shapes={
                Ambient: {amb: (1,) for amb in self.ambient_mappings.keys()},
                Control: {ctrl: (1,) for ctrl in self.control_mappings.keys()},
            }
        )

    def output_interface(self) -> Interface:
        """Define output interface based on defined output functions."""
        return Interface(
            all_shapes={
                ModelOutput: {out: (1,) for out in self.output_functions.keys()}
            }
        )


def symbolic_model_params_from_dict(param_dict: Dict[str, Any]) -> SymbolicModelParams:
    """Construct `SymbolicModelParams` from a dictionary definition."""
    symbol_mapping_dict: Dict[str, Dict] = param_dict["symbol_mappings"]
    symbols: Dict[str, sp.Symbol] = {}
    control_mappings: Dict[Control, str] = {}
    ambient_mappings: Dict[Ambient, str] = {}

    # Define symbols and mappings
    for ctrl_var, sym_name in symbol_mapping_dict["control"].items():
        symbols[sym_name] = sp.Symbol(sym_name)
        control_mappings[Control(ctrl_var)] = sym_name
    for amb_var, sym_name in symbol_mapping_dict["ambient"].items():
        symbols[sym_name] = sp.Symbol(sym_name)
        ambient_mappings[Ambient(amb_var)] = sym_name

    # Define output functions
    output_functions: Dict[ModelOutput, Callable[..., np.ndarray]] = {}
    for output, expr_str in param_dict["output_functions"].items():
        expr = sp.sympify(expr_str, locals=symbols)
        output_functions[ModelOutput(output)] = sp.lambdify(
            tuple(symbols.values()), expr, "numpy"
        )

    return SymbolicModelParams(
        symbols=symbols,
        control_mappings=control_mappings,
        ambient_mappings=ambient_mappings,
        output_functions=output_functions,
    )


class SymbolicModel(PlantModel):
    """Plant model defined via symbolic expressions evaluated with NumPy."""

    def __init__(self, name: str, params: SymbolicModelParams):
        """Initialize symbolic model."""
        super().__init__(name=name, params=params)
        self._params = params
        self._symbol_order = list(params.symbols.keys())

    def _evaluate(
        self,
        meteorological_condition: DataPoint[Ambient],
        control_input: DataPoint[Control],
    ) -> DataPoint[ModelOutput]:
        """Evaluate symbolic expressions numerically."""
        value_map = {}
        for ctrl_var, symbol_name in self._params.control_mappings.items():
            value_map[symbol_name] = control_input[ctrl_var]
        for amb_var, symbol_name in self._params.ambient_mappings.items():
            value_map[symbol_name] = meteorological_condition[amb_var]

        symbol_values = [value_map[s] for s in self._symbol_order]
        result = {
            model_output: func(*symbol_values)
            for model_output, func in self._params.output_functions.items()
        }

        return DataPoint(data=result)
