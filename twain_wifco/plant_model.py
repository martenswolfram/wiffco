from typing import Dict, Any, Callable, Tuple, List
from abc import abstractmethod
from enum import Enum
import numpy as np
import sympy as sp

from twain_wifco.interface import (
    DataTable,
    Component,
    Ambient,
    Control,
    ModelOutput,
    Interface,
    get_default_value,
)
from twain_wifco.scattered_interpolation import (
    ScatteredInterpolatorParams,
    ScatteredInterpolator,
    scattered_interpolator_params_from_dict,
)

# ======================================================================
# Base Model Class
# ======================================================================

class PlantModel(Component):
    """Abstract base class for physical or data-driven plant models.

    A `PlantModel` maps ambient and control conditions to model outputs,
    e.g., computing electrical power and damage rate based on wind speed
    and control setpoints.
    """

    @abstractmethod
    @Component.with_validation
    def evaluate(
        self,
        meteorological_condition: DataTable[Ambient],
        control_input: DataTable[Control],
    ) -> DataTable[ModelOutput]:
        """Evaluate the model for given ambient and control inputs.

        Args:
            meteorological_condition: Ambient conditions such as wind speed or direction.
            control_input: Control variables such as yaw or power regulation.

        Returns:
            DataTable[ModelOutput]: Model output quantities.
        """
        ...

# ======================================================================
# Model Type Enum
# ======================================================================

class ModelType(Enum):
    """Enumeration of available plant model types."""
    FACTORIZED_SCATTERED_INTERPOLATOR = "factorized_scattered_interpolator"
    SYMBOLIC = "symbolic"
    FLORIS = "floris"


# ======================================================================
# Factorized Scattered Interpolation
# ======================================================================

class FactorizedScatteredInterp(PlantModel):
    """Plant model where outputs are the product of two scattered interpolators:
    one for ambient conditions and one for control inputs.
    """

    def __init__(self,
                 name: str,
                 ambient_interp_params: ScatteredInterpolatorParams,
                 control_interp_params: ScatteredInterpolatorParams):
        
        self.component_name = name
        self._ambient_interp = ScatteredInterpolator(ambient_interp_params)
        self._control_interp = ScatteredInterpolator(control_interp_params)

        self.input_interface = Interface(
            all_shapes={
                Ambient: self._ambient_interp.support_data.shapes(),
                Control: self._control_interp.support_data.shapes(),
            }
        )
        self.output_interface = Interface(
            all_shapes={ModelOutput: self._ambient_interp.out_data.shapes()}
        )

    @Component.with_validation
    def evaluate(
        self,
        meteorological_condition: DataTable[Ambient],
        control_input: DataTable[Control],
    ) -> DataTable[ModelOutput]:
        """Evaluate the factorized model."""
        ambient_eval = self._ambient_interp.evaluate(query=meteorological_condition)
        control_eval = self._control_interp.evaluate(query=control_input)

        result = {
            out_var: ambient_eval[out_var] * control_eval[out_var]
            for out_var in self._ambient_interp.out_data_point.keys()
        }
        return DataTable(data=result)

def factorized_scattered_interp_from_dict(
    param_dict: Dict[str, Any]
) -> FactorizedScatteredInterp:
    """Create `FactorizedScatteredInterp` from a configuration dictionary."""
    name = param_dict["name"]
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
    return FactorizedScatteredInterp(
        name=name,
        control_interp_params=control_interp_params,
        ambient_interp_params=ambient_interp_params,
    )

# ======================================================================
# Symbolic Model
# ======================================================================

class SymbolicModel(Component):
    """Parameters for a symbolic model defined by symbolic expressions."""

    def __init__(
        self,
        name: str,
        ambient_list: List[Ambient],
        control_list: List[Control],
        ambient_shapes: Dict[Ambient, Tuple[int, ...]],
        control_shapes: Dict[Control, Tuple[int, ...]],
        output_functions: Dict[ModelOutput, Callable[..., np.ndarray]]
    ):
        self.component_name = name
        self._ambient_list = ambient_list
        self._control_list = control_list
        self._output_functions = output_functions
        
        self.input_interface = Interface(
            all_shapes={
                Ambient: ambient_shapes,
                Control: control_shapes,
            }
        )

        # Determine output shapes via default computation
        default_ambient = list(
            get_default_value(data_var=amb_var, shape=shape) for \
            amb_var, shape in ambient_shapes.items())
        default_control = list(
            get_default_value(data_var=ctrl_var, shape=shape) for \
            ctrl_var, shape in control_shapes.items())
        default_result = {
            model_output: func(*(default_ambient + default_control))
            for model_output, func in self._output_functions.items()
        }

        output_shapes = {output: res.shape for \
                         output, res in default_result.items()}

        self.output_interface = Interface(
            all_shapes={
                ModelOutput: output_shapes
            }
        )

    @Component.with_validation
    def evaluate(
        self,
        meteorological_condition: DataTable[Ambient],
        control_input: DataTable[Control],
    ) -> DataTable[ModelOutput]:
                
        # Input values in correct order
        values = [meteorological_condition[amb_var] for amb_var in self._ambient_list] + \
            [control_input[ctrl_var] for ctrl_var in self._control_list]
        
        result = {
            model_output: func(*values)
            for model_output, func in self._output_functions.items()
        }

        return DataTable(data=result)


def symbolic_model_from_dict(param_dict: Dict[str, Any]) -> SymbolicModel:
    """Construct `SymbolicModel` from a dictionary definition."""
    name = param_dict["name"]
    symbol_mapping_dict: Dict[str, Dict] = param_dict["symbol_mappings"]
    str_to_symbol: Dict[str, sp.Symbol] = {}
    ambient_list: List[Ambient] = [] 
    ambient_shapes: Dict[Ambient, Tuple[int, ...]] = {}
    control_list: List[Control] = []
    control_shapes: Dict[Control, Tuple[int, ...]] = {}
    symbols_list = []
    # Define symbols and mappings for ambient conditions
    for amb_var_str, sym_mapping in symbol_mapping_dict["ambient"].items():
        amb_var = Ambient(amb_var_str)
        sym_str = sym_mapping["name"]
        amb_sym = sp.Symbol(sym_str)
        str_to_symbol[sym_str] = amb_sym
        symbols_list.append(amb_sym)
        ambient_list.append(amb_var)
        ambient_shapes[amb_var] = tuple(sym_mapping["shape"])
    # Define symbols and mappings for control inputs
    for ctrl_var_str, sym_mapping in symbol_mapping_dict["control"].items():
        ctrl_var = Control(ctrl_var_str)
        sym_str = sym_mapping["name"]
        ctrl_sym = sp.Symbol(sym_str)
        str_to_symbol[sym_str] = ctrl_sym
        symbols_list.append(ctrl_sym)        
        control_list.append(ctrl_var)
        control_shapes[ctrl_var] = tuple(sym_mapping["shape"])
    
    # Define output functions
    output_functions: Dict[ModelOutput, Callable[..., np.ndarray]] = {}
    for output, expr_str in param_dict["output_functions"].items():
        expr = sp.sympify(expr_str, locals=str_to_symbol)
        output_functions[ModelOutput(output)] = sp.lambdify(
            symbols_list, expr, "numpy"
        )

    return SymbolicModel(
        name=name,
        ambient_list=ambient_list,
        control_list=control_list,
        ambient_shapes=ambient_shapes,
        control_shapes=control_shapes,
        output_functions=output_functions
    )