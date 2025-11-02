from typing import Dict, Any
from abc import abstractmethod
from enum import Enum

from twain_wifco.interface import (
    DataTable,
    Component,
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
from twain_wifco.symbolic import (
    symbolic_function_from_dict,
    SymbolicFunction
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
        meteorological: DataTable[Ambient],
        control: DataTable[Control],
    ) -> DataTable[ModelOutput]:
        """Evaluate the model for given ambient and control inputs.

        Args:
            meteorological: Ambient conditions such as wind speed or direction.
            control: Control variables such as yaw or power regulation.

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
                Ambient: self._ambient_interp.support_shapes,
                Control: self._control_interp.support_shapes,
            }
        )
        self.output_interface = Interface(
            all_shapes={ModelOutput: self._ambient_interp.out_shapes}
        )

    @Component.with_validation
    def evaluate(
        self,
        meteorological: DataTable[Ambient],
        control: DataTable[Control],
    ) -> DataTable[ModelOutput]:
        """Evaluate the factorized model."""
        ambient_eval = self._ambient_interp.evaluate(query=meteorological)
        control_eval = self._control_interp.evaluate(query=control)

        result = {
            out_var: ambient_eval[out_var] * control_eval[out_var]
            for out_var in self._ambient_interp.out_shapes.keys()
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
        symbolic_function: SymbolicFunction
    ):
        self.component_name = name
        self._symbolic_function = symbolic_function
        
        self.input_interface = Interface(
            all_shapes={
                Ambient: self._symbolic_function.input_shapes[Ambient],
                Control: self._symbolic_function.input_shapes[Control],
            }
        )
        self.output_interface = Interface(
            all_shapes={
                ModelOutput: self._symbolic_function.output_shapes[ModelOutput]
            }
        )

    @Component.with_validation
    def evaluate(
        self,
        meteorological: DataTable[Ambient],
        control: DataTable[Control],
    ) -> DataTable[ModelOutput]:
                
        function_output = self._symbolic_function.evaluate(
            {
                Ambient: meteorological,
                Control: control
            }
        )
        return function_output[ModelOutput]

def symbolic_model_from_dict(param_dict: Dict[str, Any]) -> SymbolicModel:
    """Construct `SymbolicModel` from a dictionary definition."""
    name = param_dict["name"]
    symbolic_function = symbolic_function_from_dict(param_dict=param_dict)

    return SymbolicModel(
        name=name,
        symbolic_function=symbolic_function
    )