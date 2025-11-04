from typing import Dict, Any, List
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

from twain_wifco.floris_model.floris_config import (
    FlorisModel,
    wifco2floris,
    configure_floris_model
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

    def evaluate(
        self,
        meteorological: DataTable[Ambient],
        control: DataTable[Control],
    ) -> DataTable[ModelOutput]:
        self.validate_input(input_tables=[meteorological, control])
        return self._evaluate(meteorological=meteorological,
                              control=control)

    @abstractmethod
    def _evaluate(
        self,
        meteorological: DataTable[Ambient],
        control: DataTable[Control],
    ) -> DataTable[ModelOutput]:
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

    def _evaluate(
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

class SymbolicModel(PlantModel):
    """Symbolic model defined by symbolic expressions."""

    def __init__(
        self,
        name: str,
        symbolic_function: SymbolicFunction
    ):
        self.component_name = name
        self._symbolic_function = symbolic_function
        self.input_interface = self._symbolic_function.input_interface()
        self.output_interface = self._symbolic_function.output_interface()
        
    def _evaluate(
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

class FlorisWindFarmModel(PlantModel):

    def __init__(self,
                 name: str,
                 floris_model: FlorisModel,
                 ambient_variables: List[Ambient] = [],
                 control_variables: List[Control] = [],
                 output_variables: List[ModelOutput] = []):
        
        self.component_name = name
        self._floris_model = floris_model
        self._n_turbines = self._floris_model.n_turbines
        self.input_interface = Interface(
            all_shapes={Ambient: {amb_var: () for amb_var in ambient_variables},
                        Control: {ctrl_var: (self._n_turbines,) for ctrl_var in control_variables}})

        self.output_interface = Interface(
            all_shapes={ModelOutput: {out_var: (self._n_turbines,) for out_var in output_variables}})

    def _evaluate(
        self,
        meteorological: DataTable[Ambient],
        control: DataTable[Control],
    ) -> DataTable[ModelOutput]:
        
        num_points = len(meteorological)
        # Assume all ambient conditions are scalar, all control inputs are per turbine
        kwargs = {wifco2floris(amb): val for amb, val in meteorological.items()} | \
                {wifco2floris(ctrl): val for ctrl, val in control.items()}
        
        self._floris_model.set(**kwargs)
        self._floris_model.run()

        output_data = {}
        for output, shape in self.output_interface.shapes[ModelOutput].items():
            match output:
                case ModelOutput.ELECTRICAL_POWER:
                    if shape == ():
                        output_data[output] = self._floris_model.get_farm_power()
                    else:
                        output_data[output] = self._floris_model.get_turbine_powers().reshape((num_points,) + shape)
                case _:
                    raise ValueError(f"Model output {output} not provided by FLORIS model.")

        return DataTable(output_data)
    

def floris_model_from_dict(param_dict: Dict[str, Any]):
    name = param_dict.get("name", "FLORIS wind farm model")
    floris_model = configure_floris_model(
        floris_config_path_str=param_dict["floris_config_file"],
        wind_farm_layout_path_str=param_dict.get("wind_farm_layout_file", None)
    )

    ambient_variables = [Ambient(amb_var) for amb_var in param_dict["ambient_variables"]]
    control_variables = [Control(ctrl_var) for ctrl_var in param_dict["control_variables"]]
    output_variables = [ModelOutput(out_var) for out_var in param_dict["output_variables"]]
    
    return FlorisWindFarmModel(name=name,
                               floris_model=floris_model,
                               ambient_variables=ambient_variables,
                               control_variables=control_variables,
                               output_variables=output_variables)

def plant_model_from_dict(
    param_dict: Dict[str, Any]):
    model_type = ModelType(param_dict["model_type"])
    if model_type == ModelType.FACTORIZED_SCATTERED_INTERPOLATOR:
        return factorized_scattered_interp_from_dict(
            param_dict=param_dict)
    elif model_type == ModelType.SYMBOLIC:
        return symbolic_model_from_dict(
            param_dict=param_dict)
    elif model_type == ModelType.FLORIS:
        return floris_model_from_dict(
            param_dict=param_dict)
    else:
        raise NotImplementedError(f"Only factorized_scattered_interpolator, symbolic and"
                                  f" FLORIS models implemented.")
