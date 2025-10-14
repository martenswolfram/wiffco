from typing import Dict, Any, List
from abc import abstractmethod
from enum import Enum
import numpy as np
from twain_wifco.interface import (
    Component,
    ComponentParams,
    Ambient,
    Control,
    ModelOutput)
from twain_wifco.utils import (
    RadialBFInterpolatorParams,
    RadialBFInterpolator,
    rbf_interpolator_params_from_dict)

class PlantModel(Component):
    def __init__(self,
                 plant_name: str,
                 plant_params: ComponentParams):
        super().__init__(component_name=plant_name,
                         component_params=plant_params)
                
    def evaluate(self,
                 meteorological_condition: Dict[Ambient, float],
                 control_input: Dict[Control, float]):

        self._validate_inputs(inputs=(meteorological_condition.keys() | control_input.keys()))

        return self._evaluate(meteorological_condition=meteorological_condition,
                              control_input=control_input)

    @abstractmethod
    def _evaluate(self,
                  meteorological_condition: Dict[Ambient, float],
                  control_input: Dict[Control, float]):
        pass

class ModelType(Enum):
    FACTORIZED_RBF_INTERPOLATION = "factorized_rbf_interpolation"


class FactorizedRBFInterpParams(ComponentParams):
    def __init__(self,
                 control_interp_params: RadialBFInterpolatorParams,
                 ambient_interp_params: RadialBFInterpolatorParams):
        if set(control_interp_params.out_variables) != set(control_interp_params.out_variables):
            raise ValueError("Out variables of interpolation factors must be identical.")
        if set(control_interp_params.out_dims) != set(control_interp_params.out_dims):
            raise ValueError("Out variable dimensions of interpolation factors must be identical.")
        self.control_interp_params = control_interp_params
        self.ambient_interp_params = ambient_interp_params
        
    def input_variables(self):
        required_ambient = {in_var: in_dim for \
                            in_var, in_dim in zip(self.ambient_interp_params.in_variables,
                                                  self.ambient_interp_params.in_dims)}
        required_control = {in_var: in_dim for \
                            in_var, in_dim in zip(self.control_interp_params.in_variables,
                                                  self.control_interp_params.in_dims)}
        
        return required_ambient | required_control

    def output_variables(self):
        return {out_var: out_dim for \
                out_var, out_dim in zip(self.ambient_interp_params.out_variables,
                                        self.ambient_interp_params.out_dims)}

def factorized_rbf_interp_params_from_dict(
        param_dict: Dict[str, Dict | Any]):
    control_interp_params = rbf_interpolator_params_from_dict(
        param_dict=param_dict["control"],
        in_data_type=Control,
        out_data_type=ModelOutput)
    ambient_interp_params = rbf_interpolator_params_from_dict(
        param_dict=param_dict["ambient"],
        in_data_type=Ambient,
        out_data_type=ModelOutput)
    return FactorizedRBFInterpParams(
        control_interp_params=control_interp_params,
        ambient_interp_params=ambient_interp_params
    )

class FactorizedRBFInterp(PlantModel):
    def __init__(self,
                 plant_name: str,
                 plant_params: FactorizedRBFInterpParams):
        super().__init__(plant_name=plant_name,
                         plant_params=plant_params)
        
        # Initialize interpolation factors
        self.ambient_interp = RadialBFInterpolator(
            interpolator_params=plant_params.ambient_interp_params)
        self.control_interp = RadialBFInterpolator(
            interpolator_params=plant_params.control_interp_params)
        
    def _evaluate(self,
                  meteorological_condition: Dict[Ambient, np.ndarray],
                  control_input: Dict[Control, np.ndarray]):
        return self.ambient_interp.evaluate(query=meteorological_condition) \
            * self.control_interp.evaluate(query=control_input)
            
        
