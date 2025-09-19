from typing import Dict, Any
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from twain_wiffco.ambient_conditions import AmbientVariable
from twain_wiffco.control_input import ControlVariable
from twain_wiffco.model_output import OutputVariable
from scipy.interpolate import CubicSpline

class ModelType(Enum):
    INDEPENDENT_CUBIC_INTERPOLATOR = "independent_cubic_interpolator"

class WindFarmModel(ABC):
    def __init__(self, name: str):
        self.name = name

    def evaluate(self,
                 meteorological_condition: Dict[AmbientVariable, float],
                 control_input: Dict[ControlVariable, float],
                 safe_eval=True):

        if safe_eval:
            self._validate_inputs(meteorological_condition=meteorological_condition,
                                                  control_input=control_input)

        return self._evaluate(meteorological_condition=meteorological_condition,
                                              control_input=control_input)

    @abstractmethod
    def _validate_inputs(self,
                        meteorological_condition: Dict[AmbientVariable, float],
                        control_input: Dict[ControlVariable, float]):
        pass

    @abstractmethod
    def _evaluate(self,
                  meteorological_condition: Dict[AmbientVariable, float],
                  control_input: Dict[ControlVariable, float]):
        pass

class IndependentCubicInterpolatorParams:
    def __init__(self, param_dict: Dict[str, Any]):
        self.ctrl_input_data = {}
        for ctrl_var, data in param_dict["control_input_data"].items():
            self.ctrl_input_data[ControlVariable(ctrl_var)] = np.array(data)
        self.met_condition_data = {}
        for met_var, data in param_dict["meteorological_condition_data"].items():
            self.met_condition_data[AmbientVariable(met_var)] = np.array(data)
        self.single_output = OutputVariable(param_dict["single_output"])
        
class IndependentCubicInterpolator(WindFarmModel):
    def __init__(self,
                 name: str,
                 params: IndependentCubicInterpolatorParams):
        super().__init__(name=name)

        # Initialize models
        self.control_input_models = \
            {ctrl_var: CubicSpline(data[0], data[1], extrapolate=False) for ctrl_var, data in params.ctrl_input_data.items()}
        self.meteorological_condition_models = \
            {met_var: CubicSpline(data[0], data[1], extrapolate=False) for met_var, data in params.met_condition_data.items()}
        self.single_output = params.single_output

    def _validate_inputs(self,
                         meteorological_condition: Dict[AmbientVariable, float],
                         control_input: Dict[ControlVariable, float]):
        if not (self.meteorological_condition_models.keys() <= meteorological_condition.keys()):
            raise ValueError("Insufficient meteorological condition as input for model '{}'.".format(self.name))
        if not (self.control_input_models.keys() <= control_input.keys()):
            raise ValueError("Insufficient control input for wind farm model '{}'.".format(self.name))
            
    def _evaluate(self,
                 meteorological_condition: Dict[AmbientVariable, float],
                 control_input: Dict[ControlVariable, float]):
        
        out_value = 1
        for met_var, met_value in meteorological_condition.items():
            out_value *= self.meteorological_condition_models[met_var](met_value)
        for ctrl_var, ctrl_value in control_input.items():
            out_value *= self.control_input_models[ctrl_var](ctrl_value)

        return {self.single_output: out_value}