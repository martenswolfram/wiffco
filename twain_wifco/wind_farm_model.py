from typing import Dict, Any
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from twain_wifco.interface import (
    Interface,
    AmbientVariable,
    ControlVariable,
    OutputVariable,
    Component,
    ComponentParams)
from scipy.interpolate import CubicSpline

class WindFarmModel(ABC):
    def __init__(self,
                 params: ComponentParams):
        self.interface = params.interface()

    def evaluate(self,
                 meteorological_condition: Dict[AmbientVariable, float],
                 control_input: Dict[ControlVariable, float]):

        self.interface.validate_inputs(ambient_condition=meteorological_condition,
                                       control_input=control_input)

        return self._evaluate(meteorological_condition=meteorological_condition,
                              control_input=control_input)

    @abstractmethod
    def _evaluate(self,
                  meteorological_condition: Dict[AmbientVariable, float],
                  control_input: Dict[ControlVariable, float]):
        pass

class ModelType(Enum):
    INDEPENDENT_CUBIC_INTERPOLATOR = "independent_cubic_interpolator"

class IndependentCubicInterpolatorParams(ComponentParams):
    def __init__(self,
                 name: str,
                 param_dict: Dict[str, Dict | Any]):
        super().__init__(name=name)
        self.ctrl_input_data = {}
        for ctrl_var, data in param_dict["control_input_data"].items():
            self.ctrl_input_data[ControlVariable(ctrl_var)] = np.array(data)
        self.met_condition_data = {}
        for met_var, data in param_dict["meteorological_condition_data"].items():
            self.met_condition_data[AmbientVariable(met_var)] = np.array(data)
        self.single_output = OutputVariable(param_dict["single_output"])

    def interface(self):
        return Interface(component=Component.WIND_FARM_MODEL,
                         name=self.name,
                         ambient_variables=self.met_condition_data.keys(),
                         control_inputs=self.ctrl_input_data.keys())
        
class IndependentCubicInterpolator(WindFarmModel):
    def __init__(self,
                 params: IndependentCubicInterpolatorParams):
        super().__init__(params=params)

        # Initialize models
        self.control_input_models = \
            {ctrl_var: CubicSpline(data[0], data[1], extrapolate=False) for ctrl_var, data in params.ctrl_input_data.items()}
        self.meteorological_condition_models = \
            {met_var: CubicSpline(data[0], data[1], extrapolate=False) for met_var, data in params.met_condition_data.items()}
        self.single_output = params.single_output
            
    def _evaluate(self,
                 meteorological_condition: Dict[AmbientVariable, float],
                 control_input: Dict[ControlVariable, float]):
        
        out_value = 1
        for met_var, met_value in meteorological_condition.items():
            out_value *= self.meteorological_condition_models[met_var](met_value)
        for ctrl_var, ctrl_value in control_input.items():
            out_value *= self.control_input_models[ctrl_var](ctrl_value)

        return {self.single_output: out_value}