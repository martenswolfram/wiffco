from typing import Dict, Any
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from twain_wifco.interface import (
    InterfaceVariables,
    AmbientVariable,
    ControlVariable,
    OutputVariable,
    ComponentType,
    ComponentParams)
from scipy.interpolate import CubicSpline

class WindFarmModel(ABC):
    def __init__(self,
                 params: ComponentParams):
        self.interface = params.interface()

    def evaluate(self,
                 meteorological_condition: Dict[AmbientVariable, float],
                 control_input: Dict[ControlVariable, float]):

        self.interface.validate_inputs(ambient_condition=meteorological_condition.keys(),
                                       control_input=control_input.keys())

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
                 component_name: str,
                 control_input_data: Dict[ControlVariable, np.ndarray],
                 meteorological_condition_data: Dict[AmbientVariable, np.ndarray],
                 single_output: OutputVariable):
        super().__init__(component_type=ComponentType.WIND_FARM_MODEL,
                         component_name=component_name)
        self.control_input_data = control_input_data
        self.meteorological_condition_data = meteorological_condition_data
        self.single_output = single_output

    def _interface(self):
        inputs=InterfaceVariables(ambient_variables=self.meteorological_condition_data.keys(),
                               control_inputs=self.control_input_data.keys())
        outputs=InterfaceVariables(output_variables=set([self.single_output]))
        return inputs, outputs

def independent_cubic_interp_params_from_dict(name: str,
                                              param_dict: Dict[str, Any]):
    ctrl_input_data = {}
    for ctrl_var, data in param_dict["control_input_data"].items():
        ctrl_input_data[ControlVariable(ctrl_var)] = np.array(data)
    met_condition_data = {}
    for met_var, data in param_dict["meteorological_condition_data"].items():
        met_condition_data[AmbientVariable(met_var)] = np.array(data)
    single_output = OutputVariable(param_dict["single_output"])
    return IndependentCubicInterpolatorParams(component_name=name,
                                              control_input_data=ctrl_input_data,
                                              meteorological_condition_data=met_condition_data,
                                              single_output=single_output)
       
class CubicSplineWrapper:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.spline = CubicSpline(x=x, y=y, extrapolate=False)

    def evaluate(self, x):
        out = self.spline(x)
        if np.isnan(out):
            raise ValueError("CubicSplineWrapper: Extrapolation not implemented.")
        return out

class IndependentCubicInterpolator(WindFarmModel):
    def __init__(self,
                 params: IndependentCubicInterpolatorParams):
        super().__init__(params=params)

        # Initialize models
        self.control_input_models = \
            {ctrl_var: CubicSplineWrapper(data[0], data[1]) for ctrl_var, data in params.control_input_data.items()}
        self.meteorological_condition_models = \
            {met_var: CubicSplineWrapper(data[0], data[1]) for met_var, data in params.meteorological_condition_data.items()}
        self.single_output = params.single_output
            
    def _evaluate(self,
                 meteorological_condition: Dict[AmbientVariable, float],
                 control_input: Dict[ControlVariable, float]):
        
        out_value = 1
        for met_var, met_model in self.meteorological_condition_models.items():
            out_value *= met_model.evaluate(meteorological_condition[met_var])
        for ctrl_var, ctrl_model in self.control_input_models.items():
            out_value *= ctrl_model.evaluate(control_input[ctrl_var])

        return {self.single_output: out_value}