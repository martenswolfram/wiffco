from typing import Dict, Any
from abc import abstractmethod
from enum import Enum
import numpy as np
from twain_wifco.interface import (
    Component,
    ComponentParams,
    Ambient,
    Control,
    ModelOutput)
from scipy.interpolate import CubicSpline

class PlantModel(Component):
    def __init__(self,
                 plant_name: str,
                 plant_params: ComponentParams):
        super().__init__(component_name=plant_name,
                         component_params=plant_params)
        
    def evaluate(self,
                 meteorological_condition: Dict[Ambient, float],
                 control_input: Dict[Control, float]):

        self._validate_inputs(inputs=(meteorological_condition | control_input))

        return self._evaluate(meteorological_condition=meteorological_condition,
                              control_input=control_input)

    @abstractmethod
    def _evaluate(self,
                  meteorological_condition: Dict[Ambient, float],
                  control_input: Dict[Control, float]):
        pass

class ModelType(Enum):
    INDEPENDENT_CUBIC_INTERPOLATOR = "independent_cubic_interpolator"

class IndependentCubicInterpolatorParams(ComponentParams):
    def __init__(self,
                 control_input_data: Dict[Control, np.ndarray],
                 meteorological_condition_data: Dict[Ambient, np.ndarray],
                 single_output: ModelOutput):
        self.control_input_data = control_input_data
        self.meteorological_condition_data = meteorological_condition_data
        self.single_output = single_output

    def input_variables(self):
        return set(self.control_input_data.keys() | self.meteorological_condition_data.keys())

    def output_variables(self):
        return set([self.single_output])

def independent_cubic_interp_params_from_dict(param_dict: Dict[str, Any | Dict]):
    ctrl_input_data = {}
    for ctrl_var, data in param_dict["control_input_data"].items():
        ctrl_input_data[Control(ctrl_var)] = np.array(data)
    met_condition_data = {}
    for met_var, data in param_dict["meteorological_condition_data"].items():
        met_condition_data[Ambient(met_var)] = np.array(data)
    single_output = ModelOutput(param_dict["single_output"])
    return IndependentCubicInterpolatorParams(control_input_data=ctrl_input_data,
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

class IndependentCubicInterpolator(PlantModel):
    def __init__(self,
                 plant_name: str,
                 plant_params: IndependentCubicInterpolatorParams):
        super().__init__(plant_name=plant_name,
                         plant_params=plant_params)

        # Initialize models
        self.control_input_models = \
            {ctrl_var: CubicSplineWrapper(data[0], data[1]) for ctrl_var, data in plant_params.control_input_data.items()}
        self.meteorological_condition_models = \
            {met_var: CubicSplineWrapper(data[0], data[1]) for met_var, data in plant_params.meteorological_condition_data.items()}
        self.single_output = plant_params.single_output
                        
    def _evaluate(self,
                  meteorological_condition: Dict[Ambient, float],
                  control_input: Dict[Control, float]):
        
        out_value = 1
        for met_var, met_model in self.meteorological_condition_models.items():
            out_value *= met_model.evaluate(meteorological_condition[met_var])
        for ctrl_var, ctrl_model in self.control_input_models.items():
            out_value *= ctrl_model.evaluate(control_input[ctrl_var])

        return {self.single_output: out_value}
    