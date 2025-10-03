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

        self._validate_inputs(inputs=(meteorological_condition.keys() | control_input.keys()))

        return self._evaluate(meteorological_condition=meteorological_condition,
                              control_input=control_input)

    @abstractmethod
    def _evaluate(self,
                  meteorological_condition: Dict[Ambient, float],
                  control_input: Dict[Control, float]):
        pass

class ModelType(Enum):
    INDEPENDENT_CUBIC_INTERPOLATION = "independent_cubic_interpolation"

class InterpolationMapping:
    def __init__(self,
                 control_data: Dict[Control, np.ndarray],
                 meteorological_data: Dict[Ambient, np.ndarray]):
        self.control_data = control_data
        self.meteorological_data = meteorological_data

class IndependentCubicInterpolationParams(ComponentParams):
    def __init__(self,
                 interpolation_mappings: Dict[ModelOutput, InterpolationMapping]):
        self.interpolation_mappings = interpolation_mappings

    def input_variables(self):
        required_ambient = set().union(*[mapping.meteorological_data.keys() for mapping in self.interpolation_mappings.values()])
        required_control = set().union(*[mapping.control_data.keys() for mapping in self.interpolation_mappings.values()])        
        return set().union(required_ambient, required_control)

    def output_variables(self):
        return set(self.interpolation_mappings.keys())

def independent_cubic_interp_params_from_dict(param_dict: Dict[str, Any | Dict]):
    interpolation_mappings = {}
    for out_var, interpolation_mapping in param_dict["interpolation_mappings"].items():
        interpolation_mapping: Dict[str, Dict]
        ctrl_data = {}
        for ctrl_var, data in interpolation_mapping["control_data"].items():
            ctrl_var: str
            data: List[List[float]]
            ctrl_data[Control(ctrl_var)] = np.array(data)
        met_data = {}
        for met_var, data in interpolation_mapping["meteorological_data"].items():
            met_var: str
            data: List[List[float]]
            met_data[Ambient(met_var)] = np.array(data)
        
        out_var: str
        interpolation_mappings[ModelOutput(out_var)] = InterpolationMapping(
              meteorological_data=met_data,
              control_data=ctrl_data)
    return IndependentCubicInterpolationParams(
        interpolation_mappings=interpolation_mappings)
       
class CubicSplineWrapper:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.spline = CubicSpline(x=x, y=y, extrapolate=False)

    def evaluate(self, x):
        out = self.spline(x)
        if np.isnan(out):
            raise ValueError("CubicSplineWrapper: Extrapolation not implemented.")
        return out

class ScalarCubicInterpolation:
    def __init__(self,
                 interpolation_mapping: InterpolationMapping):
        self.control_models = \
            {ctrl_var: CubicSplineWrapper(data[0], data[1]) for ctrl_var, data in interpolation_mapping.control_data.items()}
        self.meteorological_models = \
            {met_var: CubicSplineWrapper(data[0], data[1]) for met_var, data in interpolation_mapping.meteorological_data.items()}

    def evaluate(self,
                 meteorological_condition: Dict[Ambient, float],
                 control: Dict[Control, float]):
        out_value = 1
        for met_var, met_model in self.meteorological_models.items():
            out_value *= met_model.evaluate(meteorological_condition[met_var])
        for ctrl_var, ctrl_model in self.control_models.items():
            out_value *= ctrl_model.evaluate(control[ctrl_var])
        return out_value

class IndependentCubicInterpolation(PlantModel):
    def __init__(self,
                 plant_name: str,
                 plant_params: IndependentCubicInterpolationParams):
        super().__init__(plant_name=plant_name,
                         plant_params=plant_params)

        # Initialize models
        self.scalar_output_models: Dict[ModelOutput, ScalarCubicInterpolation] = {}
        for out, interpolation_mapping in plant_params.interpolation_mappings.items():
            self.scalar_output_models[out] = \
                ScalarCubicInterpolation(interpolation_mapping=interpolation_mapping)
                        
    def _evaluate(self,
                  meteorological_condition: Dict[Ambient, float],
                  control_input: Dict[Control, float]):
        
        model_outputs = {}
        for out, model in self.scalar_output_models.items():
            model_outputs[out] = model.evaluate(meteorological_condition=meteorological_condition,
                                                control=control_input)
        return model_outputs
    