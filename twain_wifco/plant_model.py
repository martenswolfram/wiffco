from typing import Dict, Any, List
from abc import abstractmethod
from enum import Enum
import numpy as np
from twain_wifco.interface import (
    DataPoint,
    Component,
    ComponentParams,
    Ambient,
    Control,
    ModelOutput,
    Interface)
from twain_wifco.utils import (
    ScatteredInterpolatorParams,
    ScatteredInterpolator,
    scattered_interpolator_params_from_dict)

class PlantModel(Component):
    def __init__(self,
                 plant_name: str,
                 plant_params: ComponentParams):
        super().__init__(component_name=plant_name,
                         component_params=plant_params)
                
    def evaluate(self,
                 meteorological_condition: DataPoint[Ambient],
                 control_input: DataPoint[Control]):

        self.validate_inputs(input_data={Ambient: meteorological_condition,
                                         Control: control_input})

        return self._evaluate(meteorological_condition=meteorological_condition,
                              control_input=control_input)

    @abstractmethod
    def _evaluate(self,
                  meteorological_condition: Dict[Ambient, np.ndarray],
                  control_input: Dict[Control, np.ndarray]):
        pass

class ModelType(Enum):
    FACTORIZED_SCATTERED_INTERPOLATOR = "factorized_scattered_interpolator"


class FactorizedScatteredInterpParams(ComponentParams):
    def __init__(self,
                 control_interp_params: ScatteredInterpolatorParams,
                 ambient_interp_params: ScatteredInterpolatorParams):
        if control_interp_params.out_data.shapes() != control_interp_params.out_data.shapes():
            raise ValueError("Out variables shapes of interpolation factors must be identical.")
        self.control_interp_params = control_interp_params
        self.ambient_interp_params = ambient_interp_params
        
    def input_interface(self) -> Interface:
        return Interface(all_data_type_shapes={Ambient: self.ambient_interp_params.support_data.shapes(),
                                               Control: self.control_interp_params.support_data.shapes()})

    def output_interface(self) -> Interface:
        return Interface(all_data_type_shapes={ModelOutput: self.ambient_interp_params.out_data.shapes()})
    
def factorized_scattered_interp_params_from_dict(
        param_dict: Dict[str, Dict | Any]):
    control_interp_params = scattered_interpolator_params_from_dict(
        param_dict=param_dict["control"],
        support_data_type=Control,
        out_data_type=ModelOutput)
    ambient_interp_params = scattered_interpolator_params_from_dict(
        param_dict=param_dict["ambient"],
        support_data_type=Ambient,
        out_data_type=ModelOutput)
    return FactorizedScatteredInterpParams(
        control_interp_params=control_interp_params,
        ambient_interp_params=ambient_interp_params
    )

class FactorizedScatteredInterp(PlantModel):
    def __init__(self,
                 plant_name: str,
                 plant_params: FactorizedScatteredInterpParams):
        super().__init__(plant_name=plant_name,
                         plant_params=plant_params)
        
        # Initialize interpolation factors
        self.ambient_interp = ScatteredInterpolator(
            interpolator_params=plant_params.ambient_interp_params)
        self.control_interp = ScatteredInterpolator(
            interpolator_params=plant_params.control_interp_params)
        
    def _evaluate(self,
                  meteorological_condition: Dict[Ambient, np.ndarray],
                  control_input: Dict[Control, np.ndarray]):
        ambient_eval = self.ambient_interp.evaluate(query=meteorological_condition)
        control_eval = self.control_interp.evaluate(query=control_input)
        return DataPoint({out_var: ambient_eval[out_var] * control_eval[out_var] for \
                          out_var in self.ambient_interp.out_data_point.keys()})
            
        
