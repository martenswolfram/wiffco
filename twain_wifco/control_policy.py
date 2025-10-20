from typing import Dict, Any
from abc import abstractmethod
from enum import Enum
import numpy as np
from twain_wifco.interface import (
    DataPoint,
    Component,
    ComponentParams,
    Ambient,
    Control,
    get_abs_tol,
    Interface)
from twain_wifco.utils import (
    ScatteredInterpolatorParams,
    ScatteredInterpolator,
    scattered_interpolator_params_from_dict)

class ControlPolicy(Component):
    def __init__(self,
                 policy_name: str,
                 policy_params: ComponentParams):
        super().__init__(component_name=policy_name,
                         component_params=policy_params)
        
    def get_control_setpoints(self,
                              ambient_condition: DataPoint[Ambient]) -> DataPoint[Control]:
        
        self.validate_inputs(input_data={Ambient: ambient_condition})
        
        return self._get_control_setpoints(ambient_condition=ambient_condition)

    @abstractmethod
    def _get_control_setpoints(self,
                               ambient_condition: DataPoint[Ambient]) -> DataPoint[Control]:
        pass

    @abstractmethod
    def get_ctrl_parameters(self,
                            ambient_condition: DataPoint[Ambient]) -> np.ndarray:
        pass

    @abstractmethod
    def set_ctrl_parameters_from_vector(self,
                                        x: np.ndarray,
                                        ambient_condition: DataPoint[Ambient]):
        pass

    @abstractmethod
    def get_ctrl_parameters_full(self) -> np.ndarray:
        pass

    @abstractmethod
    def set_ctrl_parameters_full(self, x: np.ndarray) -> None:
        pass

class ControlPolicyType(Enum):
    SCATTERED_INTERP_POLICY = "scattered_interp"

class ScatteredInterpPolicyParams(ComponentParams):
    def __init__(self,
                 ambient_interp_params: ScatteredInterpolatorParams):
        self.ambient_interp_params = ambient_interp_params
        
    def input_interface(self) -> Interface:
        return Interface(all_data_type_shapes={
            Ambient: self.ambient_interp_params.support_data.shapes()})

    def output_interface(self) -> Interface:
        return Interface(all_data_type_shapes={
            Control: self.ambient_interp_params.out_data.shapes()})
    
def scattered_interp_policy_params_from_dict(
        param_dict: Dict[str, Dict | Any]):
    ambient_interp_params = scattered_interpolator_params_from_dict(
        param_dict=param_dict,
        support_data_type=Ambient,
        out_data_type=Control)
    return ScatteredInterpPolicyParams(
        ambient_interp_params=ambient_interp_params
    )

class ScatteredInterpPolicy(ControlPolicy):
    def __init__(self,
                 policy_name: str,
                 policy_params: ScatteredInterpPolicyParams):
        super().__init__(policy_name=policy_name,
                         policy_params=policy_params)
        
        # Initialize interpolation
        self.ambient_interp = ScatteredInterpolator(
            interpolator_params=policy_params.ambient_interp_params)

        self.amb_abs_tols = list(get_abs_tol(amb_var) for \
                                 amb_var in self.ambient_interp.support_data.order)
                
    def find_ambient_index(self,
                           ambient_condition: DataPoint[Ambient]) -> int:
        ambient_values = ambient_condition.to_vector()
        # Find correct support point
        mask = np.all(np.isclose(
            self.ambient_interp.support_data.to_matrix(),
            ambient_values,
            atol=self.amb_abs_tols),
            axis=1)
        row_index = np.where(mask)[0]
        if not len(row_index):
            raise ValueError("DiscreteControlPolicy: Ambient condition not found in support points.")
        return row_index[0]
        
    def get_ctrl_parameters(self,
                            ambient_condition: DataPoint[Ambient]) -> np.ndarray:
        return self.ambient_interp.evaluate_to_vector(query=ambient_condition)
        
    def _get_control_setpoints(self,
                               ambient_condition: DataPoint[Ambient]) -> DataPoint[Control]:
        return self.ambient_interp.evaluate(query=ambient_condition)
    
    def set_ctrl_parameters_from_vector(self,
                                        x: np.ndarray,
                                        ambient_condition: DataPoint[Ambient]):
        amb_index = self.find_ambient_index(ambient_condition=ambient_condition)
        self.ambient_interp.out_data[amb_index] = x

    def get_ctrl_parameters_full(self):
        # return flattened control setpoint data
        return np.ravel(self.ambient_interp.out_data)

    def set_ctrl_parameters_full(self, x: np.ndarray):
        # Set control setpoints
        self.ambient_interp.out_data = np.reshape(x, shape=self.ambient_interp.out_data.shape)
        
