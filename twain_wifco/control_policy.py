from typing import Dict, Any, List
from abc import abstractmethod
from enum import Enum
import numpy as np
from twain_wifco.interface import (
    Component,
    ComponentParams,
    Ambient,
    Control,
    get_abs_tol)
from twain_wifco.utils import (
    hstacked_from_dict,
    partition_into_dict) 

class ControlPolicy(Component):
    def __init__(self,
                 policy_name: str,
                 policy_params: ComponentParams):
        super().__init__(component_name=policy_name,
                         component_params=policy_params)
        
    def get_control_setpoints(self,
                              ambient_condition: Dict[Ambient, np.ndarray]) -> Dict[Control, np.ndarray]:
        
        self.validate_inputs(inputs=ambient_condition)
        
        return self._get_control_setpoints(ambient_condition=ambient_condition)

    @abstractmethod
    def _get_control_setpoints(self,
                               ambient_condition: Dict[Ambient, np.ndarray]) -> Dict[Control, np.ndarray]:
        pass

    @abstractmethod
    def get_ctrl_parameters(self,
                        ambient_condition: Dict[Ambient, np.ndarray]) -> np.ndarray:
        pass

    @abstractmethod
    def set_ctrl_parameters_from_vector(self,
                             x: np.ndarray,
                             ambient_condition: Dict[Ambient, np.ndarray]):
        pass

    @abstractmethod
    def get_ctrl_parameters_full(self) -> np.ndarray:
        pass

    @abstractmethod
    def set_ctrl_parameters_full(self, x: np.ndarray):
        pass

class ControlPolicyType(Enum):
    DISCRETE_POLICY = "discrete_policy"

class DiscreteControlPolicyParams(ComponentParams):
    def __init__(self,
                 ambient_support_points: Dict[Ambient, np.ndarray],
                 control_setpoints: Dict[Control, np.ndarray]):
        self.ambient_support_points = ambient_support_points
        self.ambient_variables = list(self.ambient_support_points.keys())
        self.control_setpoints = control_setpoints
        self.control_variables = list(self.control_setpoints.keys())
        
    def input_format(self):
        return {amb_var: self.ambient_support_points[amb_var].shape[1] for \
                            amb_var in self.ambient_support_points.keys()}

    def output_format(self):
        return {ctrl_var: self.control_setpoints[ctrl_var].shape[1] for \
                ctrl_var in self.control_setpoints.keys()}

def discrete_policy_params_from_dict(param_dict: Dict[str, Any]):
    ambient_support_points = {Ambient(amb_var): np.array(supp) for \
                              amb_var, supp in param_dict["ambient_support_points"].items()}
    control_setpoints = {Control(out_var): np.array(out_vals) for \
                         out_var, out_vals in param_dict["control_setpoints"].items()}

    return DiscreteControlPolicyParams(ambient_support_points=ambient_support_points,
                                       control_setpoints=control_setpoints)
        
class DiscreteControlPolicy(ControlPolicy):
    def __init__(self,
                 policy_name: str,
                 policy_params: DiscreteControlPolicyParams):
        super().__init__(policy_name=policy_name,
                         policy_params=policy_params)
        self.ambient_variables = policy_params.ambient_variables
        self.control_variables = policy_params.control_variables
        self.ambient_support_data = hstacked_from_dict(    
                policy_params.ambient_support_points,
                policy_params.ambient_variables)
        self.control_setpoint_data = hstacked_from_dict(    
                policy_params.control_setpoints,
                policy_params.control_variables)
        ctrl_dims = np.array(list(policy_params.control_setpoints[ctrl_var].shape[1] for \
                                 ctrl_var in self.control_variables))
        self.ctrl_partition_indices = np.cumsum(ctrl_dims[:-1])

        self.amb_abs_tols = list(get_abs_tol(amb_var) for \
                                 amb_var in self.ambient_variables)

        
    def find_ambient_index(self,
                           ambient_condition: Dict[Ambient, np.ndarray]) -> int:
        ambient_values = hstacked_from_dict(ambient_condition,
                                            self.ambient_variables)
        # Find correct support point
        mask = np.all(np.isclose(
            self.ambient_support_data,
            ambient_values,
            atol=self.amb_abs_tols),
            axis=1)
        row_index = np.where(mask)[0]
        if not len(row_index):
            raise ValueError("DiscreteControlPolicy: Ambient condition not found in support points.")
        return row_index[0]

    def get_ctrl_parameters(self,
                            ambient_condition: Dict[Ambient, np.ndarray]) -> np.ndarray:
        amb_index = self.find_ambient_index(ambient_condition=ambient_condition)
        return self.control_setpoint_data[amb_index]
        
    def _get_control_setpoints(self,
                               ambient_condition: Dict[Ambient, np.ndarray]) -> Dict[Control, np.ndarray]:
        setpoint_parameters = self.get_ctrl_parameters(ambient_condition=ambient_condition)
        return partition_into_dict(stacked_array=setpoint_parameters,
                                   variables=self.control_variables,
                                   partition_indices=self.ctrl_partition_indices)

    def set_ctrl_parameters_from_vector(self,
                                        x: np.ndarray,
                                        ambient_condition: Dict[Ambient, np.ndarray]):
        amb_index = self.find_ambient_index(ambient_condition=ambient_condition)
        self.control_setpoint_data[amb_index] = x

    def get_ctrl_parameters_full(self):
        # return flattened control setpoint data
        return np.ravel(self.control_setpoint_data)

    def set_ctrl_parameters_full(self, x: np.ndarray):
        # Set control setpoints
        self.control_setpoint_data = np.reshape(x, shape=self.control_setpoint_data.shape)
        

