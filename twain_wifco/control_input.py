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

class ControlPolicy(Component):
    def __init__(self,
                 policy_name: str,
                 policy_params: ComponentParams):
        super().__init__(component_name=policy_name,
                         component_params=policy_params)
        
    def get_control_setpoints(self,
                              ambient_condition: Dict[Ambient, float]) -> Dict[Control, float]:
        
        self._validate_inputs(inputs=ambient_condition.keys())
        
        return self._get_control_setpoints(ambient_condition=ambient_condition)

    @abstractmethod
    def _get_control_setpoints(self,
                               ambient_condition: Dict[Ambient, float]) -> Dict[Control, float]:
        pass

    @abstractmethod
    def get_ctrl_parameters(self,
                        ambient_condition: Dict[Ambient, float]) -> np.ndarray:
        pass

    @abstractmethod
    def set_ctrl_parameters_from_vector(self,
                             x: np.ndarray,
                             ambient_condition: Dict[Ambient, float]):
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
                 ambient_variables: List[Ambient],
                 ambient_conditions_support: np.ndarray,
                 control_inputs: List[Control],
                 control_setpoints: np.ndarray):
        self.ambient_variables = ambient_variables
        self.ambient_conditions_support = ambient_conditions_support
        self.control_variables = control_inputs
        self.control_setpoints = control_setpoints
        
    def input_variables(self):
        return set(self.ambient_variables)

    def output_variables(self):
        return set(self.control_variables)

def discrete_policy_params_from_dict(param_dict: Dict[str, Any]):
    ambient_variables = [Ambient(ambient_var) for ambient_var in param_dict["ambient_variables"]]
    ambient_conditions_support = np.array(param_dict["ambient_conditions_support"])
    if ambient_conditions_support.shape[1] != len(ambient_variables):
        raise ValueError("DiscreteControlPolicyParams: Ambient conditions support data and ambient variables dimensions mismatch.")
    control_inputs = [Control(ctrl_var) for ctrl_var in param_dict["control_variables"]]
    control_setpoints = np.array(param_dict["control_setpoints"])
    if control_setpoints.shape[1] != len(control_inputs):
        raise ValueError("DiscreteControlPolicyParams: Control setpoints data and control input dimensions mismatch.")
    if control_setpoints.shape[0] != ambient_conditions_support.shape[0]:
        raise ValueError("DiscreteControlPolicyParams: Control setpoints data and ambient conditions support dimensions mismatch.")
    
    return DiscreteControlPolicyParams(ambient_variables=ambient_variables,
                                       ambient_conditions_support=ambient_conditions_support,
                                       control_inputs=control_inputs,
                                       control_setpoints=control_setpoints)
        
class DiscreteControlPolicy(ControlPolicy):
    def __init__(self,
                 policy_name: str,
                 policy_params: DiscreteControlPolicyParams):
        super().__init__(policy_name=policy_name,
                         policy_params=policy_params)
        
        self.ambient_variables = policy_params.ambient_variables
        self.ambient_conditions_support = policy_params.ambient_conditions_support
        self.control_variables = policy_params.control_variables
        self.control_setpoints = policy_params.control_setpoints
        self.amb_abs_tols = list(get_abs_tol(amb_var) for amb_var in self.ambient_variables)
        
    def find_ambient_index(self,
                           ambient_condition: Dict[Ambient, float]) -> int:
        ambient_variables = np.array([ambient_condition[amb_var] for amb_var in self.ambient_variables])
        # Find correct support point
        mask = np.all(np.isclose(self.ambient_conditions_support,
                                 ambient_variables,
                                 atol=self.amb_abs_tols),
                                 axis=1)
        row_index = np.where(mask)[0]
        if not len(row_index):
            raise ValueError("DiscreteControlPolicy: Ambient condition not found in support points.")
        return row_index[0]

    def _get_control_setpoints(self,
                               ambient_condition: Dict[Ambient, float]) -> Dict[Control, float]:
        amb_index = self.find_ambient_index(ambient_condition=ambient_condition)
        return {ctrl_var: ctrl_val for ctrl_var, ctrl_val in \
                zip(self.control_variables, self.control_setpoints[amb_index, :])}

    def get_ctrl_parameters(self,
                        ambient_condition: Dict[Ambient, float]) -> np.ndarray:
        amb_index = self.find_ambient_index(ambient_condition=ambient_condition)
        return self.control_setpoints[amb_index, :]

    def set_ctrl_parameters_from_vector(self,
                             x: np.ndarray,
                             ambient_condition: Dict[Ambient, float]):
        amb_index = self.find_ambient_index(ambient_condition=ambient_condition)
        self.control_setpoints[amb_index, :] = x

    def get_ctrl_parameters_full(self):
        # return flattened control setpoints
        return np.ravel(self.control_setpoints)

    def set_ctrl_parameters_full(self, x: np.ndarray):
        # Set control setpoints
        self.control_setpoints = np.reshape(x, shape=self.control_setpoints.shape)
        

