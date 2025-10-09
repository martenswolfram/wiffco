from typing import Dict, Any, List
from abc import abstractmethod
from enum import Enum
import numpy as np
from twain_wifco.interface import (
    Component,
    ComponentParams,
    Ambient,
    Control,
    argsort_enum_list)

class ControlPolicy(Component):
    def __init__(self,
                 policy_name: str,
                 policy_params: ComponentParams):
        super().__init__(component_name=policy_name,
                         component_params=policy_params)
        
    def get_control_setpoints(self,
                              ambient_condition: Dict[Ambient, float]):
        
        self._validate_inputs(inputs=ambient_condition.keys())
        
        return self._get_control_setpoints(ambient_condition=ambient_condition)

    @abstractmethod
    def _get_control_setpoints(self,
                               ambient_condition: Dict[Ambient, float]):
        pass

    @abstractmethod
    def get_x_vector(self) -> np.ndarray:
        pass

    @abstractmethod
    def set_from_x_vector(self, x: np.ndarray):
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
        self.control_inputs = control_inputs
        self.control_setpoints = control_setpoints
        
    def input_variables(self):
        return set(self.ambient_variables)

    def output_variables(self):
        return set(self.control_inputs)

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
        
        # Lexicographic sort to allow comparison
        amb_sort = argsort_enum_list(policy_params.ambient_variables)
        self.ambient_variables = [policy_params.ambient_variables[ind] for ind in amb_sort]
        lexsort_index = np.lexsort(policy_params.ambient_conditions_support[:, amb_sort].T[::-1])
        self.ambient_conditions_support = policy_params.ambient_conditions_support[lexsort_index][:, amb_sort]
        ctrl_sort = argsort_enum_list(policy_params.control_inputs)
        self.control_inputs = [policy_params.control_inputs[ind] for ind in ctrl_sort] 
        self.control_setpoints = policy_params.control_setpoints[lexsort_index][:, ctrl_sort]
        
    def _get_control_setpoints(self,
                               ambient_condition: Dict[Ambient, float]):
        ambient_variables = np.array([ambient_condition[amb_var] for amb_var in self.ambient_variables])
        # Find correct support point
        mask = np.all(np.isclose(self.ambient_conditions_support,
                                 ambient_variables),
                                 axis=1)
        row_index = np.where(mask)[0]
        if not len(row_index):
            raise ValueError("DiscreteControlPolicy: Ambient condition not found in support points.")

        return {ctrl_var: ctrl_val for ctrl_var, ctrl_val in zip(self.control_inputs, self.control_setpoints[row_index[0], :])}
    
    def _equals_specific(self, other: "DiscreteControlPolicy") -> bool:
        if not set(self.control_inputs) == set(other.control_inputs):
            return False
        perm_other_controls = [other.control_inputs.index(label) for label in self.control_inputs]
        if not set(self.ambient_variables) == set(other.ambient_variables):
            return False
        perm_other_ambient = [other.ambient_variables.index(label) for label in self.ambient_variables]
        return np.array_equal(self.control_setpoints, other.control_setpoints[:, perm_other_controls]) and \
            np.array_equal(self.ambient_conditions_support, other.ambient_conditions_support[:, perm_other_ambient])

    def get_x_vector(self):
        # return flattened control setpoints
        return np.ravel(self.control_setpoints)

    def set_from_x_vector(self, x: np.ndarray):
        # Set control setpoints
        self.control_setpoints = np.reshape(x, shape=self.control_setpoints.shape)
        

