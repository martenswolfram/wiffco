from typing import Dict, Any, List
from abc import abstractmethod
from enum import Enum
import numpy as np
from twain_wifco.interface import (
    Component,
    ComponentParams,
    Ambient,
    Control)

class ControlPolicy(Component):
    def __init__(self,
                 policy_name: str,
                 policy_params: ComponentParams):
        super().__init__(component_name=policy_name,
                         component_params=policy_params)
        
    def get_control_setpoints(self,
                              ambient_condition: Dict[Ambient, float]):
        
        self._validate_inputs(inputs=ambient_condition)
        
        return self._get_control_setpoints(ambient_condition=ambient_condition)

    @abstractmethod
    def _get_control_setpoints(self,
                               ambient_condition: Dict[Ambient, float]):
        pass

class ControlPolicyType(Enum):
    DISCRETE_POLICY = "discrete_policy"

class DiscreteControlPolicyParams(ComponentParams):
    def __init__(self,
                 ambient_variables: List[Ambient],
                 ambient_conditions_support: np.ndarray,
                 control_inputs: List[Control],
                 control_setpoints: np.ndarray,
                 ambient_condition_tols: np.ndarray):
        self.ambient_variables = ambient_variables
        self.ambient_conditions_support = ambient_conditions_support
        self.control_inputs = control_inputs
        self.control_setpoints = control_setpoints
        self.ambient_condition_tols = ambient_condition_tols

    def input_variables(self):
        return set(self.ambient_variables)

    def output_variables(self):
        return set(self.control_inputs)

def discrete_policy_params_from_dict(param_dict: Dict[str, Any]):
    ambient_variables = [Ambient(ambient_var) for ambient_var in param_dict["ambient_variables"]]
    ambient_conditions_support = np.array(param_dict["ambient_conditions_support"])
    if ambient_conditions_support.shape[0] != len(ambient_variables):
        raise ValueError("DiscreteControlPolicyParams: Ambient conditions support data and ambient variables dimensions mismatch.")
    control_inputs = [Control(ctrl_var) for ctrl_var in param_dict["control_variables"]]
    control_setpoints = np.array(param_dict["control_setpoints"])
    
    ambient_condition_tols = param_dict.get("ambient_condition_tols", None)
    if ambient_condition_tols is None:
        # Determine tolerance depending on the range of ambient variable values
        ambient_condition_tols = np.ptp(ambient_conditions_support, axis=1) * 1e-5
    return DiscreteControlPolicyParams(ambient_variables=ambient_variables,
                                       ambient_conditions_support=ambient_conditions_support,
                                       control_inputs=control_inputs,
                                       control_setpoints=control_setpoints,
                                       ambient_condition_tols=ambient_condition_tols)
        
class DiscreteControlPolicy(ControlPolicy):
    def __init__(self,
                 policy_name: str,
                 policy_params: DiscreteControlPolicyParams):
        super().__init__(policy_name=policy_name,
                         policy_params=policy_params)
        self.ambient_variables = policy_params.ambient_variables
        self.ambient_conditions_support = policy_params.ambient_conditions_support
        self.control_inputs = policy_params.control_inputs
        self.control_setpoints = policy_params.control_setpoints
        self.ambient_condition_tols = policy_params.ambient_condition_tols
        
    def _get_control_setpoints(self,
                               ambient_condition: Dict[Ambient, float]):
        ambient_variables = np.array([ambient_condition[amb_var] for amb_var in self.ambient_variables])
        # Find correct support point
        mask = np.all(np.isclose(self.ambient_conditions_support,
                                 ambient_variables[:, None],
                                 atol=self.ambient_condition_tols[:, None]),
                                 axis=0)
        col_index = np.where(mask)[0]
        if not len(col_index):
            raise ValueError("DiscreteControlPolicy: Ambient condition not found in support points.")

        return {ctrl_var: ctrl_val for ctrl_var, ctrl_val in zip(self.control_inputs, self.control_setpoints[:, col_index[0]])}