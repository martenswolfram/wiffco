from typing import Dict, Any, List
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from twain_wifco.interface import (
    InterfaceInputs,
    InterfaceOutputs,
    AmbientVariable,
    ControlVariable,
    ComponentType,
    ComponentParams)

class ControlPolicy(ABC):
    def __init__(self,
                 params: ComponentParams):
        self.interface = params.interface()

    def get_control_setpoints(self,
                              ambient_condition: Dict[AmbientVariable, float]):
        
        self.interface.validate_inputs(ambient_condition=ambient_condition)
        
        return self._get_control_setpoints(ambient_condition=ambient_condition)

    @abstractmethod
    def _get_control_setpoints(self,
                               ambient_condition: Dict[AmbientVariable, float]):
        pass

class ControlPolicyType(Enum):
    DISCRETE_CONTROL_POLICY = "discrete_control_policy"

class DiscreteControlPolicyParams(ComponentParams):
    def __init__(self,
                 component_name: str,
                 ambient_variables: List[AmbientVariable],
                 ambient_conditions_support: np.ndarray,
                 control_inputs: List[ControlVariable],
                 control_setpoints: np.ndarray,
                 ambient_condition_tols: np.ndarray):
        super().__init__(component_type=ComponentType.CONTROL_POLICY,
                         component_name=component_name)
        self.ambient_variables = ambient_variables
        self.ambient_conditions_support = ambient_conditions_support
        self.control_inputs = control_inputs
        self.control_setpoints = control_setpoints
        self.ambient_condition_tols = ambient_condition_tols
        
    def _interface(self):
        inputs = InterfaceInputs(ambient_variables=set(self.ambient_variables))
        outputs = InterfaceOutputs(control_inputs=self.control_inputs)
        return inputs, outputs

def discrete_policy_params_from_dict(name: str,
                                     param_dict: Dict[str, Any]):
    ambient_variables = [AmbientVariable(ambient_var) for ambient_var in param_dict["ambient_variables"]]
    ambient_conditions_support = np.array(param_dict["ambient_conditions_support"])
    if ambient_conditions_support.shape[0] != len(ambient_variables):
        raise ValueError("DiscreteControlPolicyParams: Ambient conditions support data and ambient variables dimensions mismatch.")
    control_inputs = [ControlVariable(ctrl_var) for ctrl_var in param_dict["control_variables"]]
    control_setpoints = np.array(param_dict["control_setpoints"])
    
    ambient_condition_tols = param_dict.get("ambient_condition_tols", None)
    if ambient_condition_tols is None:
        # Determine tolerance depending on the range of ambient variable values
        ambient_condition_tols = np.ptp(ambient_conditions_support, axis=1) * 1e-5
    return DiscreteControlPolicyParams(component_name=name,
                                       ambient_variables=ambient_variables,
                                       ambient_conditions_support=ambient_conditions_support,
                                       control_inputs=control_inputs,
                                       control_setpoints=control_setpoints,
                                       ambient_condition_tols=ambient_condition_tols)
        
class DiscreteControlPolicy(ControlPolicy):
    def __init__(self,
                 params: DiscreteControlPolicyParams):
        super().__init__(params=params)
        self.params = params
    
    def _get_control_setpoints(self,
                               ambient_condition: Dict[AmbientVariable, float]):
        ambient_variables = np.array([ambient_condition[amb_var] for amb_var in self.params.ambient_variables])
        # Find correct support point
        mask = np.all(np.isclose(self.params.ambient_conditions_support,
                                 ambient_variables[:, None],
                                 atol=self.params.ambient_condition_tols[:, None]),
                                 axis=0)
        col_index = np.where(mask)[0]
        if not len(col_index):
            raise ValueError("DiscreteControlPolicy: Ambient condition not found in support points.")

        return {ctrl_var: ctrl_val for ctrl_var, ctrl_val in zip(self.params.control_inputs, self.params.control_setpoints[:, col_index[0]])}