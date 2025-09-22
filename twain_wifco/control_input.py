from typing import Dict, Any
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from twain_wifco.interface import (
    Interface,
    AmbientVariable,
    ControlVariable,
    Component,
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
                 name: str,
                 param_dict: Dict[str, Dict | Any]):
        super().__init__(name=name)
        self.ambient_variables = self.ambient_variables = \
            [AmbientVariable(ambient_var) for ambient_var in param_dict["ambient_variables"]]
        self.ambient_conditions_support = np.array(param_dict["ambient_conditions_support"])
        if self.ambient_conditions_support.shape[0] != len(self.ambient_variables):
            raise ValueError("DiscreteControlPolicyParams: Ambient conditions support data and ambient variables dimensions mismatch.")
        self.control_inputs = \
            [ControlVariable(ctrl_var) for ctrl_var in param_dict["control_variables"]]
        self.ambient_condition_tols = param_dict.get("ambient_condition_tols", None)
        if self.ambient_condition_tols is None:
            # Determine tolerance depending on the range of ambient variable values
            self.ambient_condition_tols = np.ptp(self.ambient_conditions_support, axis=1) * 1e-5

    def interface(self):
        return Interface(component=Component.CONTROL_POLICY,
                         name=self.name,
                         ambient_variables=set(self.ambient_variables))
    
class DiscreteControlPolicy(ControlPolicy):
    def __init__(self,
                 params: DiscreteControlPolicyParams):
        super().__init__(params=params)
        self.params = params
        self.control_setpoints = None

    def set_control_policy(self, control_setpoints: np.ndarray):
        self.control_setpoints = control_setpoints
        if self.control_setpoints.shape[0] != len(self.params.control_inputs) or \
            self.control_setpoints.shape[1] != self.params.ambient_conditions_support.shape[1]:
            raise ValueError("DiscreteControlPolicy: Control setpoints dimensions mismatch.")
    
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

        return self.control_setpoints[:, col_index[0]]