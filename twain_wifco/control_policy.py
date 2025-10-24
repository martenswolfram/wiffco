from typing import Dict, Any
import numpy as np
from twain_wifco.interface import (
    DataPoint,
    DataTable,
    Component,
    ComponentParams,
    Ambient,
    Control,
    Interface)

class DiscreteControlPolicyParams(ComponentParams):
    def __init__(self,
                 ambient_support_data: DataTable[Ambient],
                 control_out_data: DataTable[Control]):
        self.ambient_support_data = ambient_support_data
        self.control_out_data = control_out_data
        
    def input_interface(self) -> Interface:
        return Interface(all_shapes={
            Ambient: self.ambient_support_data.shapes()})

    def output_interface(self) -> Interface:
        return Interface(all_shapes={
            Control: self.control_out_data.shapes()})

def discrete_control_policy_params_from_dict(
        param_dict: Dict[str, Dict | Any]):

    ambient_support_data = DataTable(
        {Ambient(amb_var): np.array(data) for \
         amb_var, data in param_dict["ambient_support_data"].items()})
    control_out_data = DataTable(
        {Control(ctrl_var): np.array(data) for \
         ctrl_var, data in param_dict["control_out_data"].items()})

    return DiscreteControlPolicyParams(
        ambient_support_data=ambient_support_data,
        control_out_data=control_out_data
    )

class DiscreteControlPolicy(Component):
    def __init__(self,
                 policy_name: str,
                 policy_params: DiscreteControlPolicyParams):
        super().__init__(component_name=policy_name,
                         component_params=policy_params)
        self.ambient_support_data = policy_params.ambient_support_data
        self.control_out_data = policy_params.control_out_data

    def get_control_setpoints(self,
                              ambient_condition: DataPoint[Ambient]) -> DataPoint[Control]:
        
        self.validate_inputs(input_data={Ambient: ambient_condition})

        point_index = self.ambient_support_data.find_matching_point(
            data_point=ambient_condition)
        return self.control_out_data.get_point(point_index)

    def set_control_data(self,
                         control_data_vector: np.array):
        self.control_out_data.update_from_vector(control_data_vector)
        
    def random_perturbation(self,
                            scale: float = 1):
        perturbed_ctrl_out_data = {}
        for ctrl_var, data in self.control_out_data.items():
            magnitude = scale * self.control_out_data.abs_tol(ctrl_var)
            perturbed_ctrl_out_data[ctrl_var] = data + np.random.uniform(
                low=-magnitude, high=magnitude, size=data.shape)
        perturbed_params = DiscreteControlPolicyParams(
            ambient_support_data=self.ambient_support_data,
            control_out_data=DataTable(perturbed_ctrl_out_data))
        return DiscreteControlPolicy(policy_name="perturbed_policy",
                                     policy_params=perturbed_params)