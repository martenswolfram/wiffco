from typing import Dict, Any, Tuple
import numpy as np

from twain_wifco.interface import (
    DataTable,
    DataTable,
    Component,
    Ambient,
    Control,
    Interface,
    get_default_value
)

# ======================================================================
# Discrete Control Policy
# ======================================================================

class DiscreteControlPolicy(Component):
    
    def __init__(self,
                 name: str,
                 ambient_support_data: DataTable[Ambient],
                 control_out_data: DataTable[Control]):
        
        self.component_name = name
        self._ambient_support_data = ambient_support_data
        self._control_out_data = control_out_data
        self.input_interface = Interface(
            all_shapes={Ambient: self._ambient_support_data.shapes()})
        self.output_interface = Interface(
            all_shapes={
                Control: self._control_out_data.shapes()})
    
    def control_out_data(self):
        return self._control_out_data

    @Component.with_validation
    def get_control_setpoints(
        self, ambient_condition: DataTable[Ambient]
    ) -> DataTable[Control]:
        point_index = self._ambient_support_data.find_matching_point(data_point=ambient_condition)
        return self._control_out_data.get_point(point_index)

    def random_perturbation(self,
                            scale: float = 1.0,
                            discrete_steps: Dict[Control, float] = None) -> "DiscreteControlPolicy":
        
        perturbed_ctrl_out_data = {}
        for ctrl_var, data in self._control_out_data.items():
            if discrete_steps is None:
                magnitude = scale * self._control_out_data.abs_tol(ctrl_var)
                perturbed_ctrl_out_data[ctrl_var] = data + np.random.uniform(
                    low=-magnitude, high=magnitude, size=data.shape
                )
            else:
                perturbed_ctrl_out_data[ctrl_var] = data + np.random.choice(
                    a=[-discrete_steps[ctrl_var], 0, discrete_steps[ctrl_var]],
                    p=[1/3, 1/3, 1/3],
                    size=data.shape
                )

        return DiscreteControlPolicy(
            name=f"{self.component_name}_perturbed",
            ambient_support_data=self._ambient_support_data,
            control_out_data=DataTable(perturbed_ctrl_out_data)
        )

    def repr_details(self):
        out = (f"Ambient support:\n{self._ambient_support_data}"
               f"Control setpoints:\n{self._control_out_data}")
        return out


def discrete_control_policy_from_dict(
    param_dict: Dict[str, Any]
) -> DiscreteControlPolicy:
    """Construct `DiscreteControlPolicy` from a configuration dictionary.

    The dictionary is expected to contain numeric data for each ambient and control variable:
        {
            "ambient_support_data": { "wind_speed": [...], "wind_direction": [...] },
            "control_out_data": { "power_regulation": [...], "yaw_angle": [...] }
        }

    Args:
        param_dict: Configuration dictionary (e.g., loaded from JSON).

    Returns:
        DiscreteControlPolicy: Discrete conytrol policy object.
    """
    name = param_dict["name"]
    
    ambient_support_data = DataTable(
        {Ambient(var): np.array(data) for var, data in param_dict["ambient_support_data"].items()}
    )
    control_out_data = DataTable(
        {Control(var): np.array(data) for var, data in param_dict["control_out_data"].items()}
    )

    return DiscreteControlPolicy(
        name=name,
        ambient_support_data=ambient_support_data,
        control_out_data=control_out_data,
    )

def default_discrete_policy(
    control_shapes: Dict[Control, Tuple[int, ...]],
    ambient_support: DataTable[Ambient]):
    num_samples = len(ambient_support)
    control_setpoints = DataTable({
        ctrl_var: get_default_value(data_var=ctrl_var,
                                    shape=(num_samples,) + shape) for \
        ctrl_var, shape in control_shapes.items()})
    return DiscreteControlPolicy(
        name="discrete_control_policy",
        ambient_support_data=ambient_support,
        control_out_data=control_setpoints)