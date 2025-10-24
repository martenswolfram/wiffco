from typing import Dict, Any
import numpy as np

from twain_wifco.interface import (
    DataPoint,
    DataTable,
    Component,
    ComponentParams,
    Ambient,
    Control,
    Interface,
)


# ======================================================================
# Parameter Class
# ======================================================================

class DiscreteControlPolicyParams(ComponentParams):
    """Defines parameters for a discrete control policy.

    A discrete control policy maps ambient conditions to corresponding
    control setpoints using lookup tables stored as `DataTable` instances.
    """

    def __init__(
        self,
        ambient_support_data: DataTable[Ambient],
        control_out_data: DataTable[Control],
    ):
        """Initialize policy parameters.

        Args:
            ambient_support_data: Data table of ambient condition points.
            control_out_data: Data table of corresponding control setpoints.
        """
        self.ambient_support_data = ambient_support_data
        self.control_out_data = control_out_data

    def input_interface(self) -> Interface:
        """Define required ambient input variables."""
        return Interface(all_shapes={Ambient: self.ambient_support_data.shapes()})

    def output_interface(self) -> Interface:
        """Define output control variables."""
        return Interface(all_shapes={Control: self.control_out_data.shapes()})


def discrete_control_policy_params_from_dict(
    param_dict: Dict[str, Any]
) -> DiscreteControlPolicyParams:
    """Construct `DiscreteControlPolicyParams` from a configuration dictionary.

    The dictionary is expected to contain numeric data for each ambient and control variable:
        {
            "ambient_support_data": { "wind_speed": [...], "wind_direction": [...] },
            "control_out_data": { "power_regulation": [...], "yaw_steering": [...] }
        }

    Args:
        param_dict: Configuration dictionary (e.g., loaded from JSON).

    Returns:
        DiscreteControlPolicyParams: Parsed parameter object.
    """
    ambient_support_data = DataTable(
        {Ambient(var): np.array(data) for var, data in param_dict["ambient_support_data"].items()}
    )
    control_out_data = DataTable(
        {Control(var): np.array(data) for var, data in param_dict["control_out_data"].items()}
    )

    return DiscreteControlPolicyParams(
        ambient_support_data=ambient_support_data,
        control_out_data=control_out_data,
    )


# ======================================================================
# Discrete Control Policy
# ======================================================================

class DiscreteControlPolicy(Component):
    """Discrete mapping from ambient conditions to control setpoints.

    This component implements a lookup policy where each ambient condition
    corresponds to one control setpoint entry from pre-defined tables.
    """

    def __init__(self, name: str, params: DiscreteControlPolicyParams):
        """Initialize a discrete control policy.

        Args:
            name: Name of the control policy.
            params: Parameter object defining ambient and control data tables.
        """
        super().__init__(component_name=name, component_params=params)
        self.ambient_support_data = params.ambient_support_data
        self.control_out_data = params.control_out_data

    def get_control_setpoints(
        self, ambient_condition: DataPoint[Ambient]
    ) -> DataPoint[Control]:
        """Return control setpoints corresponding to given ambient condition.

        Args:
            ambient_condition: Ambient condition data point.

        Returns:
            DataPoint[Control]: Control setpoint for the matching condition.

        Raises:
            ValueError: If the given ambient condition does not match any support point.
        """
        self.validate_inputs(input_data={Ambient: ambient_condition})
        point_index = self.ambient_support_data.find_matching_point(data_point=ambient_condition)
        return self.control_out_data.get_point(point_index)

    def set_control_data(self, control_data_vector: np.ndarray) -> None:
        """Update the control table from a flat vector.

        Args:
            control_data_vector: Flattened array of control values.
        """
        self.control_out_data.update_from_vector(control_data_vector)

    def random_perturbation(self, scale: float = 1.0) -> "DiscreteControlPolicy":
        """Create a perturbed copy of this control policy.

        Adds uniform random perturbations scaled by each variable's absolute tolerance.

        Args:
            scale: Multiplier for the perturbation magnitude (default: 1.0).

        Returns:
            DiscreteControlPolicy: New perturbed control policy instance.
        """
        perturbed_ctrl_out_data = {}
        for ctrl_var, data in self.control_out_data.items():
            magnitude = scale * self.control_out_data.abs_tol(ctrl_var)
            perturbed_ctrl_out_data[ctrl_var] = data + np.random.uniform(
                low=-magnitude, high=magnitude, size=data.shape
            )

        perturbed_params = DiscreteControlPolicyParams(
            ambient_support_data=self.ambient_support_data,
            control_out_data=DataTable(perturbed_ctrl_out_data),
        )

        return DiscreteControlPolicy(
            name=f"{self.component_name}_perturbed", params=perturbed_params
        )
