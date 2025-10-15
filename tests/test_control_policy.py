import pathlib
import pytest
import numpy as np
from twain_wifco.config import control_policy_from_json
from twain_wifco.interface import Ambient, Control

    
def test_discrete_control_policy():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "discrete_control_policy.json"
    simple_control_policy = control_policy_from_json(json_path=json_path)
    
    ambient_support_data = np.array([[ 10,    0,  7],
                                     [ 20,   60,  5],
                                     [ 30,  120,  2],
                                     [ 20,  180,  5],
                                     [ 10,  240,  5],
                                     [  5,  300,  7]])
    control_setpoint_data = np.array([[1, 5],
                                      [2, 6],
                                      [3, 7],
                                      [4, 8],
                                      [3, 9],
                                      [2, 0]])
    
    # Initialization
    assert simple_control_policy.component_name == "discrete_control_policy"
    assert simple_control_policy.ambient_variables == \
        [Ambient.WIND_SPEED, Ambient.WIND_DIRECTION, Ambient.ELECTRICITY_PRICE]
    assert simple_control_policy.control_variables == \
        [Control.POWER_REGULATION, Control.YAW_STEERING]
    assert simple_control_policy.ambient_support_data == \
        pytest.approx(ambient_support_data)
    assert simple_control_policy.control_setpoint_data == \
        pytest.approx(control_setpoint_data)
    
    # retrieve control setpoints for dicrete ambient conditions
    # Invalid ambient condition keys
    ambient_condition = {Ambient.WIND_SPEED: np.array([1]),
                         Ambient.WIND_DIRECTION: np.array([0])}
    with pytest.raises(ValueError) as excinfo:
        simple_control_policy.get_control_setpoints(ambient_condition)
    assert "Insufficient input variables" in str(excinfo.value)

    # Valid ambient condition
    ambient_condition = {Ambient.WIND_SPEED: np.array([20]),
                         Ambient.WIND_DIRECTION: np.array([180]),
                         Ambient.ELECTRICITY_PRICE: np.array([5])}
    control_setpoints = simple_control_policy.get_control_setpoints(ambient_condition)
    assert control_setpoints == pytest.approx({Control.POWER_REGULATION: np.array([4]),
                                               Control.YAW_STEERING: np.array([8])})
