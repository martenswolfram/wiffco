import pathlib
import pytest
import numpy as np
from twain_wifco.config import control_policy_from_json
from twain_wifco.interface import (
    Ambient,
    Control,
    DataPoint)

    
def test_discrete_control_policy():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "scattered_interp_policy.json"
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
    assert simple_control_policy.component_name == "interp_control_policy"
    
    # retrieve control setpoints for dicrete ambient conditions
    # Invalid ambient condition keys
    invalid_amb_condition = DataPoint({Ambient.WIND_SPEED: np.array([20])})
    
    with pytest.raises(ValueError) as excinfo:
        simple_control_policy.get_control_setpoints(invalid_amb_condition)
    assert "Insufficient input variables" in str(excinfo.value)

    # Valid ambient condition
    valid_ambient_condition = DataPoint(
        {Ambient.WIND_SPEED: np.array([20]),
         Ambient.WIND_DIRECTION: np.array([180]),
         Ambient.ELECTRICITY_PRICE: np.array([5])})
    expected_output =  DataPoint({Control.POWER_REGULATION: np.array([4]),
                                  Control.YAW_STEERING: np.array([8])})

    control_setpoints = simple_control_policy.get_control_setpoints(valid_ambient_condition)
    assert control_setpoints == expected_output