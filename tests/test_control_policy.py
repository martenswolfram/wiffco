import pathlib
import pytest
import numpy as np
from twain_wifco.config import parse_json_file
from twain_wifco.control_policy import discrete_control_policy_from_dict
from twain_wifco.interface import (
    Ambient,
    Control,
    DataTable)

    
def test_discrete_control_policy():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "discrete_control_policy.jsonc"
    param_dict = parse_json_file(path=json_path)
    simple_control_policy = discrete_control_policy_from_dict(
        param_dict=param_dict)
        
    # retrieve control setpoints for dicrete ambient conditions
    # Invalid ambient condition keys
    invalid_amb_conditions = DataTable({Ambient.WIND_SPEED_MPS: np.array([20, 10])})
    
    with pytest.raises(ValueError) as excinfo:
        simple_control_policy.get_control(ambient=invalid_amb_conditions)
        assert "missing required variables of type Ambient" in str(excinfo.value)

    # Valid ambient condition
    valid_ambient_conditions = DataTable(
        {Ambient.WIND_SPEED_MPS: np.array([20, 20]),
         Ambient.WIND_DIRECTION_DEG: np.array([180, 60]),
         Ambient.ELECTRICITY_PRICE_EPKWH: np.array([5, 5])})
    expected_output =  DataTable({Control.POWER_REGULATION: np.array([4, 2]),
                                  Control.YAW_ANGLE: np.array([8, 6])})

    control_setpoints = simple_control_policy.get_control(ambient=valid_ambient_conditions)
    assert control_setpoints == expected_output