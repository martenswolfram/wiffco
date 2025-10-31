import pathlib
import pytest
import numpy as np
from twain_wifco.config import control_policy_from_json
from twain_wifco.interface import (
    Ambient,
    Control,
    DataTable)

    
def test_discrete_control_policy():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "discrete_control_policy.jsonc"
    simple_control_policy = control_policy_from_json(json_path=json_path)
        
    # retrieve control setpoints for dicrete ambient conditions
    # Invalid ambient condition keys
    invalid_amb_condition = DataTable({Ambient.WIND_SPEED: np.array(20)})
    
    with pytest.raises(ValueError) as excinfo:
        simple_control_policy.get_control_setpoints(ambient_condition=invalid_amb_condition)
        assert "missing required variables of type Ambient" in str(excinfo.value)

    # Valid ambient condition
    valid_ambient_condition = DataTable(
        {Ambient.WIND_SPEED: np.array(20),
         Ambient.WIND_DIRECTION: np.array(180),
         Ambient.ELECTRICITY_PRICE: np.array(5)})
    expected_output =  DataTable({Control.POWER_REGULATION: np.array(4),
                                  Control.YAW_ANGLE: np.array(8)})

    control_setpoints = simple_control_policy.get_control_setpoints(ambient_condition=valid_ambient_condition)
    assert control_setpoints == expected_output