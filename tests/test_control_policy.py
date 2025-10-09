import pathlib
import pytest
import numpy as np
from twain_wifco.config import control_policy_from_json
from twain_wifco.interface import Ambient, Control

    
def test_discrete_control_policy():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "discrete_control_policy.json"
    simple_control_policy = control_policy_from_json(json_path=json_path)
    
    ambient_conditions_support = np.array([[ 2, 120, 30 ],
                                           [ 5,  60, 20 ],
                                           [ 5, 180, 20 ],
                                           [ 5, 240, 10 ],
                                           [ 7,   0, 10 ],
                                           [ 7, 300,  5 ],
                                           ])
    
    # Initialization
    assert simple_control_policy.component_name == "discrete_control_policy"
    assert simple_control_policy.ambient_variables == \
        [Ambient.ELECTRICITY_PRICE, Ambient.WIND_DIRECTION, Ambient.WIND_SPEED]
    assert simple_control_policy.ambient_conditions_support == \
        pytest.approx(ambient_conditions_support)
    assert simple_control_policy.control_inputs == \
        [Control.POWER_REGULATION, Control.YAW_STEERING]

    # retrieve control setpoints for dicrete ambient conditions
    # Invalid ambient condition keys
    ambient_condition = {Ambient.WIND_SPEED: 1,
                         Ambient.WIND_DIRECTION: 0}
    with pytest.raises(ValueError) as excinfo: 
        simple_control_policy.get_control_setpoints(ambient_condition)
    assert "Insufficient input variables" in str(excinfo.value)

    # Invalid ambient condition values
    ambient_condition = {Ambient.WIND_SPEED: 1,
                         Ambient.WIND_DIRECTION: 0,
                         Ambient.ELECTRICITY_PRICE: 10}
    
    # Valid ambient condition
    ambient_condition = {Ambient.WIND_SPEED: 20,
                         Ambient.WIND_DIRECTION: 180,
                         Ambient.ELECTRICITY_PRICE: 5}
    control_setpoints = simple_control_policy.get_control_setpoints(ambient_condition)
    assert control_setpoints == pytest.approx({Control.POWER_REGULATION: 4,
                                               Control.YAW_STEERING: 8})

    # Comparison
    json_path = test_data_folder / "discrete_control_policy_perm.json"
    simple_control_policy_permuted = control_policy_from_json(json_path=json_path)
    assert simple_control_policy_permuted == simple_control_policy
