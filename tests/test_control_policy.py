import pathlib
import pytest
import numpy as np
from twain_wiffco.config import parse_json_file, control_policy_from_dict
from twain_wiffco.ambient_conditions import AmbientVariable
from twain_wiffco.control_input import ControlVariable

    
def test_discrete_control_policy():
    json_path = pathlib.Path(__file__).parent / "data" / "simple_control_policy.json"
    param_dict = parse_json_file(path=json_path)
    simple_control_policy = control_policy_from_dict(param_dict=param_dict)
    
    ambient_conditions_support = np.array([
            [ 10,   20,   30,   20,   10,    5],
            [  0,   60,  120,  180,  240,  300],
            [ 10,    5,    2,    5,    5,   10]
        ])
    
    # Initialization
    assert simple_control_policy.name == "simple_control_policy"
    assert simple_control_policy.params.ambient_variables == \
        [AmbientVariable.WIND_SPEED, AmbientVariable.WIND_DIRECTION, AmbientVariable.ELECTRICITY_PRICE]
    assert simple_control_policy.params.ambient_conditions_support == \
        pytest.approx(ambient_conditions_support)
    assert simple_control_policy.params.control_inputs == \
        [ControlVariable.POWER_REGULATION, ControlVariable.YAW_STEERING]

    # Set control policy
    # Invalid input
    invalid_setpoints = np.array([[1, 2],
                                  [3, 4]])
    with pytest.raises(ValueError) as excinfo: 
        simple_control_policy.set_control_policy(control_setpoints=invalid_setpoints)
    assert "Control setpoints dimensions mismatch" in str(excinfo.value)

    # Valid input
    valid_setpoints = np.array([[1, 2, 3,  4,  5,  6],
                                [7, 8, 9, 10, 11, 12]])
    simple_control_policy.set_control_policy(control_setpoints=valid_setpoints)

    # retrieve control setpoints for dicrete ambient conditions
    # Invalid ambient condition keys
    ambient_condition = {AmbientVariable.WIND_SPEED: 1,
                         AmbientVariable.WIND_DIRECTION: 0}
    with pytest.raises(ValueError) as excinfo: 
        simple_control_policy.get_control_setpoints(ambient_condition)
    assert "Insufficient ambient condition" in str(excinfo.value)

    # Invalid ambient condition values
    ambient_condition = {AmbientVariable.WIND_SPEED: 1,
                         AmbientVariable.WIND_DIRECTION: 0,
                         AmbientVariable.ELECTRICITY_PRICE: 10}
    with pytest.raises(ValueError) as excinfo: 
        simple_control_policy.get_control_setpoints(ambient_condition)
    assert "Ambient condition not found" in str(excinfo.value)

    # Valid ambient condition
    ambient_condition = {AmbientVariable.WIND_SPEED: 20,
                         AmbientVariable.WIND_DIRECTION: 180,
                         AmbientVariable.ELECTRICITY_PRICE: 5}
    control_setpoints = simple_control_policy.get_control_setpoints(ambient_condition)
    assert control_setpoints == pytest.approx(np.array([4, 10]))

    


        
    

    