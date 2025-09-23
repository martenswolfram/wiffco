import pathlib
import pytest
from twain_wifco.config import parse_json_file, wind_farm_model_from_dict
from twain_wifco.interface import (
    AmbientVariable,
    ControlVariable,
    OutputVariable)

    
def test_simple_power_model():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "simple_power_model.json"
    param_dict = parse_json_file(path=json_path)
    simple_power_model = wind_farm_model_from_dict(param_dict=param_dict)
    
    # Initialization
    assert simple_power_model.interface.name == "Model 'simple_power_model'"
    assert simple_power_model.control_input_models.keys() == set([ControlVariable.POWER_REGULATION])
    assert simple_power_model.meteorological_condition_models.keys() == set([AmbientVariable.WIND_SPEED])
    assert simple_power_model.single_output == OutputVariable.ELECTRICAL_POWER

    # Invalid input
    valid_met_condition = {AmbientVariable.WIND_SPEED: 20}
    invalid_ctrl_input = {ControlVariable.YAW_STEERING: 2}
    with pytest.raises(ValueError) as excinfo: 
        simple_power_model.evaluate(meteorological_condition=valid_met_condition,
                                    control_input=invalid_ctrl_input)
    assert "Insufficient Control Input" in str(excinfo.value)

    # Valid input
    valid_ctrl_input = {ControlVariable.POWER_REGULATION: 2}
    expected_output = {OutputVariable.ELECTRICAL_POWER: 4 * 1.4}

    output = simple_power_model.evaluate(meteorological_condition=valid_met_condition,
                                         control_input=valid_ctrl_input)
    assert output == pytest.approx(expected=expected_output)
    
