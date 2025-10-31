import pathlib
import pytest
import numpy as np
from twain_wifco.config import plant_model_from_json
from twain_wifco.interface import (
    Ambient,
    Control,
    ModelOutput,
    DataTable)

test_data_folder = pathlib.Path(__file__).parent / "data"

def test_scattered_plant_model():
    json_path = test_data_folder / "model_scattered.jsonc"
    power_damage_scattered_model = plant_model_from_json(json_path=json_path)
            
    # Invalid input
    valid_met_condition = DataTable({Ambient.WIND_SPEED: np.array(20)})
    invalid_ctrl_input =  DataTable({Control.YAW_ANGLE: np.array(2)})
    with pytest.raises(ValueError) as excinfo: 
        power_damage_scattered_model.evaluate(meteorological_condition=valid_met_condition,
                                        control_input=invalid_ctrl_input)
    assert "missing required variables of type Control" in str(excinfo.value)

    # Valid input
    valid_ctrl_input = DataTable({Control.POWER_REGULATION: np.array(2)})
    expected_output =  DataTable({ModelOutput.DAMAGE_RATE:  np.array(4 * 4),
                                  ModelOutput.ELECTRICAL_POWER: np.array(4 * 1.4)})

    output = power_damage_scattered_model.evaluate(meteorological_condition=valid_met_condition,
                                                   control_input=valid_ctrl_input)
    assert output == expected_output

def test_symbolic_model():
    json_path = test_data_folder / "model_symbolic.jsonc"
    power_damage_sybolic_model = plant_model_from_json(json_path=json_path)
            
    # Invalid input
    valid_met_condition = DataTable({Ambient.WIND_SPEED: np.array(20)})
    invalid_ctrl_input =  DataTable({Control.YAW_ANGLE: np.array(2)})
    with pytest.raises(ValueError) as excinfo: 
        power_damage_sybolic_model.evaluate(meteorological_condition=valid_met_condition,
                                        control_input=invalid_ctrl_input)
    assert "missing required variables of type Control" in str(excinfo.value)

    # Valid input
    valid_ctrl_input = DataTable({Control.POWER_REGULATION: np.array([2, 2])})
    expected_output =  DataTable({ModelOutput.DAMAGE_RATE: 4 * 4 * np.array([1, 1]),
                                  ModelOutput.ELECTRICAL_POWER: 4 * np.sqrt(2) * np.array([1, 1])})

    output = power_damage_sybolic_model.evaluate(meteorological_condition=valid_met_condition,
                                                   control_input=valid_ctrl_input)
    assert output == expected_output

def test_floris_model():
    json_path = test_data_folder / "model_floris.jsonc"
    floris_power_model = plant_model_from_json(json_path=json_path)
    
       
    # Invalid input
    valid_met_condition = DataTable({Ambient.WIND_SPEED: np.array(20.),
                                     Ambient.WIND_DIRECTION: np.array(270.),
                                     Ambient.TURBULENCE_INTENSITY: np.array(0.)})
    invalid_ctrl_input =  DataTable({Control.POWER_REGULATION: np.array(2.)})
    with pytest.raises(ValueError) as excinfo: 
        floris_power_model.evaluate(meteorological_condition=valid_met_condition,
                                        control_input=invalid_ctrl_input)
    assert "missing required variables of type Control" in str(excinfo.value)

    # Valid input
    valid_ctrl_input = DataTable({Control.YAW_ANGLE: np.array([0., 0., 0., 0.])})
    output = floris_power_model.evaluate(meteorological_condition=valid_met_condition,
                                         control_input=valid_ctrl_input)
    assert output == DataTable({ModelOutput.ELECTRICAL_POWER: np.full(shape=(4,), fill_value=5e6)})
