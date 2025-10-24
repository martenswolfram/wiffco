import pathlib
import pytest
import numpy as np
from twain_wifco.config import plant_model_from_json
from twain_wifco.interface import (
    Ambient,
    Control,
    ModelOutput,
    DataPoint)

test_data_folder = pathlib.Path(__file__).parent / "data"

def test_scattered_plant_model():
    json_path = test_data_folder / "model_scattered.json"
    power_damage_scattered_model = plant_model_from_json(json_path=json_path)
    
    # Initialization
    assert power_damage_scattered_model.component_name == "power_damage_scattered"    
        
    # Invalid input
    valid_met_condition = DataPoint({Ambient.WIND_SPEED: np.array([20])})
    invalid_ctrl_input =  DataPoint({Control.YAW_STEERING: np.array([2])})
    with pytest.raises(ValueError) as excinfo: 
        power_damage_scattered_model.evaluate(meteorological_condition=valid_met_condition,
                                        control_input=invalid_ctrl_input)
    assert "missing required variables of type Control" in str(excinfo.value)

    # Valid input
    valid_ctrl_input = DataPoint({Control.POWER_REGULATION: np.array([2])})
    expected_output =  DataPoint({ModelOutput.DAMAGE_RATE: np.array([4 * 4]),
                                  ModelOutput.ELECTRICAL_POWER: np.array([4 * 1.4])})

    output = power_damage_scattered_model.evaluate(meteorological_condition=valid_met_condition,
                                                   control_input=valid_ctrl_input)
    assert output == expected_output

def test_symbolic_model():
    json_path = test_data_folder / "model_symbolic.json"
    power_damage_sybolic_model = plant_model_from_json(json_path=json_path)
    
    # Initialization
    assert power_damage_sybolic_model.component_name == "power_damage_symbolic"    
        
    # Invalid input
    valid_met_condition = DataPoint({Ambient.WIND_SPEED: np.array([20])})
    invalid_ctrl_input =  DataPoint({Control.YAW_STEERING: np.array([2])})
    with pytest.raises(ValueError) as excinfo: 
        power_damage_sybolic_model.evaluate(meteorological_condition=valid_met_condition,
                                        control_input=invalid_ctrl_input)
    assert "missing required variables of type Control" in str(excinfo.value)

    # Valid input
    valid_ctrl_input = DataPoint({Control.POWER_REGULATION: np.array([2])})
    expected_output =  DataPoint({ModelOutput.DAMAGE_RATE: np.array([4 * 4]),
                                  ModelOutput.ELECTRICAL_POWER: np.array([4 * np.sqrt(2)])})

    output = power_damage_sybolic_model.evaluate(meteorological_condition=valid_met_condition,
                                                   control_input=valid_ctrl_input)
    assert output == expected_output
