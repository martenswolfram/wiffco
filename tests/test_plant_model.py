import pathlib
import pytest
import numpy as np
from twain_wifco.config import plant_model_from_json
from twain_wifco.interface import (
    Ambient,
    Control,
    ModelOutput,
    DataPoint)
    
def test_plant_model():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "power_damage_scattered_model.json"
    power_damage_rbf_model = plant_model_from_json(json_path=json_path)
    
        
    # Invalid input
    valid_met_condition = DataPoint({Ambient.WIND_SPEED: np.array([20])})
    invalid_ctrl_input =  DataPoint({Control.YAW_STEERING: np.array([2])})
    with pytest.raises(ValueError) as excinfo: 
        power_damage_rbf_model.evaluate(meteorological_condition=valid_met_condition,
                                        control_input=invalid_ctrl_input)
    assert "Insufficient input variables" in str(excinfo.value)

    # Valid input
    valid_ctrl_input = DataPoint({Control.POWER_REGULATION: np.array([2])})
    expected_output =  DataPoint({ModelOutput.ELECTRICAL_POWER: np.array([4 * 1.4]),
                                  ModelOutput.DAMAGE_RATE: np.array([4 * 8])})

    output = power_damage_rbf_model.evaluate(meteorological_condition=valid_met_condition,
                                             control_input=valid_ctrl_input)
    assert output == expected_output
