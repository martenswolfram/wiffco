import pathlib
import pytest
import numpy as np
from twain_wifco.config import parse_json_file
from twain_wifco.plant_model import plant_model_from_dict
from twain_wifco.interface import (
    Ambient,
    Control,
    ModelOutput,
    DataTable)

test_data_folder = pathlib.Path(__file__).parent / "data"

def test_scattered_plant_model():
    json_path = test_data_folder / "model_scattered.jsonc"
    param_dict = parse_json_file(path=json_path)
    power_damage_scattered_model = plant_model_from_dict(param_dict=param_dict)
            
    # Invalid input
    valid_met_condition = DataTable({Ambient.WIND_SPEED: np.array([20, 10, 20])})
    invalid_ctrl_input =  DataTable({Control.YAW_ANGLE: np.array([2, 1, 2])})
    with pytest.raises(ValueError) as excinfo: 
        power_damage_scattered_model.evaluate(meteorological=valid_met_condition,
                                        control=invalid_ctrl_input)
    assert "missing required variables of type Control" in str(excinfo.value)

    # Valid input
    valid_ctrl_input = DataTable({Control.POWER_REGULATION: np.array([2, 3, 2])})
    expected_output =  DataTable({ModelOutput.DAMAGE_RATE:  np.array([4 * 4, 1 * 9, 4 * 4]),
                                  ModelOutput.ELECTRICAL_POWER: np.array([4 * 1.4, 1 * 1.7, 4 * 1.4])})

    output = power_damage_scattered_model.evaluate(meteorological=valid_met_condition,
                                                   control=valid_ctrl_input)
    assert output == expected_output

def test_symbolic_model():
    json_path = test_data_folder / "model_symbolic.jsonc"
    param_dict = parse_json_file(path=json_path)
    power_damage_symbolic_model = plant_model_from_dict(param_dict=param_dict)
            
    # Invalid input
    valid_met_condition = DataTable({Ambient.WIND_SPEED: np.array([20,
                                                                   10,
                                                                   20])})
    invalid_ctrl_input =  DataTable({Control.YAW_ANGLE: np.array([2,
                                                                  1,
                                                                  2])})
    with pytest.raises(ValueError) as excinfo: 
        power_damage_symbolic_model.evaluate(meteorological=valid_met_condition,
                                             control=invalid_ctrl_input)
    assert "missing required variables of type Control" in str(excinfo.value)

    # Valid input
    valid_ctrl_input = DataTable({Control.POWER_REGULATION: np.array([[2, 2],
                                                                      [3, 3],
                                                                      [2, 2]])})
    expected_output = DataTable(
        {
            ModelOutput.DAMAGE_RATE: np.array(
                [[2**2 * (20 / 10)**2, 2**2 * (20 / 10)**2],
                [3**2 * (10 / 10)**2, 3**2 * (10 / 10)**2],
                [2**2 * (20 / 10)**2, 2**2 * (20 / 10)**2]]),
            ModelOutput.ELECTRICAL_POWER: np.array(
                [[2**0.5 * (20 / 10)**2, 2**0.5 * (20 / 10)**2],
                [3**0.5 * (10 / 10)**2, 3**0.5 * (10 / 10)**2],
                [2**0.5 * (20 / 10)**2, 2**0.5 * (20 / 10)**2]])
        }
    )

    output = power_damage_symbolic_model.evaluate(meteorological=valid_met_condition,
                                                  control=valid_ctrl_input)
    assert output == expected_output

def test_floris_model():
    json_path = test_data_folder / "model_floris.jsonc"
    param_dict = parse_json_file(path=json_path)
    floris_power_model = plant_model_from_dict(param_dict=param_dict)
       
    # Invalid input
    valid_met_conditions = DataTable({Ambient.WIND_SPEED: np.array([20., 10.]),
                                      Ambient.WIND_DIRECTION: np.array([270., 30]),
                                      Ambient.TURBULENCE_INTENSITY: np.array([0., 0.])})
    invalid_ctrl_input =  DataTable({Control.POWER_REGULATION: np.array([2., 3.])})
    with pytest.raises(ValueError) as excinfo: 
        floris_power_model.evaluate(meteorological=valid_met_conditions,
                                    control=invalid_ctrl_input)
    assert "missing required variables of type Control" in str(excinfo.value)

    # Valid input
    valid_ctrl_input = DataTable({Control.YAW_ANGLE: np.array([[0., 0., 0., 0.],
                                                               [1., 1., 1., 1.]])})
    output = floris_power_model.evaluate(meteorological=valid_met_conditions,
                                         control=valid_ctrl_input)
    assert output.shapes()[ModelOutput.ELECTRICAL_POWER] == (4,)
    assert len(output) == 2
    assert np.all(output[ModelOutput.ELECTRICAL_POWER][0, ...] > output[ModelOutput.ELECTRICAL_POWER][1, ...])
