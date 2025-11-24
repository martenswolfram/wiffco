import pathlib
import pytest
import numpy as np
from twain_wifco.config import parse_json_file
from twain_wifco.symbolic import symbolic_function_from_dict
from twain_wifco.interface import (
    DataTable,
    ModelOutput,
    Control,
    Ambient)

def test_symbolic_function():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "symbolic_function.jsonc"
    param_dict = parse_json_file(path=json_path)
    symbolic_function = symbolic_function_from_dict(param_dict=param_dict)
    
    met_condition = DataTable({Ambient.WIND_SPEED_MPS: np.array([20,
                                                             10,
                                                             20])})
    ctrl_input = DataTable({Control.POWER_REGULATION: np.array([[2, 2],
                                                                [3, 3],
                                                                [2, 2]])})

    result = symbolic_function.evaluate({
        Ambient: met_condition,
        Control: ctrl_input
    })

    expected_result = DataTable(
        data={
            ModelOutput.ELECTRICAL_POWER: np.array([
                [2**0.5 * (20 / 10)**2, 2**0.5 * (20 / 10)**2],
                [3**0.5 * (10 / 10)**2, 3**0.5 * (10 / 10)**2],
                [2**0.5 * (20 / 10)**2, 2**0.5 * (20 / 10)**2]]),
            ModelOutput.DAMAGE_RATE: np.array([
                [2**2 * (20 / 10)**2, 2**2 * (20 / 10)**2],
                [3**2 * (10 / 10)**2, 3**2 * (10 / 10)**2],
                [2**2 * (20 / 10)**2, 2**2 * (20 / 10)**2]])
        }
    )
    assert result[ModelOutput] == expected_result
    pass