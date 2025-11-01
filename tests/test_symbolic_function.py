import pathlib
import pytest
import numpy as np
from twain_wifco.config import symbolic_function_from_json
from twain_wifco.interface import (
    DataTable,
    ModelOutput,
    Control,
    Ambient)

def test_symbolic_function():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "symbolic_function.jsonc"
    symbolic_function = symbolic_function_from_json(json_path=json_path)
    
    met_condition = DataTable({Ambient.WIND_SPEED: np.array([20,
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