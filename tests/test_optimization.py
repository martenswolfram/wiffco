import pathlib
import pytest
import numpy as np
from twain_wifco.config import (
    control_evaluation_system_from_json,
    control_optimization_from_json)
from twain_wifco.interface import (
    Control)

test_data_folder = pathlib.Path(__file__).parent / "data"
    
def test_control_evaluation_system():
    json_path = test_data_folder / "control_evaluation_system.json"
    control_evaluation_system = control_evaluation_system_from_json(json_path=json_path)
    assert control_evaluation_system.name == "discrete_evaluation_system"

def test_grid_search():
    json_path = test_data_folder / "grid_search_optimization.json"
    grid_search = control_optimization_from_json(json_path=json_path)
    
    # Initialization
    assert grid_search.optimization_name == "grid_search"
    assert grid_search.control_setpoints.keys() == set([Control.POWER_REGULATION])
    assert grid_search.control_setpoints[Control.POWER_REGULATION] == pytest.approx(np.array([0, 1, 2, 3, 4]))
    assert grid_search.num_ambient_conditions == 5
    
