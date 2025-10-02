import pathlib
import pytest
import numpy as np
from twain_wifco.config import parse_json_file, ambient_statistics_from_dict
from twain_wifco.interface import Ambient

def test_statistics():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "discrete_ambient_statistics.json"
    param_dict = parse_json_file(path=json_path)
    discrete_ambient_statistics = ambient_statistics_from_dict(param_dict=param_dict)
    
    ordered_prevalence = np.array([0.4, 0.25, 0.19, 0.1, 0.05, 0.01])
    ordered_support_points = np.array([[   20,   10,  30,  20,   5,  10],
                                       [  180,  240, 120,  60, 300,   0],
                                       [    5,    5,   2,   5,  10,  10]])
    # Initialization
    assert discrete_ambient_statistics.component_name == "discrete_ambient_statistics"
    assert discrete_ambient_statistics.support_variables == \
        [Ambient.WIND_SPEED, Ambient.WIND_DIRECTION, Ambient.ELECTRICITY_PRICE]
    assert discrete_ambient_statistics.ordered_prevalence == pytest.approx(ordered_prevalence)
    assert discrete_ambient_statistics.ordered_support_points == pytest.approx(ordered_support_points)
    
    # Sample without N specifed
    sys_sample_default = discrete_ambient_statistics.systematic_sample()
    assert sys_sample_default.support_variables == discrete_ambient_statistics.support_variables
    assert sys_sample_default.normalized_weights == pytest.approx(ordered_prevalence)
    assert sys_sample_default.support_values == pytest.approx(ordered_support_points)
    assert sys_sample_default.probability_covered == pytest.approx(1)

    # Invalid N
    with pytest.raises(ValueError) as excinfo: 
        discrete_ambient_statistics.systematic_sample(N=0)
    assert "Invalid number of samples" in str(excinfo.value)
    
    # Sample with larger N specifed
    N = 100
    sys_sample_larger = discrete_ambient_statistics.systematic_sample(N=N)
    assert sys_sample_larger.support_variables == discrete_ambient_statistics.support_variables
    assert sys_sample_larger.normalized_weights == pytest.approx(ordered_prevalence)
    assert sys_sample_larger.support_values == pytest.approx(ordered_support_points)
    assert sys_sample_larger.probability_covered == pytest.approx(1)

    # Sample with smaller N specifed
    N = 4
    sys_sample_smaller = discrete_ambient_statistics.systematic_sample(N=N)
    probability_covered = np.sum(ordered_prevalence[:N])
    assert sys_sample_smaller.support_variables == discrete_ambient_statistics.support_variables
    assert sys_sample_smaller.normalized_weights == pytest.approx(ordered_prevalence[:N] / probability_covered)
    assert sys_sample_smaller.support_values == pytest.approx(ordered_support_points[:, :N])
    assert sys_sample_smaller.probability_covered == pytest.approx(probability_covered)
    
    # Expected value
    expected_vector = ordered_support_points @ ordered_prevalence
    expected_value = {Ambient.WIND_SPEED: expected_vector[0],
                Ambient.WIND_DIRECTION: expected_vector[1],
                Ambient.ELECTRICITY_PRICE: expected_vector[2]}
    assert discrete_ambient_statistics.expected_value() == pytest.approx(expected_value)

