import pathlib
import pytest
from typing import Dict
import numpy as np
from twain_wifco.config import statistics_from_json
from twain_wifco.interface import Ambient

def support_points_equal(support_points_1: Dict[Ambient, np.ndarray],
                         support_points_2: Dict[Ambient, np.ndarray]):
    for var in support_points_1.keys():
        if not support_points_2[var] == \
            pytest.approx(support_points_1[var]):
            return False
    return True
    

def test_statistics():
    test_data_folder = pathlib.Path(__file__).parent / "data"
    json_path = test_data_folder / "discrete_ambient_statistics.json"
    discrete_ambient_statistics = statistics_from_json(json_path=json_path)
    
    ordered_prevalence = np.array([0.4, 0.25, 0.19, 0.1, 0.05, 0.01])
    ordered_support_points = {
            Ambient.WIND_SPEED: np.array([[ 20], 
                                          [ 10], 
                                          [ 30], 
                                          [ 20], 
                                          [  5], 
                                          [ 10]]),
            Ambient.WIND_DIRECTION: np.array([[180], 
                                              [240], 
                                              [120], 
                                              [ 60], 
                                              [300], 
                                              [  0]]),
            Ambient.ELECTRICITY_PRICE: np.array([[ 5],
                                                 [ 5],
                                                 [ 2],
                                                 [ 5],
                                                 [ 7],
                                                 [ 7]])
    }
    
    # Initialization
    assert discrete_ambient_statistics.component_name == "discrete_ambient_statistics"
    assert discrete_ambient_statistics.ordered_prevalence == pytest.approx(ordered_prevalence)
    assert support_points_equal(ordered_support_points,
                                discrete_ambient_statistics.ordered_support_points)
    
    # Sample without N specifed
    sys_sample_default = discrete_ambient_statistics.systematic_sample()
    assert sys_sample_default.normalized_weights == pytest.approx(ordered_prevalence)
    assert support_points_equal(ordered_support_points,
                                sys_sample_default.support_points)
    assert sys_sample_default.probability_covered == pytest.approx(1)

    # Invalid N
    with pytest.raises(ValueError) as excinfo: 
        discrete_ambient_statistics.systematic_sample(N_max=0)
    assert "Invalid number of samples" in str(excinfo.value)
    
    # Sample with larger N specifed
    N = 100
    sys_sample_larger = discrete_ambient_statistics.systematic_sample(N_max=N)
    assert sys_sample_larger.normalized_weights == pytest.approx(ordered_prevalence)
    assert support_points_equal(ordered_support_points,
                                sys_sample_larger.support_points)
    assert sys_sample_larger.probability_covered == pytest.approx(1)

    # Sample with smaller N specifed
    N = 4
    sys_sample_smaller = discrete_ambient_statistics.systematic_sample(N_max=N)
    probability_covered = np.sum(ordered_prevalence[:N])
    assert sys_sample_smaller.normalized_weights == pytest.approx(ordered_prevalence[:N] / probability_covered)
    assert support_points_equal({var: supp[:N, :] for var, supp in ordered_support_points.items()},
                                sys_sample_smaller.support_points)
    assert sys_sample_smaller.probability_covered == pytest.approx(probability_covered)
    
    # Expected value
    computed = discrete_ambient_statistics.expected_value()
    for var, supp in ordered_support_points.items():
        expected_value = ordered_prevalence @ supp
        assert expected_value == pytest.approx(computed[var])